"""Neural models for TransitVet: AstroNet CNN and a patch Transformer with masked-patch pretraining."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def _tower(chans, pool_k, length):
    layers, c_in = [], 1
    for c in chans:
        layers += [nn.Conv1d(c_in, c, 5, padding=2), nn.ReLU(), nn.Conv1d(c, c, 5, padding=2), nn.ReLU(), nn.MaxPool1d(pool_k, 2)]
        c_in = c
    net = nn.Sequential(*layers, nn.Flatten())
    return net, net(torch.zeros(1, 1, length)).shape[1]

class AstroNet(nn.Module):
    """local_global config of google-research/exoplanet-ml (verified)."""
    def __init__(s):
        super().__init__()
        s.g, ng = _tower([16, 32, 64, 128, 256], 5, 2001); s.l, nl = _tower([16, 32], 7, 201)
        fc, c = [], ng + nl
        for _ in range(4): fc += [nn.Linear(c, 512), nn.ReLU()]; c = 512
        s.fc = nn.Sequential(*fc, nn.Linear(512, 1))
        for m in s.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(s, g, l): return s.fc(torch.cat([s.g(g[:, None]), s.l(l[:, None])], 1))[:, 0]

G_PAD, G_PATCH, L_PAD, L_PATCH = 2016, 32, 208, 8      # 63 global + 26 local patches

def patches(g, l):
    g = F.pad(g, (0, G_PAD - g.shape[1]), mode='replicate') if g.dim() == 2 else g
    l = F.pad(l, (0, L_PAD - l.shape[1]), mode='replicate')
    return g.reshape(g.shape[0], -1, G_PATCH), l.reshape(l.shape[0], -1, L_PATCH)

class PatchTransformer(nn.Module):
    def __init__(s, d=128, layers=4, heads=4, ff=256, drop=0.1):
        super().__init__()
        s.eg, s.el = nn.Linear(G_PATCH, d), nn.Linear(L_PATCH, d)
        n = G_PAD // G_PATCH + L_PAD // L_PATCH
        s.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)
        s.typ = nn.Parameter(torch.randn(2, d) * 0.02)
        s.cls = nn.Parameter(torch.zeros(1, 1, d)); s.mask_tok = nn.Parameter(torch.zeros(1, 1, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True), layers)
        s.norm = nn.LayerNorm(d); s.head = nn.Linear(d, 1)
        s.rec_g, s.rec_l = nn.Linear(d, G_PATCH), nn.Linear(d, L_PATCH)
        s.ng = G_PAD // G_PATCH
    def encode(s, g, l, mask=None):
        pg, pl = patches(g, l)
        x = torch.cat([s.eg(pg) + s.typ[0], s.el(pl) + s.typ[1]], 1)
        if mask is not None: x = torch.where(mask[..., None], s.mask_tok.expand_as(x), x)
        x = torch.cat([s.cls.expand(x.shape[0], -1, -1), x], 1) + s.pos
        return s.norm(s.enc(x)), pg, pl
    def forward(s, g, l): return s.head(s.encode(g, l)[0][:, 0])[:, 0]
    def pretrain_loss(s, g, l, ratio=0.4):
        n = s.pos.shape[1] - 1
        mask = torch.rand(g.shape[0], n) < ratio
        h, pg, pl = s.encode(g, l, mask)
        rec = torch.cat([s.rec_g(h[:, 1:1 + s.ng]), F.pad(s.rec_l(h[:, 1 + s.ng:]), (0, G_PATCH - L_PATCH))], 1)
        tgt = torch.cat([pg, F.pad(pl, (0, G_PATCH - L_PATCH))], 1)
        width = torch.cat([torch.full((s.ng,), G_PATCH), torch.full((n - s.ng,), L_PATCH)])
        err = ((rec - tgt) ** 2).sum(-1) / width
        return (err * mask).sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------- Mamba-2 (state-space duality, Dao & Gu 2024)
class Mamba2Block(nn.Module):
    """A Mamba-2 block in plain PyTorch. The selective state-space scan is computed in its matrix ("SSD") form,
    y_t = sum_{s<=t} exp(sum_{s<j<=t} dt_j A) (C_t . B_s) dt_s x_s, which is exact and fast for short sequences.
    Run in both directions over the patch sequence, because a folded light curve has no arrow of time."""
    def __init__(s, d, d_state=32, expand=2, headdim=32, d_conv=4):
        super().__init__()
        di = expand * d; s.h, s.p, s.n = di // headdim, headdim, d_state; s.di = di
        s.norm = nn.LayerNorm(d)
        s.inp = nn.Linear(d, 2 * di + 2 * d_state + s.h)          # x, z, B, C, dt
        s.conv = nn.Conv1d(di + 2 * d_state, di + 2 * d_state, d_conv, groups=di + 2 * d_state, padding=d_conv - 1)
        s.A_log = nn.Parameter(torch.log(torch.empty(s.h).uniform_(1, 16))); s.D = nn.Parameter(torch.ones(s.h))
        s.dt_bias = nn.Parameter(torch.log(torch.expm1(torch.empty(s.h).uniform_(1e-3, 1e-1))))
        s.gn = nn.LayerNorm(di); s.out = nn.Linear(2 * di, d)
    def _ssd(s, u):
        B_, T, _ = u.shape
        x, z, xbc_dt = torch.split(s.inp(u), [s.di, s.di, 2 * s.n + s.h], -1)
        xbc, dt = torch.split(torch.cat([x, xbc_dt], -1), [s.di + 2 * s.n, s.h], -1)
        xbc = F.silu(s.conv(xbc.transpose(1, 2))[..., :T].transpose(1, 2))
        x, Bm, Cm = torch.split(xbc, [s.di, s.n, s.n], -1)
        dt = F.softplus(dt + s.dt_bias)                                # (B, T, H)
        a = dt * -torch.exp(s.A_log)                                   # log decay per step, (B, T, H)
        cs = torch.cumsum(a, 1).transpose(1, 2)                        # (B, H, T)
        seg = cs[..., :, None] - cs[..., None, :]                      # sum over (s, t]
        Lm = torch.exp(seg.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=u.device).tril(), float('-inf')))
        xh = x.view(B_, T, s.h, s.p)
        y = torch.einsum('bhts,bts,bsh,bshp->bthp', Lm, Cm @ Bm.transpose(1, 2), dt, xh) + xh * s.D[:, None]
        return s.gn(y.reshape(B_, T, s.di)) * F.silu(z)
    def forward(s, u):
        v = s.norm(u)
        return u + s.out(torch.cat([s._ssd(v), s._ssd(v.flip(1)).flip(1)], -1))


class PatchMamba(nn.Module):
    """Same patch input as PatchTransformer (63 global + 26 local patches), Mamba-2 blocks instead of attention."""
    def __init__(s, d=128, layers=4, d_state=32, drop=0.1):
        super().__init__()
        s.eg, s.el = nn.Linear(G_PATCH, d), nn.Linear(L_PATCH, d)
        n = G_PAD // G_PATCH + L_PAD // L_PATCH
        s.pos = nn.Parameter(torch.randn(1, n, d) * 0.02); s.typ = nn.Parameter(torch.randn(2, d) * 0.02)
        s.blocks = nn.ModuleList([Mamba2Block(d, d_state) for _ in range(layers)])
        s.drop = nn.Dropout(drop); s.norm = nn.LayerNorm(d); s.head = nn.Linear(d, 1)
    def forward(s, g, l):
        pg, pl = patches(g, l)
        x = torch.cat([s.eg(pg) + s.typ[0], s.el(pl) + s.typ[1]], 1) + s.pos
        for b in s.blocks: x = s.drop(b(x))
        return s.head(s.norm(x).mean(1))[:, 0]
