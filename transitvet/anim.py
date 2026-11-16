"""Animated scenes for the live view. Each scene is a small self-contained web page (canvas + script) that plays
one analysis step with that star's real data. Python builds the data; anim_js.js draws it in the browser."""
import json
import os

import numpy as np

_JS = os.path.join(os.path.dirname(__file__), 'anim_js.js')

# how long each scene plays (seconds), so the next step does not replace it before it has been seen
DWELL = {'load': 5.2, 'flatten': 7.0, 'search': 0.0, 'found': 8.2, 'found_doubled': 9.6, 'fold': 5.4, 'check': 5.6, 'models': 15.5}

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Manrope:wght@500;600;800&family=Sora:wght@600;800&display=swap" rel="stylesheet">
<style>
html,body{margin:0;background:transparent;color:#ece9e1;font:14px "Manrope",system-ui,sans-serif}
.wrap{border:1px solid rgba(214,222,240,.14);border-radius:12px;background:#070a14;padding:10px 12px 12px;box-sizing:border-box}
.top{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px}
.step{font:600 13px "JetBrains Mono",monospace;color:#5ee7ff;letter-spacing:.02em}
#tabs{display:flex;gap:6px}#tabs button,#rp{font:500 12px "Manrope",sans-serif;color:#c9c6bd;background:#0b1020;border:1px solid rgba(214,222,240,.18);border-radius:999px;padding:3px 10px;cursor:pointer}
#tabs button.on{color:#07101c;background:#5ee7ff;border-color:#5ee7ff}#rp:hover,#tabs button:hover{border-color:#a78bfa}
#say{min-height:40px;font-size:14px;line-height:1.45;color:#d9d6ce;margin:2px 0 8px}#say.in{animation:fi .5s ease both}
@keyframes fi{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.main{display:grid;grid-template-columns:minmax(0,1fr) 190px;gap:10px}
#nums{display:flex;flex-direction:column;gap:6px;font-size:12px}#nums div{display:flex;flex-direction:column;border-left:2px solid #a78bfa;padding:2px 0 2px 8px;animation:fi .4s ease both}
#nums span{color:#8f93a0}#nums b{font:600 14px "JetBrains Mono",monospace;color:#ece9e1;overflow-wrap:anywhere}
.prog{height:3px;background:rgba(214,222,240,.08);border-radius:2px;margin-top:8px;overflow:hidden}.prog i{display:block;height:100%;width:0;background:linear-gradient(90deg,#a78bfa,#5ee7ff)}
@media (max-width:640px){.main{grid-template-columns:1fr}#nums{flex-direction:row;flex-wrap:wrap}#nums div{min-width:40%}}
</style></head><body><div class="wrap">
<div class="top"><div class="step">@@TITLE@@</div><div style="display:flex;gap:8px;align-items:center"><div id="tabs"></div><button id="rp" title="Play this step again">↻ replay</button></div></div>
<div id="say"></div>
<div class="main"><div><canvas id="cv"></canvas></div><aside id="nums"></aside></div>
<div class="prog"><i id="pb"></i></div>
</div><script>window.D=@@DATA@@;</script><script>@@JS@@</script></body></html>"""

_js_cache = None


def page(kind, title, data, height):
    global _js_cache
    if _js_cache is None:
        with open(_JS, encoding='utf-8') as fh:
            _js_cache = fh.read()
    data = dict(data, kind=kind, h=height)
    import html
    js = json.dumps(data, separators=(',', ':'), default=_num).replace('</', '<\\/')
    return PAGE.replace('@@TITLE@@', html.escape(title)).replace('@@DATA@@', js).replace('@@JS@@', _js_cache)


def frame_height(height):
    """iframe height for a canvas of this height (header, narration, progress bar; numbers move below on phones)."""
    return height + 110


def _num(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    raise TypeError(type(v))


def r(a, d=1):
    a = np.asarray(a, float)
    return [None if not np.isfinite(v) else round(float(v), d) for v in a]


def thin(t, f, n=7000, keep_low=True):
    """Indices to draw: every low point (possible dip, up to 3000) plus an even sample of the rest, n in all."""
    t = np.asarray(t); f = np.asarray(f)
    if t.size <= n:
        return np.arange(t.size)
    idx = np.zeros(t.size, bool)
    if keep_low:
        med = np.nanmedian(f); sd = 1.4826 * np.nanmedian(np.abs(f - med))
        low = np.where(f < med - 3 * sd)[0]
        if low.size > 3000:
            low = low[np.argsort(f[low])[:3000]]
        idx[low] = True
    rest = np.where(~idx)[0]
    idx[rest[np.linspace(0, rest.size - 1, max(0, n - idx.sum())).astype(int)]] = True
    return np.where(idx)[0]


def load_data(segs, meta):
    meds = [float(np.nanmedian(f)) for _, f in segs]
    M = float(np.median(meds))
    n_all = sum(len(t) for t, _ in segs)
    per = max(1, n_all // 6000)
    out = []
    for (t, f), m in zip(segs, meds):
        ok = np.isfinite(t) & np.isfinite(f)
        t, f = t[ok][::per], f[ok][::per]
        out.append(dict(t=r(t, 3), y=r((f / M - 1) * 1e6, 0), off=round((m / M - 1) * 1e6, 1)))
    ts = np.sort(np.concatenate([s[0][np.isfinite(s[0])] for s in segs]))
    dt = np.diff(ts)
    return dict(segs=out, mission=meta['mission'], n_points=int(n_all),
                cadence_min=float(np.median(dt) * 1440) if dt.size else None, gaps=int((dt > 1).sum()))


def flatten_data(segs, t, f):
    T, R = [], []
    for ts, fs in segs:
        fs = fs / np.nanmedian(fs); ok = np.isfinite(ts) & np.isfinite(fs)
        T.append(ts[ok]); R.append(fs[ok])
    T, R = np.concatenate(T), np.concatenate(R)
    o = np.argsort(T, kind='stable'); T, R = T[o], R[o]
    j = np.clip(np.searchsorted(T, t), 0, T.size - 1)
    same = np.abs(T[j] - t) < 1e-7
    raw = np.where(same, R[j], np.nan)
    kept = np.zeros(T.size, bool); kept[j[same]] = True
    i = thin(t, f)
    ri = np.where(~kept)[0]
    ri = ri[np.linspace(0, ri.size - 1, min(400, ri.size)).astype(int)] if ri.size else ri
    mad = lambda a: 1.4826e6 * np.nanmedian(np.abs(a - np.nanmedian(a)))
    return dict(t=r(t[i], 3), r=r((raw[i] - 1) * 1e6, 0), trend=r((raw[i] / f[i] - 1) * 1e6, 0), f=r((f[i] - 1) * 1e6, 0),
                rem_t=r(T[ri], 3), rem_r=r((R[ri] - 1) * 1e6, 0), removed=int((~kept).sum()), n_kept=int(t.size),
                sd_before=float(mad(R)), sd_after=float(mad(f)))


def masked(t, sigs):
    keep = np.ones(t.size, bool)
    for s in sigs:
        ph = (t - s['t0'] + 0.5 * s['period']) % s['period'] - 0.5 * s['period']
        keep &= np.abs(ph) >= 1.5 * s['duration']        # same width as the search (3 durations in all)
    return keep


def search_data(t, f, k, prev, n_trials, pmin, pmax):
    keep = masked(t, prev)
    tt, ff = t[keep], f[keep]
    i = thin(tt, ff, 6000)
    return dict(t=r(tt[i], 3), f=r((ff[i] - 1) * 1e6, 0), k=k, n_trials=int(n_trials), pmin=pmin, pmax=pmax, n_all=int(tt.size),
                others=[s['period'] for s in prev])


def found_data(base, sig, per, pw, n_dips):
    st = max(1, per.size // 2000)
    o = np.argsort(per[::st])
    return dict(base, mode='found', pg=dict(p=r(per[::st][o], 4), w=r(pw[::st][o], 3)),
                sig=dict(period=sig['period'], t0=sig['t0'], dur_d=sig['duration'], dur_h=24 * sig['duration'], depth=sig['depth_ppm'],
                         snr=sig['bls_snr'], weak=bool(sig['bls_snr'] < 7.1), doubled=bool(sig.get('period_doubled')),
                         half_sigma=sig.get('half_period_sigma'), n_dips=n_dips))


def transits(t, f, sig, max_tr=12, per_tr=110):
    P, t0, d = sig['period'], sig['t0'], sig['duration']
    e = np.round((t - t0) / P)
    x = (t - t0 - e * P) / d
    near = np.abs(x) < 2.5
    out = []
    for n in np.unique(e[near]):
        m = near & (e == n)
        if m.sum() < 5:
            continue
        xs, ys = x[m], (f[m] - 1) * 1e6
        if xs.size > per_tr:
            j = np.linspace(0, xs.size - 1, per_tr).astype(int); xs, ys = xs[j], ys[j]
        out.append(dict(n=int(n), x=r(xs, 3), y=r(ys, 0), npts=int(m.sum())))
    out.sort(key=lambda o: -o['npts'])
    out = sorted(out[:max_tr], key=lambda o: o['n'])
    return out


def half_orbit(t, f, sig, cap=600):
    P, t0, d = sig['period'], sig['t0'], sig['duration']
    x = ((t - t0 - 0.5 * P + 0.5 * P) % P - 0.5 * P) / d
    m = np.abs(x) < 2.5
    xs, ys = x[m], (f[m] - 1) * 1e6
    if xs.size > cap:
        j = np.linspace(0, xs.size - 1, cap).astype(int); xs, ys = xs[j], ys[j]
    return dict(x=r(xs, 3), y=r(ys, 0))


def view_data(rec):
    from . import tv_models as M
    return dict(g=r(rec['global_astronet'], 3), l=r(rec['local_astronet'], 3),
                ng=M.G_PAD // M.G_PATCH, nl=M.L_PAD // M.L_PATCH)
