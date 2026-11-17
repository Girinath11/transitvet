"""Analyse the injection-recovery rows: recovery curves vs expected SNR, logistic fits for the dashboard, paper figure."""
import json, glob, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

rows = []
for p in glob.glob('/home/user/tv/inj/res/*.jsonl'):
    m = 'kepler' if 'kepler' in p else 'tess'
    for l in open(p):
        d = json.loads(l); d['mission'] = m; rows.append(d)
import pickle, gzip, os
def _rad(star):
    p = f"/home/user/tv/inj/cache/{star.replace(' ', '_')}.pkl.gz"
    try: return pickle.load(gzip.open(p))['star'].get('star_rad', float('nan'))
    except Exception: return float('nan')
RAD = {s: _rad(s) for s in {r['star'] for r in rows}}
giant = [r for r in rows if RAD[r['star']] == RAD[r['star']] and RAD[r['star']] > 5]   # the injected depth would be a star, not a planet
rows = [r for r in rows if r not in giant]
allrows = rows
few = [r for r in rows if r['n_tr'] < 3]                 # fewer than 3 dips land in the data: the search cannot find these by design
big = [r for r in rows if r['n_tr'] >= 3 and r['rp_re'] > 16]   # larger than about 1.5 Jupiter radii: not planet-like
rows = [r for r in rows if r['n_tr'] >= 3 and r['rp_re'] <= 16 and r['exp_snr'] > 0]
x = np.log10(np.array([r['exp_snr'] for r in rows]))
found = np.array([r['found'] for r in rows]).astype(int)
cand = np.array([r['found'] and r.get('verdict') == 'candidate' for r in rows]).astype(int)
uncert = np.array([r['found'] and r.get('verdict') == 'uncertain' for r in rows]).astype(int)
mis = np.array([r['mission'] for r in rows])

def fit(y):
    lr = LogisticRegression(C=1e4).fit(x[:, None], y)
    return [float(lr.intercept_[0]), float(lr.coef_[0, 0])]
fits = dict(found=fit(found), candidate=fit(cand))
def x_at(a, b, p): return 10 ** ((np.log(p / (1 - p)) - a) / b)
edges = [0, 3, 7.1, 10, 15, 25, 50, 1e9]
bins = []
for lo, hi in zip(edges[:-1], edges[1:]):
    s = (10 ** x >= lo) & (10 ** x < hi)
    if s.sum():
        bins.append(dict(lo=lo, hi=hi if hi < 1e8 else None, n=int(s.sum()), found=int(found[s].sum()), candidate=int(cand[s].sum()),
                         uncertain=int(uncert[s].sum())))
summ = dict(n_giant=len(giant), n_all=len(allrows), n_few_dips=len(few), n_big=len(big), big_found=sum(r['found'] for r in big),
            big_candidate=sum(r['found'] and r.get('verdict') == 'candidate' for r in big), n=len(rows), n_tess=int((mis == 'tess').sum()), n_kepler=int((mis == 'kepler').sum()),
            n_stars=len({r['star'] for r in rows}), found=int(found.sum()), candidate=int(cand.sum()), uncertain=int(uncert.sum()),
            fit=fits, snr50_found=float(x_at(*fits['found'], 0.5)), snr90_found=float(x_at(*fits['found'], 0.9)),
            snr50_candidate=float(x_at(*fits['candidate'], 0.5)), snr90_candidate=float(x_at(*fits['candidate'], 0.9)), bins=bins)
# verdicts of found planets above SNR 15 (clearly detectable)
hi = 10 ** x >= 15
fa = np.array([r['found'] and r.get('verdict') == 'false alarm' for r in rows]).astype(int)
summ['above15'] = dict(false_alarm=int(fa[hi].sum()), n=int(hi.sum()), found=int(found[hi].sum()), candidate=int(cand[hi].sum()), uncertain=int(uncert[hi].sum()))
# verdicts of found planets by host type (what the star's own TOI turned out to be)
import pandas as pd
toi = pd.read_csv('/home/user/tv/out/tess/transitvet_tess_toi_table.csv', low_memory=False)
hostlab = toi.groupby('tid').label.max()
def host(r):
    if r['mission'] != 'tess': return 'kepler'
    v = hostlab.get(int(r['star'].split()[1]))
    return 'planet host' if v == 1 else ('false-alarm host' if v == 0 else 'other')
summ['by_host'] = {}
for h in ('planet host', 'false-alarm host'):
    rr = [r for r in rows if host(r) == h and r['found']]
    summ['by_host'][h] = dict(stars=len({r['star'] for r in rows if host(r) == h}), found=len(rr),
                              candidate=sum(r['verdict'] == 'candidate' for r in rr), uncertain=sum(r['verdict'] == 'uncertain' for r in rr),
                              false_alarm=sum(r['verdict'] == 'false alarm' for r in rr), keep=sum(r.get('triage') == 'keep' for r in rr))
# called false alarm by planet size, for found planets
sz = np.array([r['rp_re'] for r in rows])
summ['fa_by_size'] = {k: dict(found=int(found[s_].sum()), false_alarm=int(fa[s_].sum()), candidate=int(cand[s_].sum()))
                      for k, s_ in (('under4', sz < 4), ('4to10', (sz >= 4) & (sz < 10)), ('10to16', sz >= 10))}
print(json.dumps(summ, indent=1))
json.dump(summ, open('/home/user/tv/inj/summary.json', 'w'), indent=1)
json.dump(dict(fit=fits, note='logistic in log10(expected SNR); from the injection-recovery test, see validation/INJECTION.md',
               n=len(rows), date='2026-11-17'), open('/home/user/tv/repo/transitvet/sensitivity.json', 'w'), indent=1)

# figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
g = np.linspace(x.min() - 0.1, x.max() + 0.1, 200)
for key, col, lab in (('found', '#2c5aa0', 'found by the search'), ('candidate', '#3a7a2a', 'found and called candidate')):
    a, b = fits[key]
    if key == 'found':
        ax[0].plot(10 ** g, 1 / (1 + np.exp(-(a + b * g))), color=col, lw=2, label=lab + ' (fit)')
    else:   # no smooth fit: the share called candidate levels off well below 1 and a logistic would overstate it
        xs = [np.sqrt(max(bn['lo'], 10 ** x.min()) * (bn['hi'] or 10 ** x.max())) for bn in bins]
        ax[0].plot(xs, [bn[key] / bn['n'] for bn in bins], color=col, lw=1.2, ls='--', label=lab)
    for bn in bins:
        lo = max(bn['lo'], 10 ** x.min()); hi_ = bn['hi'] or 10 ** x.max(); xc = np.sqrt(lo * hi_)
        k = bn[key]; n = bn['n']; p = k / n; e = np.sqrt(p * (1 - p) / n + 1e-4)
        ax[0].errorbar(xc, p, yerr=e, fmt='o', color=col, ms=5, capsize=2)
ax[0].axvline(7.1, color='#888', ls=':', lw=1); ax[0].text(7.4, 0.04, 'search cut-off 7.1', fontsize=8, color='#555')
ax[0].set_xscale('log'); ax[0].set_xlabel('expected signal-to-noise of the injected planet'); ax[0].set_ylabel('fraction')
ax[0].set_ylim(-0.03, 1.03); ax[0].legend(fontsize=8, loc='upper left'); ax[0].set_title(f'Recovery of {len(rows)} injected planets', fontsize=10)
P_ = np.array([r['period'] for r in rows]); R_ = np.array([r['rp_re'] for r in rows])
for sel, c, mk, lab in ((found == 0, '#bbbbbb', 'x', 'missed'), ((found == 1) & (cand == 0), '#d08a1e', 's', 'found, not candidate'),
                        (cand == 1, '#3a7a2a', 'o', 'found, candidate')):
    ax[1].scatter(P_[sel], R_[sel], c=c, marker=mk, s=26, label=lab)
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('period (days)'); ax[1].set_ylabel('planet size (Earth radii)')
ax[1].legend(fontsize=8); ax[1].set_title('Each injected planet', fontsize=10)
fig.tight_layout(); fig.savefig('/home/user/tv/repo/paper/figures/fig_sensitivity.png', dpi=200)

json.dump(dict(fit=fits, by_host=summ['by_host'], snr50_found=summ['snr50_found'], snr90_found=summ['snr90_found'], n=len(rows), date='2026-11-17',
               note='logistic in log10(expected SNR); from the injection-recovery test, see validation/INJECTION.md'),
          open('/home/user/tv/repo/transitvet/sensitivity.json', 'w'), indent=1)
json.dump(summ, open('/home/user/tv/inj/summary.json', 'w'), indent=1)
