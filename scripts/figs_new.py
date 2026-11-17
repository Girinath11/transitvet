import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/home/user/tv/src'); import tv_common as c
FIG = '/home/user/tv/repo/paper/figures'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
# ---- Fig: folded views of a planet candidate and an eclipsing binary (Kepler test stars)
G, L, t = c.load_kepler(); sp = json.load(open('/home/user/tv/e2/kepler_star_split.json')); te = np.array(sp['test'])
tt = t.iloc[te]
pc = tt[(tt.label == 1) & tt.snr.between(30, 60) & (tt.secondary_sigma.abs() < 2)].sort_values('snr').index[0]
eb = tt[(tt.label == 0) & tt.snr.between(30, 200) & (tt.secondary_sigma > 10)].sort_values('snr').index[0]
fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1]))
for r, (i, name) in enumerate(((pc, 'planet candidate (PC)'), (eb, 'false alarm (AFP)'))):
    row = t.loc[i]
    ax[r, 0].plot(np.linspace(-0.5, 0.5, G.shape[1]), G[i], lw=0.6, color='#2c5aa0')
    ax[r, 1].plot(np.linspace(-4, 4, L.shape[1]), L[i], lw=0.9, color='#2c5aa0')
    ax[r, 0].set_ylabel('normalised flux')
    ax[r, 0].set_title(f"KIC {int(row.kepid)}, P = {row.tce_period:.2f} d: {name}, global view (2,001 bins)", fontsize=8.5, loc='left')
    ax[r, 1].set_title('local view (201 bins)', fontsize=8.5, loc='left')
ax[1, 0].set_xlabel('phase (orbits)'); ax[1, 1].set_xlabel('time from dip centre (durations)')
fig.tight_layout(); fig.savefig(f'{FIG}/fig_views.png', dpi=220)
print('views', int(t.loc[pc].kepid), int(t.loc[eb].kepid))
# ---- Fig: 95% check
E = json.load(open('/home/user/tv/journal/eval.json')); C = json.load(open('/home/user/tv/journal/curves.json'))
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(width_ratios=[1.1, 1.1, 1]))
for k, mis in enumerate(('kepler', 'tess')):
    y = np.array(C[mis]['y']); p = np.array(C[mis]['conf_p']); q = E[mis]['conformal']
    b = np.linspace(0, 1, 26)
    ax[k].hist(p[y == 1], b, color='#3a7a2a', alpha=0.6, label='planets'); ax[k].hist(p[y == 0], b, color='#c0392b', alpha=0.5, label='false alarms')
    ax[k].axvline(1 - q['q1'], color='#3a7a2a', ls='--', lw=1); ax[k].axvline(q['q0'], color='#c0392b', ls='--', lw=1)
    ax[k].set_yscale('log'); ax[k].set_xlabel('conformal combiner score'); ax[k].set_title(f"{'Kepler' if mis=='kepler' else 'TESS'} test ({E[mis]['n_test']:,})", fontsize=9)
ax[0].set_ylabel('signals'); ax[0].legend(fontsize=7, frameon=False)
lab = ['Kepler', 'TESS']; n = [E[m]['n_test'] for m in ('kepler', 'tess')]; kk = []
for m in ('kepler', 'tess'):   # the tool's calls: conformal sets, plus score/check disagreements sent to a person
    pp = np.array(C[m]['combined']); pc = np.array(C[m]['conf_p']); q = E[m]['conformal']
    i1, i0 = (1 - pc) <= q['q1'], pc <= q['q0']; k2 = i1 & ~i0 & (pp >= 0.5); d2 = i0 & ~i1 & (pp < 0.5)
    kk.append(dict(keep=int(k2.sum()), drop=int(d2.sum()), person=int((~k2 & ~d2).sum())))
print('tool calls', kk)
left = np.zeros(2)
for key, col, nm in (('keep', '#3a7a2a', 'keep'), ('person', '#999999', 'needs a person'), ('drop', '#c0392b', 'drop')):
    v = np.array([x[key] / nn for x, nn in zip(kk, n)]); ax[2].barh(lab, v, left=left, color=col, label=nm); left += v
ax[2].set_xlim(0, 1); ax[2].set_xlabel('share of test signals'); ax[2].legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
fig.tight_layout(); fig.savefig(f'{FIG}/fig_conformal.png', dpi=220)
# ---- Fig: open TESS signals
u = pd.read_csv('/home/user/tv/repo/validation/unlabelled_tess_scores.csv')
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for v, col in (('false alarm', '#c0392b'), ('uncertain', '#999999'), ('candidate', '#2c5aa0')):
    s = u[u.verdict == v]; ax.scatter(s.planet_radius_re.where(s.planet_radius_re > 0.3), s.score, s=4, alpha=0.45, color=col, label=f'{v} ({len(s):,})')
ax.set_xscale('log'); ax.set_xlabel('estimated size (Earth radii)'); ax.set_ylabel('combined score'); ax.axhline(0.5, color='k', lw=0.6, ls=':')
ax.legend(fontsize=7, frameon=False, markerscale=3); ax.set_title(f'{len(u):,} open TESS signals (ExoFOP PC/APC)', fontsize=9)
fig.tight_layout(); fig.savefig(f'{FIG}/fig_open_signals.png', dpi=220)
print(u.verdict.value_counts().to_dict(), u.planet_radius_re.isna().sum())
# ---- Fig: PR curves from the same test scores the text reports
from sklearn.metrics import precision_recall_curve, average_precision_score as ap
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
for k, mis in enumerate(('kepler', 'tess')):
    y = np.array(C[mis]['y'])
    for key, col, nm in (('gb', '#d08a1e', 'gradient boosting'), ('mamba', '#7b4fa0', 'Mamba-2'), ('tf', '#888888', 'Transformer'), ('combined', '#2c5aa0', 'combined')):
        p = np.array(C[mis][key]); pr, rc, _ = precision_recall_curve(y, p)
        ax[k].plot(rc, pr, color=col, lw=2 if key == 'combined' else 1.1, label=f'{nm} ({ap(y, p):.3f})')
    ax[k].set_xlabel('recall'); ax[k].set_title(f"{'Kepler' if mis == 'kepler' else 'TESS'} test ({len(y):,} signals)", fontsize=9)
    ax[k].legend(fontsize=7, frameon=False, loc='lower left'); ax[k].set_ylim(0.4, 1.01)
ax[0].set_ylabel('precision'); fig.tight_layout(); fig.savefig(f'{FIG}/fig_pr_curves.png', dpi=220)
