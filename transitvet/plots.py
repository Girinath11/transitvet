"""Figures for one star: light curve, BLS periodogram, and folded views per signal."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL = {'candidate': '#66d19e', 'false alarm': '#f07b6c', 'uncertain': '#ffc861'}
# deep-space look, matching the TransitVet website
BG, PANEL, INK, MUTE, LINE, ICE, STAR = '#03050b', '#0a0e19', '#ece9e1', '#8f93a0', '#232838', '#5ee7ff', '#a78bfa'
plt.rcParams.update({
    'figure.facecolor': PANEL, 'axes.facecolor': PANEL, 'savefig.facecolor': PANEL,
    'axes.edgecolor': LINE, 'axes.labelcolor': MUTE, 'axes.titlecolor': INK, 'text.color': INK,
    'xtick.color': MUTE, 'ytick.color': MUTE, 'grid.color': LINE, 'axes.grid': True, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9.5, 'axes.titlesize': 10.5,
    'legend.facecolor': PANEL, 'legend.edgecolor': LINE, 'legend.labelcolor': INK,
    'axes.prop_cycle': matplotlib.cycler(color=[ICE, STAR, '#ff8bd1', '#66d19e']),
})


def lightcurve_figure(result):
    t, f = result['search_curve']
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.2), gridspec_kw=dict(height_ratios=[1.3, 1]))
    ax[0].plot(t, (f - 1) * 1e6, '.', ms=1, color=ICE, alpha=0.55)
    for i, s in enumerate(result['signals']):
        n = np.arange(np.ceil((t.min() - s['t0']) / s['period']), np.floor((t.max() - s['t0']) / s['period']) + 1)
        for tc in s['t0'] + n * s['period']:
            ax[0].axvline(tc, color=COL[s['verdict']], alpha=0.35, lw=0.8)
    ax[0].set_ylabel('brightness change (ppm)'); ax[0].set_xlabel('time (days)')
    ax[0].set_title('Detrended light curve (coloured lines = dips found, colour = verdict)')
    for k, pg in enumerate(result['periodograms']):
        ax[1].plot(pg['period'], pg['power'] / np.nanmax(pg['power']) + k * 1.1, lw=0.6)
    ax[1].set_xscale('log'); ax[1].set_xlabel('period (days)'); ax[1].set_ylabel('BLS power\n(one line per search)')
    ax[1].set_yticks([])
    fig.tight_layout()
    return fig


def signal_figure(rec, i):
    fig, ax = plt.subplots(1, 2, figsize=(11, 2.8), gridspec_kw=dict(width_ratios=[2, 1]))
    c = COL[rec['verdict']]
    ax[0].plot(np.linspace(-0.5, 0.5, 2001), rec['global_astronet'], '.', ms=1.5, color=c)
    ax[0].set_title(f"Signal {i}: whole orbit (global view), period {rec['period']:.4f} d")
    ax[0].set_xlabel('phase (orbits)')
    x = np.linspace(-4, 4, 201)
    ax[1].plot(x, rec['local_astronet'], '.', ms=3, color=c)
    ax[1].set_title('Zoom on the dip (local view)'); ax[1].set_xlabel('time from dip centre (durations)')
    for a in ax:
        a.set_yticks([])
    fig.tight_layout()
    return fig


def pixel_figure(rec):
    """Camera images of the clearest segment: the normal (out-of-dip) image and the difference image
    (which pixels got darker during the dip). + = target star (catalogue), x = where the dip comes from."""
    b = rec.get('px_best')
    if not b:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for a, key, title in ((ax[0], 'oot', 'Camera image (outside the dips)'), (ax[1], 'diff', 'Difference image (what got darker)')):
        a.imshow(b[key], origin='lower', cmap='magma'); a.grid(False)
        a.plot(*b['target_px'], '+', color=ICE, ms=16, mew=2.5, label='target star')
        a.plot(*b['source_px'], 'x', color='white', ms=12, mew=2.5, label='source of the dip')
        a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])
    ax[1].legend(loc='upper right', fontsize=7, framealpha=0.8)
    fig.suptitle(f"Segment {b['label']} · dip seen in the pixels at SNR {b['diff_snr']:.0f}", fontsize=9)
    fig.tight_layout()
    return fig
