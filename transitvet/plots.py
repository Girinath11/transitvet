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


SITE_URL = 'girinath11.github.io/transitvet'


def share_card(target, mission, rec, i, checks):
    """One shareable image (1200 x 675 px) for one signal: verdict, key numbers, the two folded views and the report card.
    checks = pipeline.report_card(rec). Returns PNG bytes."""
    import io
    v = rec['verdict']; c = COL[v]
    fig = plt.figure(figsize=(12, 6.75), dpi=100, facecolor=BG)
    t1 = fig.text(0.04, 0.92, 'Transit', fontsize=17, weight='bold', color=INK)
    bb = t1.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(bb.x1, 0.92, 'Vet', fontsize=17, weight='bold', color=STAR)
    fig.text(0.96, 0.92, f'{target} · {mission.upper()} · signal {i}', fontsize=13, color=MUTE, ha='right')
    fig.add_artist(matplotlib.lines.Line2D([0.04, 0.96], [0.895, 0.895], color=LINE, lw=1))
    fig.text(0.04, 0.80, {'candidate': 'Planet candidate', 'false alarm': 'False alarm', 'uncertain': 'Uncertain'}[v],
             fontsize=30, weight='bold', color=c)
    p = rec['scores'].get(rec['main_model'], np.nan)
    ft = rec.get('fit') or {}
    rp = ft.get('planet_radius_re', rec.get('rp_est_re', np.nan))
    nums = [('score', f'{p:.2f}'), ('orbit', f"{rec['period']:.3f} d"), ('dip depth', f"{rec['depth_ppm']:.0f} ppm"),
            ('size (Earth = 1)', f'{rp:.1f}' if rp == rp else '?'), ('dips seen', str(rec.get('n_transits', '?')))]
    for k, (lab, val) in enumerate(nums):
        x = 0.04 + k * 0.102
        fig.text(x, 0.71, val, fontsize=14, weight='bold', color=INK)
        fig.text(x, 0.675, lab, fontsize=9.5, color=MUTE)
    a1 = fig.add_axes([0.04, 0.12, 0.30, 0.46]); a2 = fig.add_axes([0.36, 0.12, 0.15, 0.46])
    a1.plot(np.linspace(-0.5, 0.5, len(rec['global_astronet'])), rec['global_astronet'], '.', ms=1.4, color=c)
    a2.plot(np.linspace(-4, 4, len(rec['local_astronet'])), rec['local_astronet'], '.', ms=2.6, color=c)
    a1.set_title('Whole orbit, folded', fontsize=10, loc='left'); a2.set_title('Close-up of the dip', fontsize=10, loc='left')
    for a in (a1, a2):
        a.set_xticks([]); a.set_yticks([]); a.set_facecolor(PANEL)
    fig.text(0.56, 0.80, 'Report card', fontsize=13, weight='bold', color=INK)
    mark = {'pass': ('✓', COL['candidate']), 'warn': ('!', COL['uncertain']), 'n/a': ('–', MUTE)}
    y = 0.74
    for check, found, status in checks[:11]:
        s, col = mark[status]
        fig.text(0.56, y, s, fontsize=12, weight='bold', color=col)
        fig.text(0.585, y, check, fontsize=10.5, color=INK)
        fig.text(0.585, y - 0.028, (found[:62] + '…') if len(found) > 63 else found, fontsize=8.5, color=MUTE)
        y -= 0.058
    fig.text(0.04, 0.04, 'A planet candidate is not a confirmed planet: it needs more observations.', fontsize=9.5, color=MUTE)
    fig.text(0.96, 0.04, SITE_URL, fontsize=10, color=ICE, ha='right')
    buf = io.BytesIO(); fig.savefig(buf, format='png', facecolor=BG); plt.close(fig)
    return buf.getvalue()
