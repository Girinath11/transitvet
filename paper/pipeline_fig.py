import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(11, 6.2)); ax.set_xlim(0, 110); ax.set_ylim(-1, 62); ax.axis('off')
def box(x, y, w, h, title, lines, fc='#ffffff', ec='#444444', bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc=fc, ec=ec, lw=1.6 if bold else 1.1))
    ax.text(x + w / 2, y + h - 2.2, title, ha='center', va='top', fontsize=10.5, weight='bold')
    ax.text(x + w / 2, y + h - 5.4, '\n'.join(lines), ha='center', va='top', fontsize=8.4, color='#333333', linespacing=1.35)
def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color='#555555', lw=1.2))
# row 1: data -> clean -> search -> per signal
box(1, 41, 19, 18, 'Light curve', ['Kepler long cadence', 'or TESS 2-min (SPOC)', 'quality flag 0'])
box(24, 41, 19, 18, 'Clean, detrend', ['biweight filter (search)', 'spline (folded views)', 'TESS binned to 30 min'])
box(47, 41, 19, 18, 'BLS search', ['0.6 d to span/3 (≤500 d)', 'SNR ≥ 7.1, up to 5 signals', 'mask and repeat'])
box(70, 41, 39, 18, 'For each signal', ['folded views: global 2,001 and local 201 bins', 'checks: depth, SNR, odd/even, half-orbit dip,', 'shape, dips alike, centroid, difference image', 'star: TIC radius, crowding,', '13 Gaia DR2 numbers (TESS)'])
arrow(20.3, 51.5, 23.7, 51.5); arrow(43.3, 51.5, 46.7, 51.5); arrow(66.3, 51.5, 69.7, 51.5)
# row 2: three models
box(70, 23, 12, 15, 'Gradient', ['boosting', 'check numbers', '+ Gaia'], fc='#eef3fb')
box(83.5, 23, 12, 15, 'Mamba-2', ['folded views', 'as patches'], fc='#eef3fb')
box(97, 23, 12, 15, 'Transformer', ['folded views,', 'masked-patch', 'pretrained'], fc='#eef3fb')
for x in (76, 89.5, 103): arrow(x, 40.7, x, 38.3)
# combiner
box(40, 23, 25, 15, 'Combined score', ['isotonic calibration', 'per model, then logistic', 'combiner on log-odds', '(validation stars only)'], fc='#eef3fb', ec='#2c5aa0', bold=True)
arrow(69.7, 30.5, 65.3, 30.5)
box(1, 23, 34, 15, '95% check (conformal)', ['class-conditional split conformal', 'keep / drop / needs a person', 'coverage >= 95% on test stars'.replace('>=', '≥')], fc='#f3f8ef', ec='#3a7a2a', bold=True)
arrow(39.7, 30.5, 35.3, 30.5)
box(1, 0.5, 34, 16.5, 'Verdict', ['candidate if score ≥ 0.5; uncertain if', '0.35–0.65, models disagree, < 3 dips,', 'dips unlike, a second dip,', 'or light from a neighbour'])
arrow(18, 22.7, 18, 17.3)
box(40, 0.5, 32, 16.5, 'Outputs', ['dashboard report card per signal', 'batch CSV for many stars', 'ranked list of 4,408 open TESS signals', 'weekly forward check against NASA'])
arrow(35.3, 9.5, 39.7, 9.5)
box(77, 0.5, 32, 16.5, 'Display only', ['transit fit (batman)', 'dip timing (TTV flag)', 'chance of catching such a planet'])
arrow(76.7, 9.5, 72.3, 9.5)
ax.text(1, 61.5, 'TransitVet pipeline: one light curve in, a calibrated call per signal out', fontsize=12, weight='bold', va='bottom')
fig.savefig('/home/user/tv/repo/paper/figures/fig_pipeline.png', dpi=200, bbox_inches='tight')
