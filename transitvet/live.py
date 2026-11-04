"""Live 'mission control' view shown while one star is analysed.

Every picture is drawn from that star's real intermediate data, handed over by pipeline.run(stage=...).
Nothing here changes the results; it only shows them as they are produced."""
import time

import numpy as np

from . import plots
from .plots import plt, ICE, STAR, MUTE, INK, COL

STAGES = [('download', 'Download', 'Fetch the light curve from NASA MAST'),
          ('clean', 'Clean', 'Drop flagged points, join the pieces'),
          ('flatten', 'Flatten', "Take out the star's slow drift"),
          ('search', 'Search', 'Try thousands of orbits (BLS)'),
          ('fold', 'Fold', 'Stack every orbit on one dip'),
          ('check', 'Check', 'Shape, size, light position, pixels'),
          ('verdict', 'Verdict', 'Three models, one calibrated score')]
_IDX = {k: i for i, (k, _, _) in enumerate(STAGES)}

CSS = """
<style>
.lv-track { position: relative; display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 6px; margin: 6px 0 14px; padding: 18px 8px 10px;
  background: radial-gradient(120% 140% at 50% 0%, rgba(120,110,255,.16), rgba(3,5,11,0) 60%); border-top: 1px solid rgba(214,222,240,.14); }
.lv-track::before { content: ""; position: absolute; left: 7%; right: 7%; top: 37px; height: 2px; background: rgba(214,222,240,.12); }
.lv-track .bar { position: absolute; left: 7%; top: 37px; height: 2px; background: linear-gradient(90deg, #a78bfa, #5ee7ff); box-shadow: 0 0 12px #5ee7ff; transition: width .6s ease; }
.lv-st { position: relative; text-align: center; min-width: 0; }
.lv-dot { position: relative; z-index: 1; width: 40px; height: 40px; margin: 0 auto 8px; border-radius: 50%; display: grid; place-items: center;
  font: 600 14px "Sora", sans-serif; color: #8f93a0; background: #0a0e19; border: 1px solid rgba(214,222,240,.22); }
.lv-st.done .lv-dot { color: #0a0820; background: linear-gradient(135deg, #a78bfa, #5ee7ff); border-color: transparent; }
.lv-st.on .lv-dot { color: #ece9e1; border-color: #a78bfa; box-shadow: 0 0 0 4px rgba(167,139,250,.18), 0 0 26px rgba(167,139,250,.7); }
.lv-st.on .lv-dot::after { content: ""; position: absolute; inset: -9px; border-radius: 50%; border: 1px dashed rgba(94,231,255,.55); animation: lv-spin 3s linear infinite; }
.lv-st.on .lv-dot::before { content: ""; position: absolute; width: 7px; height: 7px; border-radius: 50%; background: #5ee7ff; box-shadow: 0 0 8px #5ee7ff;
  top: -12px; left: 50%; margin-left: -3.5px; transform-origin: 3.5px 32px; animation: lv-orbit 2.2s linear infinite; }
@keyframes lv-spin { to { transform: rotate(360deg) } }
@keyframes lv-orbit { to { transform: rotate(360deg) } }
.lv-name { font: 600 14px "Manrope", sans-serif; color: #ece9e1; }
.lv-st:not(.done):not(.on) .lv-name { color: #8f93a0; }
.lv-sub { font-size: 12px; color: #8f93a0; line-height: 1.35; margin-top: 2px; }
@media (max-width: 800px) { .lv-sub { display: none } .lv-dot { width: 32px; height: 32px } .lv-track::before, .lv-track .bar { top: 33px } }
@media (prefers-reduced-motion: reduce) { .lv-st.on .lv-dot::after, .lv-st.on .lv-dot::before { animation: none } }
.lv-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; margin: 2px 0 6px; }
.lv-head b { font: 600 18px "Sora", sans-serif; letter-spacing: -.02em; color: #ece9e1; }
.lv-head span { font: 500 13px "JetBrains Mono", monospace; color: #5ee7ff; }
.lv-say { color: #c9c6bd; font-size: 15px; margin: 0 0 8px; }
.lv-feed { display: grid; gap: 6px; margin: 4px 0; }
.lv-item { display: grid; grid-template-columns: 22px minmax(0, 1fr); gap: 8px; font-size: 14.5px; color: #c9c6bd; animation: lv-in .45s ease both; }
.lv-item i { font-style: normal; color: #5ee7ff; }
.lv-item.warn i { color: #ffc861; } .lv-item.bad i { color: #f07b6c; } .lv-item.good i { color: #66d19e; }
@keyframes lv-in { from { opacity: 0; transform: translateY(6px) } to { opacity: 1; transform: none } }
.lv-verdicts { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }
.lv-v { border: 1px solid rgba(214,222,240,.16); border-radius: 10px; padding: 12px 14px; background: rgba(10,14,25,.8); animation: lv-in .5s ease both; }
.lv-v .k { font-size: 12.5px; color: #8f93a0; } .lv-v .t { font: 600 20px "Sora", sans-serif; margin: 2px 0; }
</style>
"""


def _track(active, done_upto):
    cells = []
    for i, (k, name, sub) in enumerate(STAGES):
        cls = 'done' if i < done_upto else ('on' if i == active else '')
        cells.append(f'<div class="lv-st {cls}"><div class="lv-dot">{"✓" if cls == "done" else i + 1}</div>'
                     f'<div class="lv-name">{name}</div><div class="lv-sub">{sub}</div></div>')
    pct = 86 * min(done_upto, len(STAGES) - 1) / (len(STAGES) - 1)
    return CSS + f'<div class="lv-track"><div class="bar" style="width:{pct:.1f}%"></div>{"".join(cells)}</div>'


def _fig(h=2.3):
    fig, ax = plt.subplots(figsize=(11, h))
    return fig, ax


class LiveView:
    """Owns a few Streamlit placeholders and redraws them as stage events arrive."""

    def __init__(self, st, target):
        self.st, self.target, self.t0 = st, target, time.time()
        self.track, self.head, self.plot, self.feed = st.empty(), st.empty(), st.empty(), st.empty()
        self.items, self.last_draw, self.peaks, self.active = [], 0.0, [], 0
        self.set('download', 'Asking NASA MAST for every light curve of this star. Bigger stars with years of data take longer.')
        try:   # bring the live view into sight (same-origin iframe; harmless if it fails)
            import streamlit.components.v1 as components
            components.html("<script>setTimeout(()=>{const d=window.parent.document;const e=[...d.querySelectorAll('.lv-track')].pop();"
                            "if(e)e.scrollIntoView({behavior:'smooth',block:'start'})},400)</script>", height=0)
        except Exception:
            pass

    # ---- helpers
    def set(self, key, say):
        self.active = _IDX[key]
        self.track.markdown(_track(self.active, self.active), unsafe_allow_html=True)
        name = STAGES[self.active][1]
        self.head.markdown(f'<div class="lv-head"><b>{self.active + 1} · {name}</b><span>{self.target} · {time.time() - self.t0:.0f} s</span></div>'
                           f'<p class="lv-say">{say}</p>', unsafe_allow_html=True)

    def show(self, fig):
        self.plot.pyplot(fig, clear_figure=True); plt.close(fig)

    def add(self, text, kind=''):
        icon = {'good': '✓', 'bad': '✗', 'warn': '!', '': '•'}[kind]
        self.items.append(f'<div class="lv-item {kind}"><i>{icon}</i><span>{text}</span></div>')
        self.feed.markdown('<div class="lv-feed">' + ''.join(self.items[-8:]) + '</div>', unsafe_allow_html=True)

    # ---- stage handlers (called from the app and from pipeline.run)
    def loaded(self, segs, meta):
        self.set('clean', f"Got {sum(len(s[0]) for s in segs):,} measurements in {len(segs)} piece(s) from {meta['mission'].upper()}. "
                          'Each piece is scaled to the same level so they join up.')
        fig, ax = _fig()
        cols = [ICE, STAR, '#ff8bd1']
        for k, (t, f) in enumerate(segs):
            ax.plot(t, (f / np.nanmedian(f) - 1) * 1e6, '.', ms=0.8, color=cols[k % 3], alpha=0.6)
        ax.set_xlabel('time (days)'); ax.set_ylabel('brightness (ppm)'); ax.set_title('Raw starlight, one colour per piece')
        fig.tight_layout(); self.show(fig)

    def __call__(self, name, **d):
        getattr(self, '_' + name)(**d)

    def _flatten(self, t, f):
        self.set('flatten', f'A sliding filter follows the slow changes of the star and divides them out. What is left: {t.size:,} points, flat except for short dips.')
        fig, ax = _fig()
        ax.plot(t, (f - 1) * 1e6, '.', ms=0.8, color=ICE, alpha=0.55)
        lo, hi = np.nanpercentile((f - 1) * 1e6, [0.2, 99.8]); ax.set_ylim(lo * 1.3, hi * 1.3)
        ax.set_xlabel('time (days)'); ax.set_ylabel('brightness (ppm)'); ax.set_title('After flattening')
        fig.tight_layout(); self.show(fig); self.t_f = (t, f)

    def _search(self, k, parts, n):
        now = time.time()
        done = sum(len(p[0]) for p in parts)
        if now - self.last_draw < 0.8 and done < n:
            return
        self.last_draw = now
        self.set('search', f'Search {k + 1}: sliding a box-shaped dip along the data at every trial orbit and scoring the fit. '
                           f'{done:,} of {n:,} trial orbits tried.')
        per = np.concatenate([p[0] for p in parts]); pw = np.concatenate([p[1] for p in parts])
        st = max(1, per.size // 5000)
        fig, ax = _fig()
        ax.plot(per[::st], pw[::st], lw=0.7, color=ICE)
        for (pk, sig) in self.peaks:
            if pk == k:
                continue
            ax.axvline(sig['period'], color=STAR, lw=0.8, alpha=0.5)
        ax.set_xscale('log'); ax.set_xlabel('trial orbit (days)'); ax.set_ylabel('how dip-like'); ax.set_yticks([])
        ax.set_title('Periodogram building up: a tall spike = a repeating dip')
        fig.tight_layout(); self.show(fig)

    def _peak(self, k, sig):
        self.peaks.append((k, sig))
        strong = sig['bls_snr'] >= 7.1
        self.add(f"Search {k + 1}: best orbit {sig['period']:.4f} d, dip {sig['depth_ppm']:.0f} ppm, signal-to-noise {sig['bls_snr']:.1f}"
                 + ('' if strong else ' (too weak, stop searching)'), 'good' if strong else '')

    def _searched(self, sigs):
        if not sigs:
            self.add('No repeating dip strong enough was found.', 'warn')

    def _fold(self, sigs, recs):
        self.set('fold', 'Every orbit is stacked on top of the others. Noise averages away and the dip stands out: '
                         'one picture of the whole orbit and one close-up of the dip, the two inputs the models read.')
        n = len(recs)
        fig, ax = plt.subplots(n, 2, figsize=(11, 1.9 * n + 0.4), gridspec_kw=dict(width_ratios=[2, 1]), squeeze=False)
        for i, r in enumerate(recs):
            ax[i, 0].plot(np.linspace(-0.5, 0.5, len(r['global_astronet'])), r['global_astronet'], '.', ms=1.4, color=ICE)
            ax[i, 1].plot(np.linspace(-4, 4, len(r['local_astronet'])), r['local_astronet'], '.', ms=2.6, color=STAR)
            ax[i, 0].set_ylabel(f'signal {i + 1}'); ax[i, 0].set_yticks([]); ax[i, 1].set_yticks([])
        ax[0, 0].set_title('Whole orbit (folded)'); ax[0, 1].set_title('Close-up of the dip')
        ax[-1, 0].set_xlabel('phase (orbits)'); ax[-1, 1].set_xlabel('time from the dip (dip lengths)')
        fig.tight_layout(); self.show(fig)

    def _check(self, i, sig, rec, star):
        if self.active != _IDX['check']:
            self.set('check', 'Each dip is questioned: is it the right size and shape for a planet, does the light stay on the target, '
                              'and is it already in NASA\'s lists?')
        rp = rec.get('rp_est_re', np.nan)
        if np.isfinite(rp):
            self.add(f'Signal {i + 1}: size about {rp:.1f} × Earth' + (' (too big for a planet)' if rp > 22.4 else ''), 'bad' if rp > 22.4 else 'good')
        fit = rec.get('fit') or {}
        if fit.get('fit_ok'):
            self.add(f"Signal {i + 1}: transit model fitted, period {fit['period']:.5f} d")
        cs = rec.get('cen_sigma', np.nan)
        if np.isfinite(cs):
            self.add(f'Signal {i + 1}: light-position shift during the dip, {cs:.1f} sigma', 'warn' if cs > 10 else 'good')
        kn = rec.get('known') or {}
        if kn.get('name'):
            self.add(f"Signal {i + 1}: in NASA's list as {kn['name']} ({kn.get('disposition', '')})")

    def _pixel_start(self):
        self.add('Downloading camera images to see which pixels get darker during the dip...')

    def _pixel(self, recs):
        for i, r in enumerate(recs):
            off, sg = r.get('px_offset_arcsec', np.nan), r.get('px_sigma', np.nan)
            if np.isfinite(off):
                far = sg > 5 and off > 10
                self.add(f'Signal {i + 1}: dip source {off:.0f}″ from the target ({sg:.1f} sigma)' + (' — likely a neighbouring star' if far else ''), 'bad' if far else 'good')
        b = next((r.get('px_best') for r in recs if r.get('px_best')), None)
        if b is not None:
            try:
                self.show(plots.pixel_figure({'px_best': b}))
            except Exception:
                pass

    def _verdict(self, signals):
        self.set('verdict', 'Three models read the folded pictures and the check numbers; their scores are joined into one calibrated score.')
        cards = []
        for i, r in enumerate(signals):
            v = r['verdict']; p = r['scores'].get(r['main_model'], np.nan)
            cards.append(f'<div class="lv-v"><div class="k">Signal {i + 1} · {r["period"]:.3f} d</div>'
                         f'<div class="t" style="color:{COL[v]}">{v}</div><div class="k">score {p:.2f}</div></div>')
        self.track.markdown(_track(len(STAGES), len(STAGES)), unsafe_allow_html=True)
        self.head.markdown(f'<div class="lv-head"><b>Done</b><span>{self.target} · {time.time() - self.t0:.0f} s</span></div>'
                           f'<div class="lv-verdicts">{"".join(cards)}</div>', unsafe_allow_html=True)
