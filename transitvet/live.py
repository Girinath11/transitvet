"""Live 'mission control' view shown while one star is analysed.

Every picture is drawn from that star's real intermediate data, handed over by pipeline.run(stage=...).
Nothing here changes the results; it only shows them as they are produced."""
import time

import numpy as np

from . import anim, plots
from .pipeline import MAX_SIGNALS as MAXS
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
.lv-chips { display: flex; flex-wrap: wrap; gap: 6px 8px; margin: 0 0 10px; }
.lv-chip { font: 500 12.5px "JetBrains Mono", monospace; color: #c9c6bd; border: 1px solid rgba(214,222,240,.16); border-radius: 999px; padding: 3px 10px; background: rgba(10,14,25,.7); }
.lv-chip b { color: #5ee7ff; font-weight: 600; }
.lv-models { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 10px; margin-top: 10px; }
.lv-m { border: 1px solid rgba(214,222,240,.16); border-radius: 10px; padding: 12px 14px; background: rgba(10,14,25,.8); animation: lv-in .5s ease both; }
.lv-m .hd { display: flex; justify-content: space-between; font-size: 13px; color: #8f93a0; margin-bottom: 6px; }
.lv-row { display: grid; grid-template-columns: 128px minmax(0, 1fr) 40px; gap: 8px; align-items: center; font-size: 13px; color: #c9c6bd; margin: 3px 0; }
.lv-row .tr { height: 6px; border-radius: 3px; background: rgba(214,222,240,.10); position: relative; overflow: hidden; }
.lv-row .tr i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; background: linear-gradient(90deg, #a78bfa, #5ee7ff); }
.lv-row .tr::after { content: ""; position: absolute; left: 50%; top: -2px; bottom: -2px; width: 1px; background: rgba(214,222,240,.35); }
.lv-row.main { color: #ece9e1; font-weight: 600; } .lv-row.main .tr { height: 9px; }
.lv-row span:last-child { font-family: "JetBrains Mono", monospace; text-align: right; }
.lv-m .ft { font-size: 12.5px; color: #8f93a0; margin-top: 8px; line-height: 1.45; }
.lv-log { margin-top: 10px; border-top: 1px solid rgba(214,222,240,.12); padding-top: 8px; font: 12.5px/1.6 "JetBrains Mono", monospace; color: #8f93a0; }
.lv-log b { color: #c9c6bd; font-weight: 500; }
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


def _why(r):
    p = r['scores'].get(r.get('main_model', 'combined'), float('nan'))
    v, out = r['verdict'], []
    if v == 'uncertain':
        if 0.35 <= p <= 0.65:
            out.append('score in the uncertain zone')
        if r.get('n_transits', 99) < 3:
            out.append('fewer than 3 dips')
        if 'second_dip_phase' in r:
            out.append('a second dip at the same period')
        if r.get('px_sigma') is not None and r.get('px_offset_arcsec') is not None and r['px_sigma'] > 5 and r['px_offset_arcsec'] > 10:
            out.append('the dip may come from a neighbouring star')
        if not out:
            out.append('the single models disagree with the combined score')
        return 'because: ' + ', '.join(out)
    return 'score at least 0.5' if v == 'candidate' else 'score below 0.5'


class LiveView:
    """Owns a few Streamlit placeholders and redraws them as stage events arrive.
    Each step gets an animated scene (transitvet/anim.py) drawn from that star's real intermediate data."""

    H = {'load': 300, 'flatten': 380, 'search': 420, 'fold': 420, 'check': 400, 'models': 600}

    def __init__(self, st, target, animate=True):
        self.st, self.target, self.t0 = st, target, time.time()
        self.track, self.head, self.anim = st.empty(), st.empty(), st.empty()
        self.feed, self.plot = st.empty(), st.empty()
        self.models, self.logbox = st.empty(), st.empty()
        self.timeline, self.chips, self.chip_key = [], {}, None
        self.items, self.last_draw, self.peaks, self.active = [], 0.0, [], 0
        self.animate, self.until = animate, 0.0
        self.segs = self.t_f = None; self.mission = 'kepler'
        self.search_base, self.parts, self.fold_sigs, self.check_sigs = {}, {}, [], []
        self.replay_clock = None    # set when a saved run is replayed: shows the original run's elapsed time
        self.set('download', 'Asking NASA MAST for every light curve of this star. Bigger stars with years of data take longer.')
        try:   # bring the live view into sight (same-origin iframe; harmless if it fails)
            import streamlit.components.v1 as components
            components.html("<script>setTimeout(()=>{const d=window.parent.document;const e=[...d.querySelectorAll('.lv-track')].pop();"
                            "if(e)e.scrollIntoView({behavior:'smooth',block:'start'})},400)</script>", height=0)
        except Exception:
            pass

    # ---- helpers
    def elapsed(self):
        return self.replay_clock if self.replay_clock is not None else time.time() - self.t0

    def set(self, key, say, chips=None):
        if chips is not None or not getattr(self, 'chips', None) or key != getattr(self, 'chip_key', None):
            self.chips, self.chip_key = chips or {}, key
        if not self.timeline or self.timeline[-1][1] != key:
            self.timeline.append((self.elapsed(), key))
            self._draw_log()
        self.active = _IDX[key]
        self.track.markdown(_track(self.active, self.active), unsafe_allow_html=True)
        name = STAGES[self.active][1]
        self.head.markdown(f'<div class="lv-head"><b>{self.active + 1} · {name}</b><span>{self.target} · {self.elapsed():.0f} s</span></div>'
                           f'<p class="lv-say">{say}</p>' + self._chips(), unsafe_allow_html=True)

    def _scene(self, kind, title, data, dwell, height=None):
        """Show one animated scene; first let the previous one finish playing."""
        if not self.animate:
            return
        wait = self.until - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            import streamlit.components.v1 as components
            h = height or self.H.get(kind, 400)
            with self.anim.container():
                components.html(anim.page(kind, title, data, h), height=anim.frame_height(h), scrolling=False)
        except Exception as e:      # the animation is a picture only: never stop an analysis because of it
            self.add(f'(animation skipped: {e})', 'warn')
            return
        self.until = time.time() + dwell

    def finish(self):
        """Let the last scene play before the caller collapses the view."""
        wait = self.until - time.time()
        if self.animate and wait > 0:
            time.sleep(wait)

    def _chips(self):
        if not self.chips:
            return ''
        return '<div class="lv-chips">' + ''.join(f'<span class="lv-chip">{k} <b>{v}</b></span>' for k, v in self.chips.items()) + '</div>'

    def _draw_log(self):
        rows = []
        for i, (t, k) in enumerate(self.timeline):
            if k == '__end':
                continue
            end = self.timeline[i + 1][0] if i + 1 < len(self.timeline) else None
            took = f'{end - t:5.1f} s' if end is not None else '  running'
            rows.append(f'<div><b>{int(t) // 60:02d}:{int(t) % 60:02d}</b>  {STAGES[_IDX[k]][1]:<9}{took}</div>')
        self.logbox.markdown('<div class="lv-log">Step log (time from start, time taken)' + ''.join(rows) + '</div>', unsafe_allow_html=True)

    def show(self, fig):
        self.plot.pyplot(fig, clear_figure=True); plt.close(fig)

    def add(self, text, kind=''):
        icon = {'good': '✓', 'bad': '✗', 'warn': '!', '': '•'}[kind]
        self.items.append(f'<div class="lv-item {kind}"><i>{icon}</i><span>{text}</span></div>')
        self.feed.markdown('<div class="lv-feed">' + ''.join(self.items[-8:]) + '</div>', unsafe_allow_html=True)

    # ---- stage handlers (called from the app and from pipeline.run)
    def loaded(self, segs, meta):
        self.segs, self.mission = segs, meta.get('mission', 'kepler')
        self.set('clean', f"Got {sum(len(s[0]) for s in segs):,} measurements in {len(segs)} piece(s) from {meta['mission'].upper()}. "
                          'Each piece is scaled to the same level so they join up.',
                 chips={'mission': meta['mission'].upper(), 'pieces': len(segs), 'points': f"{sum(len(x[0]) for x in segs):,}",
                        'time span': f"{max(np.nanmax(x[0]) for x in segs) - min(np.nanmin(x[0]) for x in segs):.0f} d"})
        self._scene('load', 'STEP 1–2 · DOWNLOAD AND JOIN', anim.load_data(segs, meta), anim.DWELL['load'])

    def __call__(self, name, **d):
        getattr(self, '_' + name)(**d)

    def _flatten(self, t, f):
        self.t_f = (np.asarray(t), np.asarray(f))
        self.set('flatten', f'A sliding filter follows the slow changes of the star and divides them out. What is left: {t.size:,} points, flat except for short dips.',
                 chips={'points kept': f'{t.size:,}', 'scatter': f'{1.4826e6 * np.nanmedian(np.abs(f - np.nanmedian(f))):.0f} ppm',
                        'filter': 'biweight, 1 d window'})
        if self.segs is not None:
            self._scene('flatten', 'STEP 3 · FLATTEN', anim.flatten_data(self.segs, *self.t_f), anim.DWELL['flatten'])

    def _search_range(self):
        from .pipeline import MIN_PERIOD, max_period
        t = self.t_f[0]
        return MIN_PERIOD, max_period(float(t.max() - t.min()))

    def _search(self, k, parts, n, done=None):
        now = time.time()
        done = sum(len(p[0]) for p in parts) if done is None else done   # a saved run stores thinned arrays plus the true count
        per = np.concatenate([p[0] for p in parts]); pw = np.concatenate([p[1] for p in parts])
        self.parts[k] = (per, pw, n)
        if k not in self.search_base and self.t_f is not None:
            pmin, pmax = self._search_range()
            prev = [s for kk, s in self.peaks if kk < k and s['bls_snr'] >= 7.1]
            self.search_base[k] = dict(anim.search_data(*self.t_f, k, prev, n, pmin, pmax), max_signals=MAXS)
            self._scene('search', f'STEP 4 · SEARCH {k + 1}', self.search_base[k], anim.DWELL['search'])
        if self.replay_clock is None and now - self.last_draw < 0.8 and done < n:
            return
        self.last_draw = now
        self.chips = {'search': f'{k + 1} of up to {MAXS}', 'trial orbits': f'{done:,} / {n:,}', 'range': f'{np.nanmin(per):.2f}–{np.nanmax(per):.0f} d',
                      'best so far': f'{per[np.nanargmax(pw)]:.4f} d' if pw.size else '—'}
        self.set('search', f'Search {k + 1}: sliding a box-shaped dip along the data at every trial orbit and scoring the fit. '
                           f'{done:,} of {n:,} trial orbits tried.', chips=self.chips)

    def _peak(self, k, sig):
        self.peaks.append((k, sig))
        strong = sig['bls_snr'] >= 7.1
        self.add(f"Search {k + 1}: best orbit {sig['period']:.4f} d, dip {sig['depth_ppm']:.0f} ppm, signal-to-noise {sig['bls_snr']:.1f}"
                 + ('' if strong else ' (too weak, stop searching)')
                 + (' · period doubled: every second dip was empty' if sig.get('period_doubled') else ''), 'good' if strong else '')
        if k in self.search_base and k in self.parts:
            per, pw, _ = self.parts[k]
            t = self.t_f[0]
            e = np.round((t - sig['t0']) / sig['period'])
            n_dips = int(np.unique(e[np.abs(t - sig['t0'] - e * sig['period']) < 0.5 * sig['duration']]).size)
            self._scene('search', f'STEP 4 · SEARCH {k + 1} · RESULT', anim.found_data(self.search_base[k], sig, per, pw, n_dips),
                        anim.DWELL['found_doubled' if sig.get('period_doubled') else 'found'])

    def _searched(self, sigs):
        if not sigs:
            self.add('No repeating dip strong enough was found.', 'warn')

    def _fold(self, sigs, recs):
        self.fold_sigs = sigs
        self.set('fold', 'Every orbit is stacked on top of the others. Noise averages away and the dip stands out: '
                         'one picture of the whole orbit and one close-up of the dip, the two inputs the models read.',
                 chips={'signals': len(recs), 'whole-orbit view': '2001 bins', 'close-up view': '201 bins',
                        'orbits stacked': ', '.join(str(int(r.get('n_transits', 0))) for r in recs)})
        if self.t_f is not None:
            t, f = self.t_f
            data = dict(sigs=[dict(period=s['period'], n_dips=int(r.get('n_transits', 0) or 0), tr=anim.transits(t, f, s), **anim.view_data(r))
                              for s, r in zip(sigs, recs)])
            self._scene('fold', 'STEP 5 · FOLD AND STACK', data, anim.DWELL['fold'] * len(sigs))

    def _check(self, i, sig, rec, star):
        if self.active != _IDX['check']:
            self.set('check', 'Each dip is questioned: is it the right size and shape for a planet, does the light stay on the target, '
                              'and is it already in NASA\'s lists?',
                     chips={'checks per signal': 'repeats, odd/even, second dip, size, shape, light position, pixels, catalogue'})
        rp = rec.get('rp_est_re', np.nan)
        if np.isfinite(rp):
            self.add(f'Signal {i + 1}: size about {rp:.1f} × Earth' + (' (too big for a planet)' if rp > 22.4 else ''), 'bad' if rp > 22.4 else 'good')
        fit = rec.get('fit') or {}
        if fit.get('fit_ok'):
            self.add(f"Signal {i + 1}: transit model fitted, period {fit['period']:.5f} d")
        kn = rec.get('known') or {}
        if kn.get('name'):
            self.add(f"Signal {i + 1}: in NASA's list as {kn['name']} ({kn.get('disposition', '')})")
        if self.t_f is not None:
            from .pipeline import report_card
            t, f = self.t_f
            fl = lambda k: float(rec[k]) if rec.get(k) is not None and np.isfinite(float(rec[k])) else None
            self.check_sigs.append(dict(period=sig['period'], rows=[list(x) for x in report_card(dict(rec, mission_=self.mission))],
                                        tr=anim.transits(t, f, sig), sec=anim.half_orbit(t, f, sig), oe=fl('odd_even_sigma'),
                                        ss=fl('secondary_sigma'), rp=(rec.get('fit') or {}).get('planet_radius_re', fl('rp_est_re')),
                                        cs=fl('cen_sigma'), co=fl('cen_offset_arcsec'), wander=bool((rec.get('ttv') or {}).get('wanders')),
                                        px=dict(off=fl('px_offset_arcsec'), sig=fl('px_sigma')) if fl('px_offset_arcsec') is not None else None))
            if i == len(self.fold_sigs) - 1 or len(self.check_sigs) == len(self.fold_sigs):
                self._scene('check', 'STEP 6 · CHECKS', dict(sigs=self.check_sigs, mission=self.mission), anim.DWELL['check'] * len(self.check_sigs))

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

    def _models_data(self, signals):
        out = []
        for r in signals:
            e = r.get('explain')
            if not e:
                return None
            tr = r.get('triage') or {}
            out.append(dict(period=r['period'], raw=e['raw'], cal=e['calibrated'], comb=e.get('combine'), conf=e.get('conformal'),
                            imp={k: v['importance'] for k, v in e['nets'].items()}, gbin=e['gb']['inputs'][:6],
                            combined=r['scores'].get('combined', r['scores'].get(r['main_model'])), verdict=r['verdict'], why=_why(r),
                            cp=tr.get('p'), call=tr.get('call'), override='disagree' in tr.get('text', ''), **anim.view_data(r)))
        return out

    def _verdict(self, signals):
        self.set('verdict', 'Three models read the folded pictures and the check numbers; their scores are joined into one calibrated score.')
        data = self._models_data(signals) if signals else None
        if data:
            self._scene('models', 'STEP 7 · MODELS AND VERDICT', dict(sigs=data), anim.DWELL['models'] * len(data))
            self.until = min(self.until, time.time() + anim.DWELL['models'])   # wait for the first signal only; the others play on
            self.finish()     # the verdict cards below would give the answer away before the scene has shown how it was reached
        cards = []
        for i, r in enumerate(signals):
            v = r['verdict']; p = r['scores'].get(r['main_model'], np.nan)
            cards.append(f'<div class="lv-v"><div class="k">Signal {i + 1} · {r["period"]:.3f} d</div>'
                         f'<div class="t" style="color:{COL[v]}">{v}</div><div class="k">score {p:.2f}</div></div>')
        self.track.markdown(_track(len(STAGES), len(STAGES)), unsafe_allow_html=True)
        self.models.markdown(self._model_panel(signals), unsafe_allow_html=True)
        self.timeline.append((self.elapsed(), '__end')); self._draw_log()
        self.head.markdown(f'<div class="lv-head"><b>Done</b><span>{self.target} · {self.elapsed():.0f} s</span></div>'
                           f'<div class="lv-verdicts">{"".join(cards)}</div>', unsafe_allow_html=True)

    _NAMES = {'gb': 'Gradient boosting', 'cnn': 'AstroNet CNN', 'tf': 'Transformer', 'mamba': 'Mamba-2', 'combined': 'Combined'}

    def _model_panel(self, signals):
        """What each model said, how they were joined, and what the guarantee decided: the full path to the verdict."""
        out = []
        for i, r in enumerate(signals):
            sc = r.get('scores', {}); main = r.get('main_model', 'combined')
            rows = ''.join(f'<div class="lv-row{" main" if k == main else ""}"><span>{self._NAMES.get(k, k)}</span>'
                           f'<span class="tr"><i style="width:{100 * v:.1f}%"></i></span><span>{v:.2f}</span></div>'
                           for k, v in sc.items() if v == v)
            tr = r.get('triage') or {}
            g = f"95% guarantee: <b>{tr.get('call', '—')}</b>. " if tr else ''
            out.append(f'<div class="lv-m"><div class="hd"><span>Signal {i + 1} · {r["period"]:.3f} d</span>'
                       f'<span style="color:{COL[r["verdict"]]}">{r["verdict"]}</span></div>{rows}'
                       f'<div class="ft">Each bar is one model\'s calibrated chance of a planet candidate (line = 0.5). '
                       f'{self._NAMES.get(main, main)} decides the verdict. {g}</div></div>')
        return ('<div class="lv-say" style="margin-top:12px">How the verdict was reached</div><div class="lv-models">' + ''.join(out) + '</div>') if out else ''
