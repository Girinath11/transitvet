"""TransitVet dashboard.  Run:  streamlit run app.py"""
import gc, json, os, sys, io, zipfile
os.environ.setdefault('TV_LAZY_MODELS', '1')   # load only the networks a star needs (less memory, e.g. on Streamlit Cloud)
import numpy as np, pandas as pd, streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from transitvet import pipeline as P, plots, extras as E, planner as PL, ui_theme as UI, live as LV, examples as EX

st.set_page_config(page_title='TransitVet', page_icon='🪐', layout='wide')
st.markdown(UI.CSS, unsafe_allow_html=True)


with st.sidebar:
    st.header('Your telescope')
    st.caption('Used by the follow-up planner. Default: Coimbatore, India.')
    SITE = dict(lat=st.number_input('Latitude (°N)', -90.0, 90.0, 10.94, 0.01, key='lat'),
                lon=st.number_input('Longitude (°E)', -180.0, 180.0, 76.74, 0.01, key='lon'),
                height_m=st.number_input('Height above sea (m)', 0.0, 5000.0, 450.0, 10.0, key='hgt'),
                aperture_cm=st.number_input('Telescope aperture (cm)', 5.0, 400.0, 20.0, 1.0, key='ap'),
                tz_hours=st.number_input('Time zone (hours from UTC)', -12.0, 14.0, 5.5, 0.5, key='tz'),
                nights=int(st.slider('Nights ahead', 1, 60, 14, key='nights')))
    st.caption('Planning estimates only: weather, the moon, comparison stars and your camera also matter.')
    st.header('Checks')
    PIXEL = st.checkbox('Pixel check (camera images)', value=True,
                        help='Downloads the Target Pixel Files (TESS: about 100 MB per sector, up to 3 sectors; Kepler: about 1 MB per quarter) '
                             'and makes difference images to see whether the dip comes from the target star or a neighbour.')


def plan_signal(res, r):
    """Observable dips of one signal from the site in the sidebar (empty list if no star position)."""
    stp = res.get('star') or {}
    if not all(np.isfinite(stp.get(k, np.nan)) for k in ('ra', 'dec')):
        return None, 'star position unknown (no TIC match)'
    if PL.never_visible(stp['dec'], SITE['lat']):
        return [], f"never above 30° from latitude {SITE['lat']:.1f}° (star declination {stp['dec']:.0f}°)"
    m = res['meta']['mission']; ft = r.get('fit') or {}
    per = ft['period'] if ft.get('fit_ok') else r['period']; t0 = (ft['t0'] if ft.get('fit_ok') else r['t0']) + E.TIME_OFFSET[m]
    err_h = 0.1 * r['duration'] / max(res['time_span_days'] / per, 1.0) * 24
    vmag = stp.get('vmag', np.nan)
    if not np.isfinite(vmag):
        vmag = stp.get('tmag', np.nan) + 0.5      # rough V from TESS magnitude
    ev = PL.plan(stp['ra'], stp['dec'], per, t0, r['duration'], err_h, r.get('depth_ppm', np.nan), vmag,
                 SITE['lat'], SITE['lon'], SITE['height_m'], SITE['aperture_cm'], SITE['nights'], SITE['tz_hours'])
    return ev, ''


@st.cache_resource(show_spinner='Loading models...')
def get_scorer():
    return P.Scorer()


def _show(fig, **kw):
    """Draw a matplotlib figure, then free it (Streamlit redraws on every click; open figures would pile up in memory)."""
    st.pyplot(fig, **kw); plots.plt.close(fig)


def analyse(loader, *args):
    with st.status(f'Analysing {args[0]} ... watch each step below', expanded=True) as box:
        view = LV.LiveView(st, args[0])
        with st.expander('Technical log', expanded=False):
            log = st.empty()
        lines = []
        def progress(msg):
            lines.append(msg); log.code('\n'.join(lines[-12:]))
        progress('Downloading the light curve from NASA MAST ...')
        segs, meta = loader(*args)
        meta['_pixel'] = PIXEL
        progress(f"Loaded {meta['target']} ({meta['mission'].upper()}, {meta['n_files']} file(s), {sum(len(s[0]) for s in segs):,} points)")
        view.loaded(segs, meta)
        res = P.run(segs, meta, get_scorer(), progress, stage=view)
        box.update(label=f"{meta['target']}: {len(res['signals'])} signal(s) in {res['seconds']:.0f} s · open to replay the steps", state='complete', expanded=False)
    gc.collect()
    return res


def replay_saved(blob):
    """Example stars: replay a saved real run (see transitvet/examples.py) instead of re-downloading and re-searching."""
    tg = blob['target']
    with st.status(f'{tg}: replaying a saved run of the full analysis ...', expanded=True) as box:
        view = LV.LiveView(st, tg + ' · replay')
        res = EX.replay(view, blob)
        box.update(label=f"{tg}: {len(res['signals'])} signal(s) · saved run from {blob['made']} (took {blob['events'][-1][0]:.0f} s) · open to replay the steps",
                   state='complete', expanded=False)
    return res


def show_result(res, key):
    scorer = get_scorer()
    m = res['meta']
    st.subheader(f"{m['target']}  ·  {m['mission'].upper()}")
    st.caption(f"{res['n_points']:,} points over {res['time_span_days']:.0f} days · analysed in {res['seconds']:.0f} s · source: {m['source']}")
    stp = res.get('star') or {}
    if stp:
        fmt = lambda v, f, u='': (format(v, f) + u) if v == v else 'unknown'
        st.caption(f"Star (TIC {stp.get('tic')}): radius {fmt(stp.get('star_rad', float('nan')), '.2f', ' Sun')} · "
                   f"temperature {fmt(stp.get('star_teff', float('nan')), '.0f', ' K')} · "
                   f"light from neighbours {fmt(100 * stp.get('contratio', float('nan')), '.1f', '%')} (TESS Input Catalog)")
    if not res['signals']:
        st.info('No repeating dip strong enough was found (BLS signal-to-noise below 7.1).')
        _show(plots.lightcurve_figure(res)); return None
    table = P.summary_table(res)
    c = st.columns(3)
    for col, v in zip(c, ['candidate', 'uncertain', 'false alarm']):
        col.metric({'candidate': 'Candidates', 'uncertain': 'Uncertain', 'false alarm': 'False alarms'}[v], int((table.verdict == v).sum()))
    st.dataframe(table[['signal', 'verdict', 'trust_check_95', 'p_main', 'catalogue', 'catalogue_name', 'period_days', 'planet_size_earth_radii',
                        'duration_hours', 'depth_ppm', 'nasa_exominer_pp', 'p_gradient_boosting', 'p_cnn', 'p_transformer', 'light_shift_sigma',
                        'source_offset_arcsec', 'pixel_offset_arcsec', 'pixel_sigma', 'flags']], hide_index=True, width='stretch')
    _show(plots.lightcurve_figure(res))
    for i, r in enumerate(res['signals'], 1):
        with st.expander(f"Signal {i}: {r['verdict'].upper()} · period {r['period']:.4f} d · depth {r['depth_ppm']:.0f} ppm", expanded=i == 1):
            kn = r.get('known', {})
            msg = f"**Official lists ({E.CATALOG_DATE}):** {kn.get('status', 'not checked')}" + (f" · {kn['name']} ({kn['disposition']})" if kn.get('name') else '')
            (st.success if 'known planet' in kn.get('status', '') else st.warning if 'false positive' in kn.get('status', '') else st.info)(msg)
            exm = r.get('exominer')
            if exm:
                if exm.get('score') == exm.get('score'):
                    st.write(f"**NASA ExoMiner++ second opinion** (published catalogue, TESS sectors 1-67): score **{exm['score']:.2f}** "
                             f"(longest NASA pipeline run {exm['best_run']}; {exm['n_runs']} run(s) range {exm['score_range']}); NASA label: {exm['nasa_label']}"
                             + (f"; {exm['note']}" if exm['note'] else '') + '. Not used in our score.')
                else:
                    st.caption('NASA ExoMiner++: ' + exm['note'])
            _show(plots.signal_figure(r, i))
            s = r['scores']
            st.write(f"Chance of being a planet candidate (calibrated): **{P.MODEL_NAMES[r['main_model']]} {s[r['main_model']]:.2f}** "
                     f"(main model for {m['mission'].upper()}) · " + ' · '.join(f"{P.MODEL_NAMES[k]} {v:.2f}" for k, v in s.items() if k != r['main_model']))
            c1, c2 = st.columns([3, 2])
            icon = {'pass': '✅ looks like a planet', 'warn': '⚠️ points to a false alarm', 'n/a': '➖ could not check'}
            c1.markdown('**Report card**')
            c1.dataframe(pd.DataFrame([dict(check=a, found=b, result=icon[c]) for a, b, c in P.report_card(r)]),
                         hide_index=True, width='stretch')
            ft = r.get('fit') or {}
            if ft.get('fit_ok'):
                c2.markdown('**Transit fit**')
                c2.write(f"Period {ft['period']:.5f} ± {ft['period_err']:.5f} d  \nSize ratio {ft['rp_rs']:.3f} ± {ft['rp_rs_err']:.3f}"
                         + (f"  \nPlanet size {ft['planet_radius_re']:.1f} ± {ft['planet_radius_err']:.1f} Earth radii" if 'planet_radius_re' in ft else ''))
            c2.markdown('**Next dips (UTC, rough)**')
            c2.dataframe(pd.DataFrame(r.get('next_dips', [])), hide_index=True, width='stretch')
            pf = plots.pixel_figure(r)
            if pf is not None:
                st.markdown('**Where does the dip come from?** (difference imaging)')
                _show(pf, width='content')
                src = r.get('px_source') or {}
                if src.get('tic'):
                    st.caption(f"Closest catalogue star to the dip's source: TIC {src['tic']} (TESS magnitude {src['tmag']:.1f}), "
                               f"{src['sep_from_target_arcsec']:.0f} arcsec from the target.")
            for fl in r['flags']:
                st.warning(fl)
            try:
                card = plots.share_card(m['target'], m['mission'], r, i, P.report_card(r))
                st.download_button('Share this result (image)', card, f"transitvet_{m['target'].replace(' ', '_')}_signal{i}.png",
                                   mime='image/png', key=f'card{key}{i}', on_click='ignore')
            except Exception as e:
                st.caption(f'Share image unavailable: {e}')
            if r['verdict'] != 'false alarm':
                st.markdown(f"**Can you observe the next dips?** ({SITE['aperture_cm']:.0f} cm telescope, next {SITE['nights']} nights)")
                try:
                    ev, why = plan_signal(res, r)
                    if ev is None or not ev:
                        st.caption(why or 'No predicted dips in this window.')
                    else:
                        d = pd.DataFrame(ev)
                        good = d[d.observable]
                        if good.empty:
                            st.caption('No dip in this window happens at night with the star high enough. Try more nights.')
                        else:
                            st.dataframe(good[['mid_local', 'start_local', 'end_local', 'min_altitude', 'snr', 'rating', 'timing_err_min']].rename(columns={
                                'mid_local': 'dip middle (local)', 'start_local': 'start watching', 'end_local': 'stop',
                                'min_altitude': 'lowest altitude (°)', 'snr': 'expected SNR', 'rating': 'detectable?', 'timing_err_min': 'timing ± (min)'}),
                                hide_index=True, width='stretch')
                            g2 = good.assign(target=res['meta']['target'], signal=i, verdict=r['verdict'], period_days=round(r['period'], 4))
                            st.session_state.setdefault('_plans_tmp', []).append(
                                g2[['target', 'signal', 'verdict', 'period_days', 'mid_local', 'start_local', 'end_local', 'min_altitude', 'snr', 'rating']])
                except Exception as e:
                    st.caption(f'Planner unavailable: {e}')
    man = P.manifest(res, scorer)
    d1, d2 = st.columns(2)
    d1.download_button('Download table (CSV)', table.to_csv(index=False), f"transitvet_{m['target'].replace(' ', '_')}.csv", key=f'csv{key}', on_click='ignore')
    d2.download_button('Download run manifest (JSON)', json.dumps(man, indent=1, default=str), f"transitvet_{m['target'].replace(' ', '_')}_manifest.json", key=f'man{key}', on_click='ignore')
    return table


st.markdown(UI.HERO, unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(['One star', 'Batch', 'About the models'])

EXAMPLES = [('TIC 1333933', 'Known hot Jupiter (TOI-7719.01)', 'saved run, ~15 s'),
            ('TIC 184952758', 'A dip that comes from a nearby star', 'saved run, ~15 s'),
            ('TIC 261136679', 'Pi Mensae: a small planet, 7 years of data', 'saved run, ~15 s'),
            ('KIC 11442793', 'Kepler-90: a star with many planets', 'saved run, ~15 s')]


def _pick(t):
    st.session_state['target'] = t
    st.session_state['_go'] = True


with tab1:
    c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment='bottom')
    target = c1.text_input('Star', placeholder='e.g. TIC 1333933 or KIC 11442793', key='target')
    mission = c2.selectbox('Mission', ['auto', 'kepler', 'tess'])
    go = c3.button('Analyse', type='primary', width='stretch') or st.session_state.pop('_go', False)
    if 'res_one' not in st.session_state and not go:
        st.markdown(UI.WELCOME, unsafe_allow_html=True)
        st.caption('Try an example:')
        ex = st.columns(len(EXAMPLES))
        for col, (tid, what, dur) in zip(ex, EXAMPLES):
            col.button(f'{tid}', key=f'ex_{tid}', help=f'{what} ({dur})', on_click=_pick, args=(tid,), width='stretch')
            col.caption(f'{what} · {dur}')
    if go and target:
        try:
            live = st.session_state.pop('_live', False)
            blob = None if live or mission != 'auto' else EX.saved(target)
            st.session_state['res_saved'] = blob['made'] if blob else None
            st.session_state['res_one'] = replay_saved(blob) if blob else analyse(P.load_star, target, None if mission == 'auto' else mission)
        except Exception as e:
            st.session_state.pop('res_one', None)
            st.error(f'Could not analyse {target}: {e}. Check the ID (KIC or TIC number) and your internet connection.')
    if 'res_one' in st.session_state:          # kept across reruns, so the result stays while you use other controls
        if st.session_state.get('res_saved'):
            a, b = st.columns([4, 1], vertical_alignment='center')
            a.caption(f"This example opened from a saved run of the full analysis ({st.session_state['res_saved']}), so you don't wait. "
                      'Same code and models as a live run.')
            b.button('Run it live', key='run_live', on_click=lambda: st.session_state.update(_live=True, _go=True), width='stretch')
        show_result(st.session_state['res_one'], 'one')
    st.divider()
    up = st.file_uploader('...or upload one light-curve file (Kepler/TESS FITS, or CSV with time and flux columns)',
                          type=['fits', 'fit', 'csv', 'txt'], key='one_up')
    up_m = st.selectbox('Mission of the uploaded file (CSV only)', ['kepler', 'tess'], key='one_m')
    if up is not None and st.button('Analyse file'):
        try:
            show_result(analyse(P.load_file, up.name, up.getvalue(), up_m if up.name.lower().endswith(('csv', 'txt')) else None), 'file')
        except Exception as e:
            st.error(f'Could not analyse {up.name}: {e}')

with tab2:
    st.write('Analyse many stars at once: paste star IDs (one per line), or upload several light-curve files or a ZIP of them.')
    ids = st.text_area('Star IDs', placeholder='KIC 11442793\nKIC 8462852\nTIC 261136679')
    files = st.file_uploader('Light-curve files', type=['fits', 'fit', 'csv', 'txt', 'zip'], accept_multiple_files=True, key='batch_up')
    b_m = st.selectbox('Mission for CSV files', ['kepler', 'tess'], key='b_m')
    if st.button('Run batch', type='primary'):
        jobs = [(P.load_star, (x.strip(), None)) for x in ids.splitlines() if x.strip()]
        for f in files or []:
            if f.name.lower().endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z:
                    jobs += [(P.load_file, (n, z.read(n), b_m)) for n in z.namelist() if n.lower().endswith(('.fits', '.fit', '.csv', '.txt'))]
            else:
                jobs.append((P.load_file, (f.name, f.getvalue(), b_m)))
        tables, bar = [], st.progress(0.0)
        st.session_state['_plans_tmp'] = []
        for k, (fn, args) in enumerate(jobs):
            name = args[0]
            try:
                with st.expander(f'{name}', expanded=False):
                    t = show_result(analyse(fn, *args), f'b{k}')
                if t is not None:
                    tables.append(t)
            except Exception as e:
                st.error(f'{name}: {e}')
            bar.progress((k + 1) / max(len(jobs), 1))
        if tables:
            allt = pd.concat(tables).sort_values('p_main', ascending=False)
            st.session_state['batch_table'] = allt
            st.session_state['batch_plans'] = list(st.session_state.get('_plans_tmp', []))
    if 'batch_table' in st.session_state:
        allt = st.session_state['batch_table']
        st.subheader('All signals, ranked by score')
        if st.checkbox('Show only candidates that are not in the official lists (the ones worth a closer look)'):
            allt = allt[(allt.verdict == 'candidate') & allt.catalogue.str.startswith('not in')]
        st.dataframe(allt, hide_index=True, width='stretch')
        plans = st.session_state.get('batch_plans', [])
        if plans:
            st.subheader('Observing plan: dips you can catch')
            pl = pd.concat(plans).sort_values(['mid_local'])
            st.caption(f"From latitude {SITE['lat']:.2f}, longitude {SITE['lon']:.2f}, {SITE['aperture_cm']:.0f} cm telescope. Sorted by date; 'clear' dips first are the best targets.")
            st.dataframe(pl, hide_index=True, width='stretch')
        st.download_button('Download ranked table (CSV)', allt.to_csv(index=False), 'transitvet_batch.csv', on_click='ignore')

with tab3:
    st.markdown("""
**What happens to each star**
1. Light curve from NASA's MAST archive (Kepler long cadence, or TESS 2-minute data averaged to 30 minutes).
2. Slow brightness changes removed (biweight filter for the search; AstroNet's spline for the model inputs).
3. Box Least Squares search for repeating dips (up to 3 per star, signal-to-noise at least 7.1).
4. Each dip folded into a whole-orbit view and a zoomed view, plus check numbers (odd/even dips, dip at half phase, shape).
5. New checks: does the star's light move on the camera during the dip (light-position check)? How big is the object, given the
   star's size from the TESS Input Catalog? How much light comes from neighbouring stars?
6. Three models score it (gradient boosting with all check numbers, AstroNet CNN, Transformer), and a **combined score** joins them.
   Scores are calibrated on validation stars, so 0.8 means "about 80% of such signals were planet candidates".
7. A transit model is fitted for a sharper period, the planet size with an error bar, and the dip shape.
8. Each signal is matched against the official Kepler (KOI) and TESS (TOI) lists, for display only (never used in the score).
9. For TESS signals, NASA's ExoMiner++ score is shown as a second opinion when the signal is in its published catalogue
   (TESS 2-min data, sectors 1-67; Valizadegan, Martinho et al. 2025). On our 542 shared test signals: ExoMiner++ PR-AUC 0.973,
   TransitVet 0.967 (difference within the uncertainty); the average of the two: 0.981.
10. **Trust check**: split conformal prediction turns the combined score into keep / drop / needs a human, with a guarantee
   that at least 95% of real planets are never dropped (measured on test stars: Kepler 95.8%, TESS 97.8%).
11. **Pixel check** (difference images): which camera pixels get darker during the dip? On TESS, a dip coming from more than
   10 arcsec away (over 5 sigma) raises a warning and turns a candidate into 'uncertain' (test stars: 25% of false alarms flagged, 3.5% of planets).
12. **Follow-up planner**: for your site and telescope (sidebar), the next dips that are at night with the star above 30 degrees,
   and the expected signal-to-noise (clear / marginal / too faint).

**Main score**: the combined score. In our held-out test it reached PR-AUC 0.968 on Kepler (AstroNet CNN alone: 0.947) and 0.946 on TESS (best single model before: 0.883).
**Verdict**: candidate if the combined score is at least 0.5, false alarm if below; *uncertain* if the score is between 0.35 and 0.65, all three single models disagree with it,
fewer than 3 dips were seen, or a second separate dip shows up at the same period.

**Known limits**: the light-position check cannot see a nearby star closer than about a pixel (4 arcsec Kepler, 21 arcsec TESS);
TESS labels are fewer and scores there are less reliable; the period found by the search can differ from the official catalogue period;
next-dip times get less exact the further they are from the data.
""")
    st.json(get_scorer().manifest)
