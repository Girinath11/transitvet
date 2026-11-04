import sys, json, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/tv/app')
from transitvet import pipeline as P
sc = P.Scorer(); segs, meta = P.load_star('TIC 261136679', 'tess')
t_raw, f_raw = segs[0]
res = P.run(segs, meta, sc, progress=lambda m: None)
t, f = res['search_curve']
s = [r for r in res['signals'] if abs(r['period'] - 6.268) < 0.05][0]
sel = (t >= t_raw.min()) & (t <= t_raw.max())
pg = res['periodograms'][1] if abs(res['signals'][0]['period'] - 6.268) > 0.05 else res['periodograms'][0]
idx = [i for i, r in enumerate(res['signals']) if r is s][0]; pg = res['periodograms'][idx]
k = np.linspace(0, len(pg['period']) - 1, 1200).astype(int)
out = dict(raw=dict(t=t_raw.round(4).tolist(), f=(f_raw / np.nanmedian(f_raw)).round(6).tolist()),
           flat=dict(t=t[sel].round(4).tolist(), f=f[sel].round(6).tolist()), allflat=dict(t=t.round(3).tolist(), f=((f - 1) * 1e6).round(0).astype(int).tolist()),
           pg=dict(p=np.asarray(pg['period'])[k].round(4).tolist(), pow=np.asarray(pg['power'])[k].round(3).tolist()),
           period=s['period'], t0=s['t0'], duration=s['duration'], depth_ppm=s['depth_ppm'],
           glob=np.asarray(s['global_astronet']).round(4).tolist(), loc=np.asarray(s['local_astronet']).round(4).tolist(),
           scores=s['scores'], verdict=s['verdict'], exo=s.get('exominer'), known=s.get('known'), card=P.report_card(s),
           fit={k2: v for k2, v in (s.get('fit') or {}).items()}, n_sectors=len(segs), n_points=res['n_points'], span=res['time_span_days'])
json.dump(out, open('/home/user/tv/show/pimen.json', 'w'), default=float); print('ok')
