"""Synthetic test file: real Kepler light curve of a quiet Sun-like star with no known signal (KIC 9139151, quarters 1-8),
with a fake planet added by batman. The answer key is written into the file header."""
import numpy as np, pandas as pd, lightkurve as lk, batman, warnings
warnings.filterwarnings('ignore')
r = lk.search_lightcurve('KIC 9139151', mission='Kepler', author='Kepler', exptime=1800)
import re
r = r[[int(re.search(r'(\d+)', str(q)).group(1)) in range(1, 9) for q in r.table['mission']]]
lcs = r.download_all(quality_bitmask='hard', flux_column='pdcsap_flux')
rows = []
for lc in lcs:
    t = np.asarray(lc.time.value, float); f = np.asarray(lc.flux.value, float); e = np.asarray(lc.flux_err.value, float)
    ok = np.isfinite(t) & np.isfinite(f); med = np.nanmedian(f[ok])
    rows.append(pd.DataFrame(dict(time=t[ok], flux=f[ok] / med, flux_err=e[ok] / med)))
df = pd.concat(rows).sort_values('time')
P, T0, RP = 9.3, df.time.min() + 2.1, 0.016          # radius ratio 0.016: about 2 Earth radii on this star, about 300 ppm deep
pm = batman.TransitParams(); pm.t0 = T0; pm.per = P; pm.rp = RP; pm.a = 19.0; pm.inc = 89.0; pm.ecc = 0.; pm.w = 90.
pm.u = [0.4, 0.25]; pm.limb_dark = 'quadratic'
m = batman.TransitModel(pm, df.time.values, supersample_factor=7, exp_time=29.4 / 1440).light_curve(pm)
depth = 1 - m.min(); df['flux'] = df.flux * m
n = len({int(round((x - T0) / P)) for x in df.time.values[m < 1]})
import os
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'KIC 9139151 with fake planet.csv'), 'w') as fh:
    fh.write(f'# SYNTHETIC TEST FILE: real Kepler light curve (PDCSAP, quarters 1-8, each divided by its median) of a quiet Sun-like star,\n'
             f'# KIC 9139151, which has no known transit signal, with a fake planet added\n'
             f'# fake planet: period {P} d, first dip at time {T0:.4f}, radius ratio {RP} (depth about {depth*1e6:.0f} ppm), {n} dips in the data\n'
             f'# time in BKJD (BJD - 2454833) days\n')
    df.to_csv(fh, index=False, float_format='%.7f')
print(len(df), len(lcs), n, T0, depth)
