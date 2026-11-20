"""Download the 9 real example light curves and save them as upload-ready CSV files (see ../README.md).
Run from any folder: python download_examples.py  (writes next to this script's parent folder)."""
import os, numpy as np, pandas as pd, lightkurve as lk, warnings
warnings.filterwarnings('ignore')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
STARS = [  # (file name, target, mission, sectors/quarters limit, note)
    ('KIC 3847138', 'kepler', None),
    ('KIC 6046540', 'kepler', None),
    ('KIC 9291378', 'kepler', None),
    ('KIC 6341932', 'kepler', None),
    ('TIC 19342878', 'tess', 3),
    ('TIC 1129033', 'tess', 3),
    ('TIC 357416482', 'tess', 3),
    ('TIC 87597187', 'tess', 3),
    ('TIC 261136679', 'tess', 3),
]
def get(target, mission, nmax):
    if mission == 'kepler':
        r = lk.search_lightcurve(target, mission='Kepler', author='Kepler', cadence='long')
    else:
        r = lk.search_lightcurve(target, mission='TESS', author='SPOC', exptime=120)
    if nmax: r = r[:nmax]
    lcs = r.download_all(quality_bitmask='hard', flux_column='pdcsap_flux')
    rows = []
    for lc in lcs:
        t = np.asarray(lc.time.value, float); f = np.asarray(lc.flux.value, float); e = np.asarray(lc.flux_err.value, float)
        ok = np.isfinite(t) & np.isfinite(f)
        med = np.nanmedian(f[ok])
        rows.append(pd.DataFrame(dict(time=t[ok], flux=f[ok] / med, flux_err=e[ok] / med)))
    return pd.concat(rows).sort_values('time')
for target, mission, nmax in STARS:
    p = os.path.join(OUT, target + '.csv')
    if os.path.exists(p): continue
    df = get(target, mission, nmax)
    unit = 'BKJD (BJD - 2454833)' if mission == 'kepler' else 'BTJD (BJD - 2457000)'
    with open(p, 'w') as fh:
        fh.write(f'# {target} ({mission}), PDCSAP flux from MAST, each quarter/sector divided by its median\n# time in {unit} days\n')
        df.to_csv(fh, index=False, float_format='%.7f')
    print(target, len(df))
