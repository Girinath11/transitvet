"""Star properties from the TESS Input Catalog (TIC v8.2 at MAST) for all labelled Kepler and TESS stars."""
import json, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from astroquery.mast import Catalogs
COLS = ['ID', 'KIC', 'rad', 'e_rad', 'Teff', 'logg', 'Tmag', 'contratio', 'mass']
def q(**kw):
    for a in range(4):
        try: return Catalogs.query_criteria(catalog='Tic', **kw)[COLS].to_pandas()
        except Exception as e: err = e; time.sleep(2 ** a)
    raise err
kep = pd.read_csv('/home/user/tv/out/labels/q1_q17_dr24_tce.csv')
kids = sorted(kep[kep.av_training_set.isin(['PC', 'AFP', 'NTP'])].kepid.unique().astype(int))
toi = pd.read_csv('/home/user/tv/out/labels/toi.csv'); tids = sorted(toi.tid.dropna().astype(int).unique())
out = []
for i in range(0, len(kids), 400):
    out.append(q(KIC=[str(k) for k in kids[i:i + 400]])); print('kic', i, flush=True)
for i in range(0, len(tids), 400):
    out.append(q(ID=[str(k) for k in tids[i:i + 400]])); print('tic', i, flush=True)
df = pd.concat(out).drop_duplicates('ID'); df.to_csv('/home/user/tv/out/labels/tic_props.csv', index=False)
print(len(df), 'rows;', df.KIC.notna().sum(), 'with KIC; rad missing', df.rad.isna().mean())
