"""MOMENT-1-small (AutonLab, 2024) embeddings of the global and local views, for Kepler and TESS.
Each view is resampled to MOMENT's 512-sample input; MOMENT normalises each series itself (RevIN).
The pretrained model is frozen: nothing here is fitted to our labels."""
import sys, os, time, numpy as np, torch
sys.path.insert(0, '/home/user/tv/src'); import tv_common as c
from momentfm import MOMENTPipeline
torch.set_num_threads(int(os.environ.get('TV_THREADS', '1')))
mis = sys.argv[1]
G, L, t = c.load_kepler() if mis == 'kepler' else c.load_tess()
m = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-small", model_kwargs={'task_name': 'embedding'}); m.init(); m.eval()
def rs(X):
    X = np.nan_to_num(X.astype(np.float32)); xs = np.linspace(0, 1, X.shape[1]); xn = np.linspace(0, 1, 512)
    return np.stack([np.interp(xn, xs, r) for r in X]).astype(np.float32)
out = {}
for name, X in (('g', G), ('l', L)):
    X = rs(X); E = []; t0 = time.time()
    for i in range(0, len(X), 64):
        with torch.no_grad(): E.append(m(x_enc=torch.tensor(X[i:i + 64])[:, None]).embeddings.numpy())
        if i % 3200 == 0: print(mis, name, i, len(X), f'{time.time() - t0:.0f}s', flush=True)
    out[name] = np.concatenate(E)
np.savez(f'/home/user/tv/fm/moment_{mis}.npz', **out); print('saved', {k: v.shape for k, v in out.items()}, flush=True)
