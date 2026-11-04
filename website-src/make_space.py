"""Procedural deep-space backgrounds (nebulae, galaxy band, deep field) rendered to JPEG.
Fractal noise + colour ramps + stars with a point-spread function and diffraction spikes."""
import numpy as np
from scipy import ndimage
from PIL import Image, ImageFilter
rng = np.random.default_rng(42)

def fbm(h, w, octaves=7, base=4, persistence=0.55, seed=0):
    r = np.random.default_rng(seed); out = np.zeros((h, w)); amp = 1.0; tot = 0
    for o in range(octaves):
        n = base * 2 ** o
        g = r.random((n * h // w + 2, n + 2))
        up = ndimage.zoom(g, (h / g.shape[0] * 1.02, w / g.shape[1] * 1.02), order=3)[:h, :w]
        out += amp * up; tot += amp; amp *= persistence
    out /= tot
    return (out - out.min()) / (out.max() - out.min())

def ridged(n):
    return 1 - np.abs(2 * n - 1)

def ramp(x, stops):
    xs = [s[0] for s in stops]; cols = np.array([s[1] for s in stops], float)
    return np.stack([np.interp(x, xs, cols[:, c]) for c in range(3)], -1)

def stars(h, w, n, seed, warm=0.25, big=12):
    r = np.random.default_rng(seed); img = np.zeros((h, w, 3))
    ys, xs = r.integers(0, h, n), r.integers(0, w, n)
    mag = r.power(6, n) ** 6
    col = np.where(r.random(n)[:, None] < warm, [1.0, 0.85, 0.65], np.where(r.random(n)[:, None] < 0.3, [0.7, 0.82, 1.0], [1, 1, 1]))
    for y, x, m, c in zip(ys, xs, mag, col):
        img[y, x] += c * (0.25 + 1.6 * m)
    img = ndimage.gaussian_filter(img, (0.7, 0.7, 0)) * 3.0
    # a few bright stars with halo + diffraction spikes
    for _ in range(big):
        y, x = r.integers(0, h), r.integers(0, w); c = np.array([1, 0.92, 0.8]) if r.random() < .5 else np.array([0.8, 0.9, 1])
        L = int(r.integers(40, 140)); a = r.uniform(.6, 1.2)
        for d in range(-L, L + 1):
            f = a * np.exp(-abs(d) / (L / 3.5))
            if 0 <= x + d < w: img[y, x + d] += c * f
            if 0 <= y + d < h: img[y + d, x] += c * f
        yy, xx = np.ogrid[-30:31, -30:31]; halo = a * 1.5 * np.exp(-(xx ** 2 + yy ** 2) / 30.0)
        y0, y1, x0, x1 = max(0, y - 30), min(h, y + 31), max(0, x - 30), min(w, x + 31)
        img[y0:y1, x0:x1] += halo[(y0 - y + 30):(y1 - y + 30), (x0 - x + 30):(x1 - x + 30), None] * c
    return img

def save(arr, path, q=82):
    arr = np.clip(arr, 0, 1) ** (1 / 1.1)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path, quality=q, optimize=True, progressive=True)

H, W = 1080, 1920
# 1. hero: amber-teal emission nebula, dense on the right
n1 = fbm(H, W, seed=1); n2 = fbm(H, W, seed=2, base=3); n3 = ridged(fbm(H, W, seed=3, base=5))
xx = np.linspace(0, 1, W)[None, :]; yy = np.linspace(0, 1, H)[:, None]
mask = np.clip(1.4 * np.exp(-((xx - 0.78) ** 2 / 0.09 + (yy - 0.42) ** 2 / 0.18)) , 0, 1)
gas = np.clip((n1 * 0.55 + n3 * 0.55 - 0.42) * 1.35, 0, 1) * mask ** 1.5
dust = np.clip((n2 - 0.45) * 3, 0, 1) * mask
c = ramp(gas, [(0, (0.012, 0.012, 0.04)), (0.3, (0.08, 0.05, 0.26)), (0.55, (0.32, 0.17, 0.62)), (0.75, (0.82, 0.36, 0.72)), (1, (0.78, 0.94, 1.0))])
c *= (1 - 0.75 * dust[..., None])
c += stars(H, W, 5200, 11, big=10)
save(c, 'img/hero.jpg')
# 2. galaxy band: diagonal milky-way stripe with dust lanes
band = np.exp(-((yy - 0.5 - (xx - 0.5) * 0.45) ** 2) / 0.018)
g1 = fbm(H, W, seed=5, base=6); g2 = ridged(fbm(H, W, seed=6, base=7))
glow = band * (0.4 + 0.8 * g1)
lane = np.clip((g2 - 0.55) * 4, 0, 1) * np.exp(-((yy - 0.5 - (xx - 0.5) * 0.45) ** 2) / 0.006)
c = ramp(np.clip(glow, 0, 1), [(0, (0.01, 0.012, 0.03)), (0.4, (0.16, 0.13, 0.22)), (0.75, (0.55, 0.45, 0.42)), (1, (0.95, 0.88, 0.78))])
c *= (1 - 0.85 * lane[..., None]); c += stars(H, W, 14000, 12, warm=0.35, big=6) * (0.6 + band[..., None])
save(c, 'img/galaxy.jpg')
# 3. violet-magenta nebula, left side
m = np.clip(1.3 * np.exp(-((xx - 0.2) ** 2 / 0.12 + (yy - 0.6) ** 2 / 0.25)), 0, 1)
p1 = fbm(H, W, seed=8); p2 = ridged(fbm(H, W, seed=9, base=4)); p3 = fbm(H, W, seed=10, base=2)
gas = np.clip((p1 * 0.5 + p2 * 0.6 - 0.38) * 1.9, 0, 1) * m
c = ramp(gas, [(0, (0.008, 0.015, 0.03)), (0.35, (0.03, 0.14, 0.24)), (0.6, (0.08, 0.42, 0.55)), (0.82, (0.45, 0.35, 0.95)), (1, (0.85, 0.95, 1.0))])
c *= (1 - 0.6 * np.clip((p3 - 0.5) * 3, 0, 1)[..., None]); c += stars(H, W, 5000, 13, big=8)
save(c, 'img/nebula2.jpg')
# 4. deep field: tiny galaxies
c = np.zeros((H, W, 3)) + np.array([0.006, 0.008, 0.016])
r = np.random.default_rng(20)
for _ in range(420):
    y, x = r.integers(0, H), r.integers(0, W); s = r.uniform(1.2, 6); e = r.uniform(.3, 1); th = r.uniform(0, np.pi)
    yy2, xx2 = np.ogrid[-20:21, -20:21]; u = xx2 * np.cos(th) + yy2 * np.sin(th); v = (-xx2 * np.sin(th) + yy2 * np.cos(th)) / e
    blob = np.exp(-(u ** 2 + v ** 2) / (2 * s * s)) * r.uniform(.15, .8)
    col = np.array([1, .8, .55]) if r.random() < .5 else np.array([.65, .78, 1])
    y0, y1, x0, x1 = max(0, y - 20), min(H, y + 21), max(0, x - 20), min(W, x + 21)
    c[y0:y1, x0:x1] += blob[(y0 - y + 20):(y1 - y + 20), (x0 - x + 20):(x1 - x + 20), None] * col
c += stars(H, W, 1600, 21, big=5)
save(c, 'img/deepfield.jpg')
print('ok')
