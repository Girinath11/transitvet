import json, base64
S = json.load(open('signals.json')); P = json.load(open('pimen.json'))
r = lambda a, d=3: [round(float(v), d) for v in a]
sig = []
for s in S:
    o = {k: s[k] for k in ('mission', 'name', 'star', 'truth', 'label', 'period', 'duration_h', 'depth', 'odd_even', 'secondary', 'n_transits', 'size_re', 'exo', 'ra', 'dec')}
    if isinstance(o['exo'], list): o['exo'] = o['exo'][0] if o['exo'] else None
    o['s'] = {k: round(v, 3) for k, v in s['s'].items()}; o['g'] = r(s['g'], 3); o['l'] = r(s['l'], 3)
    sig.append(o)
pm = dict(t=r(P['allflat']['t'], 3), f=[int(v) for v in P['allflat']['f']], t0=P['t0'], period=P['period'])
img = lambda n: 'data:image/jpeg;base64,' + base64.b64encode(open(f'img/{n}.jpg', 'rb').read()).decode()
html = open('space.html').read().replace('<!--INTRO-->', open('intro_part.html').read()).replace('__SIGNALS__', json.dumps(sig, separators=(',', ':'))).replace('__PIMEN__', json.dumps(pm, separators=(',', ':')))
for k, n in (('__IMG_HERO__', 'hero'), ('__IMG_GALAXY__', 'galaxy'), ('__IMG_NEBULA__', 'nebula2'), ('__IMG_DEEP__', 'deepfield')):
    html = html.replace(k, img(n))
os_ = __import__('os'); os_.makedirs('build', exist_ok=True); OUT = 'build/transitvet-explorer.html'
open(OUT, 'w').write(html); print(len(html) // 1024, 'KB')
open(OUT.replace('transitvet-explorer', 'space-preview'), 'w').write('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>' + html + '</body></html>')

# standalone site for real hosting (GitHub Pages / Netlify / any static host)
import os
FAV = "data:image/svg+xml," + "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ccircle cx='16' cy='16' r='11' fill='%23d6e4ff'/%3E%3Ccircle cx='21' cy='13' r='5' fill='%2303050b'/%3E%3C/svg%3E"
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="description" content="TransitVet finds repeating dips in Kepler and TESS starlight and checks each one to tell likely planet candidates from eclipsing stars and glitches.">\n'
        '<meta property="og:title" content="TransitVet: find and vet planet candidates">\n'
        '<meta property="og:description" content="Search Kepler and TESS light curves for transit signals and get a plain-language verdict for each one.">\n'
        '<meta property="og:type" content="website">\n<meta name="theme-color" content="#03050b">\n'
        f'<link rel="icon" href="{FAV}">\n')
body = html.replace('<title>TransitVet</title>', '<title>TransitVet · Find and vet planet candidates</title>', 1)
pre, rest = body.split('<style>', 1)
pre = pre.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', '')
site = head + pre + '<style>' + rest.replace('<canvas id="sky"', '</head>\n<body>\n<canvas id="sky"', 1) + '\n</body>\n</html>\n'
open('../docs/index.html', 'w').write(site); print('site', len(site) // 1024, 'KB')
