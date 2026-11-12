"""Search-only check of the half-period rule on the held-out batch-run stars: which signals get doubled, and
does the doubled period match a catalogue period (within 1%)?"""
import sys, json, os
sys.path.insert(0, '/home/user/tv/repo')
from transitvet import pipeline as P
truth = {d['target']: d for d in json.load(open('/home/user/tv/scale/truth.json'))}
tg = sys.argv[1]; d = truth[tg]
try:
    segs, meta = P.load_star(tg, d['mission'])
    t, f = P.flatten_for_search(segs)
    sigs, _ = P.bls_search(t, f)
except Exception as e:
    print(json.dumps(dict(target=tg, error=str(e)))); sys.exit()
out = []
for s in sigs:
    m = lambda p: [l for cp, l in zip(d['cat_periods'], d['cat_labels']) if abs(p - cp) / cp < 0.01]
    out.append(dict(period=s['period'], doubled=s.get('period_doubled', False), sigma=s.get('half_period_sigma'),
                    match=m(s['period']), match_half=m(s['period'] / 2) if s.get('period_doubled') else None))
print(json.dumps(dict(target=tg, mission=d['mission'], sigs=out)), flush=True)
