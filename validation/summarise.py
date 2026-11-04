"""Summarise validation_raw.json into VALIDATION.md."""
import json, collections
R = json.load(open('/home/user/tv/app/validation/validation_raw.json'))
for r in R:  # one catalogue signal per period: eclipsing binaries often appear several times at the same period
    keep = []
    for m in r.get('match', []):
        if not any(abs(m['cat_period'] - k['cat_period']) / k['cat_period'] < 0.01 and str(m['cat_label']).strip() == str(k['cat_label']).strip() for k in keep):
            keep.append(m)
    if 'match' in r: r['match'] = keep
POS = {'PC', 'CP', 'KP', 'PC ', 'APC'}
def is_pos(l): return str(l).strip() in ('PC', 'CP', 'KP')
L = ['# TransitVet tool: test on held-out stars', '',
     'Stars picked at random from the TEST split (never used for training, fine-tuning or calibration): '
     '15 Kepler stars with a planet candidate (PC), 15 Kepler stars with a false alarm (AFP), 10 TESS planet stars (CP/KP/PC), 10 TESS false-positive stars (FP). '
     'For each star the tool downloads the light curve itself, runs its own dip search, and scores what it finds. '
     'Catalogue entries at the same period (within 1%) on the same star are counted once. A catalogue signal counts as found if one of the tool\'s periods matches it within 1% (or at half or double the period).', '']
for mis in ['kepler', 'tess']:
    rs = [r for r in R if r['mission'] == mis]
    err = [r for r in rs if 'error' in r]
    tab = collections.defaultdict(collections.Counter)
    for r in rs:
        for m in r.get('match', []):
            lab = str(m['cat_label']).strip()
            tab[lab]['signals'] += 1
            tab[lab][m['verdict'] or 'not found'] += 1
    L += [f'## {mis.upper()} ({len(rs)} stars, {len(err)} failed to load)', '',
          '| catalogue label (PC/CP/KP = planet; AFP, NTP, FP, FA = not a planet) | signals | not found by search | candidate | uncertain | false alarm |', '|---|---|---|---|---|---|']
    for lab in sorted(tab, key=lambda x: (not is_pos(x), x)):
        t = tab[lab]; L.append(f"| {lab} | {t['signals']} | {t['not found']} | {t['candidate']} | {t['uncertain']} | {t['false alarm']} |")
    secs = [r['seconds'] for r in rs] or [0]
    L += ['', f'Time per star (4 stars at once, 1 core each): median {sorted(secs)[len(secs)//2]/60:.1f} min, max {max(secs)/60:.1f} min.', '']
    for r in err: L.append(f"- {r['target']}: failed: {r['error']}")
    L.append('')
L += ['## Per star', '', '| star | label | catalogue signals (label, period d, tool verdict) | other signals the tool found |', '|---|---|---|---|']
for r in R:
    if 'error' in r: L.append(f"| {r['target']} | {r['label']} | error | |"); continue
    used = {m['found_index'] for m in r['match']}
    cat = '; '.join(f"{m['cat_label'].strip()} {m['cat_period']:.3f}: {m['verdict'] or 'not found'}" for m in r['match'])
    oth = '; '.join(f"{f['period']:.3f} d {f['verdict']} ({f['p_main']:.2f})" for k, f in enumerate(r['found']) if k not in used)
    L.append(f"| {r['target']} | {r['label']} | {cat} | {oth} |")
open('/home/user/tv/app/validation/VALIDATION.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L[:40]))
