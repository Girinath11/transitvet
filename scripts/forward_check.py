"""Forward test: did the open TESS signals we scored on 17 Nov 2026 later get resolved, and did we call them right?
Reads validation/unlabelled_tess_scores.csv (our frozen calls), downloads today's TOI table from the NASA Exoplanet
Archive, and writes docs/forward.json for the website. Our calls are never changed after the freeze date."""
import csv, io, json, datetime, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OURS = os.path.join(HERE, '..', 'validation', 'unlabelled_tess_scores.csv')
OUT = os.path.join(HERE, '..', 'docs', 'forward.json')
URL = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+toi,tfopwg_disp+from+toi&format=csv'
FROZEN = '2026-11-17'


def main():
    raw = urllib.request.urlopen(URL, timeout=120).read().decode()
    now = {r['toi']: r['tfopwg_disp'] for r in csv.DictReader(io.StringIO(raw))}
    if len(now) < 5000:
        sys.exit(f'TOI table looks incomplete ({len(now)} rows); not updating')
    rows = []
    for r in csv.DictReader(open(OURS)):
        d = now.get(f"{float(r['toi']):.2f}")
        if d in ('CP', 'KP', 'FP', 'FA'):
            rows.append(dict(toi=f"{float(r['toi']):.2f}", tic=int(r['tic']), now=d, planet=d in ('CP', 'KP'), score=round(float(r['score']), 3),
                             verdict=r['verdict'], check=r['check_95'], exominer=None if r['exominer'] in ('', 'nan') else round(float(r['exominer']), 3)))
    def tally(sel):
        return dict(n=len(sel), planets=sum(x['planet'] for x in sel), false_alarms=sum(not x['planet'] for x in sel))
    summary = {'score >= 0.5': tally([x for x in rows if x['score'] >= 0.5]), 'score < 0.5': tally([x for x in rows if x['score'] < 0.5]),
               'strong (candidate + keep)': tally([x for x in rows if x['verdict'] == 'candidate' and x['check'] == 'keep']),
               'drop': tally([x for x in rows if x['check'] == 'drop'])}
    json.dump(dict(frozen=FROZEN, checked=datetime.date.today().isoformat(), scored=4408, resolved=len(rows), summary=summary,
                   rows=sorted(rows, key=lambda x: -x['score'])), open(OUT, 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
