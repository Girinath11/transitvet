"""TransitVet command line: analyse many stars without the dashboard.

  python -m transitvet run "TIC 1333933" "KIC 11442793"
  python -m transitvet run --file stars.txt --out results --workers 4
  python -m transitvet test                    # run the automated tests

Batch runs are
  parallel    one star per worker process (each worker loads the models once)
  resumable   a star whose result file exists is skipped; re-run the same command after a crash
  logged      results/run.log; one JSON file per star in results/stars/; errors in results/errors.csv
  time-boxed  a star that takes longer than --timeout seconds is stopped and recorded as an error
  retried     downloads are retried with back-off (network hiccups)
Outputs: results/summary.csv (one row per signal, ranked by score), results/run_manifest.json.
"""
import argparse, csv, json, logging, os, re, signal, sys, time, traceback

import numpy as np

_SCORER = None


def _safe(name):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name.strip())


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items() if not str(k).startswith('_') and k not in ('global_astronet', 'local_astronet', 'px_best')}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 50 else None
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def with_retries(fn, *a, tries=4, base=5.0, **k):
    for i in range(tries):
        try:
            return fn(*a, **k)
        except ValueError:
            raise                      # not a network problem (e.g. no data for this star)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(base * 2 ** i)


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _init_worker():
    global _SCORER
    os.environ.setdefault('TV_BLS_WORKERS', '1')       # parallel over stars, not inside one star
    import torch
    torch.set_num_threads(1)
    from . import pipeline as P
    _SCORER = P.Scorer()
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def analyse_one(job):
    """job: dict(target, mission, out, pixel, timeout). Writes stars/<target>.json. Returns a small status dict."""
    from . import pipeline as P
    t0 = time.time()
    path = os.path.join(job['out'], 'stars', _safe(job['target']) + '.json')
    has_alarm = hasattr(signal, 'SIGALRM')
    if has_alarm:
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(int(job['timeout']))
    try:
        segs, meta = with_retries(P.load_star, job['target'], job['mission'])
        meta['_pixel'] = job['pixel']; meta['_keep_files'] = job.get('keep_files', False)
        res = P.run(segs, meta, _SCORER, progress=lambda m: None)
        tab = P.summary_table(res)
        out = dict(target=job['target'], status='ok', seconds=round(time.time() - t0, 1),
                   manifest=_jsonable(P.manifest(res, _SCORER)), table=_jsonable(tab.to_dict('records')),
                   signals=_jsonable(res['signals']))
        tmp = path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(out, fh, indent=1, default=str)
        os.replace(tmp, path)             # atomic: a crash never leaves half a file
        return dict(target=job['target'], status='ok', n_signals=len(res['signals']), seconds=out['seconds'])
    except _Timeout:
        return dict(target=job['target'], status='error', error=f"timeout after {job['timeout']} s", seconds=round(time.time() - t0, 1))
    except Exception as e:
        return dict(target=job['target'], status='error', error=f'{type(e).__name__}: {e}'[:300],
                    trace=traceback.format_exc()[-1500:], seconds=round(time.time() - t0, 1))
    finally:
        if has_alarm:
            signal.alarm(0)


def collect(out):
    """Merge all per-star files into summary.csv, ranked by the main score."""
    import pandas as pd
    rows = []
    d = os.path.join(out, 'stars')
    for f in sorted(os.listdir(d)):
        if f.endswith('.json'):
            try:
                rows += json.load(open(os.path.join(d, f)))['table']
            except Exception:
                pass
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values('p_main', ascending=False)
    df.to_csv(os.path.join(out, 'summary.csv'), index=False)
    return df


def cmd_run(a):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    targets = list(a.targets)
    if a.file:
        targets += [l.split('#')[0].strip() for l in open(a.file) if l.split('#')[0].strip()]
    targets = list(dict.fromkeys(targets))
    os.makedirs(os.path.join(a.out, 'stars'), exist_ok=True)
    log = logging.getLogger('transitvet'); log.setLevel(logging.INFO); log.propagate = False
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    for h in (logging.FileHandler(os.path.join(a.out, 'run.log')), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); log.addHandler(h)
    mission = None if a.mission == 'auto' else a.mission
    todo = [t for t in targets if a.redo or not os.path.exists(os.path.join(a.out, 'stars', _safe(t) + '.json'))]
    log.info('%d stars, %d already done, %d to analyse with %d workers', len(targets), len(targets) - len(todo), len(todo), a.workers)
    from . import pipeline as P
    if a.alpha:
        os.environ['TV_CONFORMAL_ALPHA'] = str(a.alpha)
    jobs = [dict(target=t, mission=mission, out=a.out, pixel=not a.no_pixel, timeout=a.timeout, keep_files=a.keep_downloads) for t in todo]
    err_path = os.path.join(a.out, 'errors.csv')
    t_start = time.time(); n_ok = n_err = 0
    with ProcessPoolExecutor(a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(analyse_one, j): j for j in jobs}
        for k, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            if r['status'] == 'ok':
                n_ok += 1
                log.info('[%d/%d] %s: %d signal(s), %.0f s', k, len(jobs), r['target'], r['n_signals'], r['seconds'])
            else:
                n_err += 1
                log.warning('[%d/%d] %s: FAILED %s', k, len(jobs), r['target'], r['error'])
                new = not os.path.exists(err_path)
                with open(err_path, 'a', newline='') as fh:
                    w = csv.writer(fh)
                    if new:
                        w.writerow(['time_utc', 'target', 'error'])
                    w.writerow([time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), r['target'], r['error']])
    df = collect(a.out)
    wall = time.time() - t_start
    man = dict(tool='TransitVet batch', finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), n_targets=len(targets),
               n_analysed_now=len(jobs), n_ok_now=n_ok, n_failed_now=n_err, wall_seconds=round(wall, 1), workers=a.workers,
               pixel_check=not a.no_pixel, conformal_alpha=float(os.environ.get('TV_CONFORMAL_ALPHA', 0.05)),
               n_signals_total=int(len(df)), models=json.load(open(os.path.join(os.path.dirname(__file__), 'models', 'manifest.json'))))
    json.dump(man, open(os.path.join(a.out, 'run_manifest.json'), 'w'), indent=1)
    log.info('done: %d ok, %d failed in %.0f s; %d signals in %s', n_ok, n_err, wall, len(df), os.path.join(a.out, 'summary.csv'))
    return 0 if n_err == 0 else 1


def cmd_test(a):
    import subprocess
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = [sys.executable, '-m', 'pytest', '-q'] + (['-m', 'not network'] if not a.network else [])
    return subprocess.call(args, cwd=here)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='transitvet', description='TransitVet: find and vet planet candidates in Kepler/TESS light curves.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='analyse stars (batch)')
    r.add_argument('targets', nargs='*', help='star IDs like "TIC 1333933" or "KIC 11442793"')
    r.add_argument('--file', help='text file with one star ID per line (# starts a comment)')
    r.add_argument('--mission', default='auto', choices=['auto', 'kepler', 'tess'])
    r.add_argument('--out', default='results')
    r.add_argument('--workers', type=int, default=max(1, min(4, os.cpu_count() or 1)))
    r.add_argument('--timeout', type=int, default=1800, help='seconds per star (default 1800)')
    r.add_argument('--no-pixel', action='store_true', help='skip the pixel-level check (no camera-image downloads)')
    r.add_argument('--keep-downloads', action='store_true', help='keep the camera images on disk (TESS: ~100 MB per sector)')
    r.add_argument('--alpha', type=float, choices=[0.05, 0.1, 0.2], help='trust-check level (0.05 = 95%% guarantee)')
    r.add_argument('--redo', action='store_true', help='analyse again stars that already have a result')
    t = sub.add_parser('test', help='run the automated tests')
    t.add_argument('--network', action='store_true', help='also run tests that download from MAST')
    a = ap.parse_args(argv)
    return {'run': cmd_run, 'test': cmd_test}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
