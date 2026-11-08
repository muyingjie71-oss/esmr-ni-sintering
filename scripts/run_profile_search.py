"""Search for the heating profile that maximizes J (Fig. 3a, Supplementary Note 2).

J(q) = X_U(s_q) - X_U(s_U): both Ni fields aged 871 h at fixed power, then solved under uniform heating.
At every node of the history of q, the hottest pellet shell and the hottest wall point may not exceed
those of the uniformly heated history (plus a screening margin of 2 K on the coarse grid), and every
shell must stay within 483-682 C. Candidates are screened on the coarse grid by a surrogate-assisted
trust-region search started from earlier coarse histories (esmr/data/search_start_coarse.json), and
the best ones are evaluated again on the main grid.

  python scripts/run_profile_search.py search  --out runs/search --hours 6 --workers 2
  python scripts/run_profile_search.py confirm --out runs/search --top 6 --workers 2

Every round is saved (archive.json, search_state.json); rerunning resumes. A file STOP in --out ends the
search after the current round.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # run from a checkout without installing the package
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
import multiprocessing as mp
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from esmr.model import DATA
from esmr.report import read_json, write_json
from esmr.search import Archive, propose, select_for_main


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def evaluate(job):
    """One candidate: fixed-power history with the thermal limits, then the uniform test (worker process)."""
    from esmr.histories import evaluate_profile
    folder = Path(job['folder']); t0 = time.time()
    if (folder / 'RESULT.json').exists():
        return read_json(folder / 'RESULT.json')
    if folder.exists():
        shutil.rmtree(folder)
    try:
        res, _ = evaluate_profile(job['q'], job['baseline'], folder, grid=job['grid'])
        res.update(id=job['id'], failed=False)
    except Exception as exc:  # noqa: BLE001  (recorded as a failed evaluation, never as infeasible physics)
        res = dict(id=job['id'], q=job['q'], failed=True, feasible=False, J=None, nodes=0, dp=[], dw=[], failure=repr(exc))
    res['elapsed_s'] = time.time() - t0
    write_json(folder / 'RESULT.json', res)
    if not res.get('feasible') and not job.get('keep_states'):
        for sub in ('history', 'utest'):
            shutil.rmtree(folder / sub, ignore_errors=True)
    return res


def run_batch(pool, out, round_id, cands, grid, baseline, keep):
    jobs = [dict(id=f'r{round_id:03d}_c{i:02d}', q=list(map(float, q)), grid=grid, baseline=baseline,
                 folder=str(out / 'jobs' / f'r{round_id:03d}_c{i:02d}'), keep_states=keep) for i, q in enumerate(cands)]
    return list(pool.map(evaluate, jobs))


def finalize(out, arch, st):
    top = arch.best(10)
    write_json(out / 'TOP_CANDIDATES.json', [dict(id=r['id'], J_pp=r['J'], q=r['q'], slack_K=r['s']) for r in top])
    write_json(out / 'MAIN_STAGE_CANDIDATES.json', [dict(id=r['id'], J_pp=r['J'], q=r['q']) for r in select_for_main(arch, 6, 0.06)])
    new = [r for r in arch.rec if not str(r['id']).startswith('start_')]
    with open(out / 'SEARCH_HISTORY.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); keys = ['id', 'q', 'J', 'X_test', 'feasible', 'nodes', 's', 'elapsed_s', 'failed']; w.writerow(keys)
        for r in new:
            w.writerow([json.dumps(r.get(k)) if isinstance(r.get(k), (list, dict)) else r.get(k) for k in keys])
    log(f"rounds {st['round']}, new evaluations {len(new)}, best J {top[0]['J']:.5f} pp ({top[0]['id']})" if top else 'no feasible profile')


def search(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    baseline = read_json(DATA / f'uniform_history_{a.grid}.json')
    arch = Archive(baseline['margin_K']); st_path = out / 'search_state.json'
    if (out / 'archive.json').exists() and st_path.exists():
        for r in read_json(out / 'archive.json'):
            arch.add(r)
        st = read_json(st_path); log(f"resume: {len(arch.rec)} records, round {st['round']}, radius {st['radius']:.4f}")
    else:
        for r in read_json(DATA / f'search_start_{a.grid}.json'):
            arch.add(r)
        st = dict(round=0, radius=a.radius0, center_rank=0, stagn=0, elapsed_prev=0.0, best_J=None)
        log(f'start: {len(arch.rec)} earlier histories, {len(arch.feasible())} feasible')
    rng = np.random.default_rng(a.seed + st['round'])
    best_J = arch.best(1)[0]['J']; t_start = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn')) as pool:
        while True:
            spent = st['elapsed_prev'] + time.time() - t_start
            if spent > a.hours * 3600 or st['round'] >= a.max_rounds or (out / 'STOP').exists():
                log('stopping: time, rounds or STOP'); break
            r_id = st['round'] + 1
            cands, info = propose(arch, rng, a.batch, st['radius'], st['center_rank'])
            log(f"round {r_id}: {len(cands)} candidates, radius {st['radius']:.3f}, centre {info['center_id']} J={info['center_J']:.5f}")
            results = run_batch(pool, out, r_id, cands, a.grid, baseline, a.keep_states)
            nfail = sum(bool(r.get('failed')) for r in results)
            for r in results:
                arch.add(r)
            new_best = arch.best(1)[0]['J']
            log(f"round {r_id}: feasible {sum(bool(r.get('feasible')) for r in results)}/{len(results)}, failed {nfail}, best {new_best:.5f}")
            if nfail >= max(2, len(results) // 2):
                write_json(out / 'archive.json', [{k: v for k, v in r.items() if k != 's'} for r in arch.rec])
                raise RuntimeError('most evaluations of this round failed: ' + ' | '.join(r.get('failure', '') for r in results if r.get('failed'))[:400])
            if new_best > best_J + 0.0015:
                st['radius'] = min(st['radius'] * 1.4, 0.30); st['stagn'] = 0; best_J = new_best; st['center_rank'] = 0
            else:
                st['radius'] = max(st['radius'] * 0.65, 0.006); st['stagn'] += 1
                if st['radius'] <= 0.0061 and st['stagn'] >= 3:          # local convergence: restart from the next distinct centre
                    st['center_rank'] = min(st['center_rank'] + 1, 4); st['radius'] = a.radius0; st['stagn'] = 0
            st['round'] = r_id; st['best_J'] = new_best
            st['elapsed_prev'] = st['elapsed_prev'] + time.time() - t_start; t_start = time.time()
            write_json(out / 'archive.json', [{k: v for k, v in r.items() if k != 's'} for r in arch.rec])
            write_json(st_path, st); finalize(out, arch, st)
    finalize(out, arch, st)


def confirm(a):
    out = Path(a.out); cands = read_json(out / 'MAIN_STAGE_CANDIDATES.json')[:a.top]
    baseline = read_json(DATA / 'uniform_history_main.json')
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn')) as pool:
        res = run_batch(pool, out / 'main', 900, [c['q'] for c in cands], 'main', baseline, True)
    rows = [dict(screen_id=c['id'], J_screen=c['J_pp'], q=c['q'], J_main=r.get('J'), feasible_main=bool(r.get('feasible')),
                 max_dTmax_K=max(r['dp']) if r.get('dp') else None, max_dWall_K=max(r['dw']) if r.get('dw') else None)
            for c, r in zip(cands, res)]
    ok = [r for r in rows if r['feasible_main']]
    final = dict(status='CONFIRMED', **max(ok, key=lambda r: r['J_main'])) if ok else dict(status='NO_FEASIBLE_CANDIDATE')
    write_json(out / 'MAIN_RESULTS.json', rows); write_json(out / 'FINAL.json', final)
    for r in rows:
        log(f"{r['screen_id']}: J coarse {r['J_screen']:.4f} -> J main {r['J_main']} feasible {r['feasible_main']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)
    s = sub.add_parser('search'); s.add_argument('--out', required=True); s.add_argument('--grid', default='coarse', choices=['coarse', 'main'])
    s.add_argument('--hours', type=float, default=3.0); s.add_argument('--workers', type=int, default=2)
    s.add_argument('--batch', type=int, default=12); s.add_argument('--max-rounds', type=int, default=200)
    s.add_argument('--radius0', type=float, default=0.08); s.add_argument('--seed', type=int, default=20261101)
    s.add_argument('--keep-states', action='store_true')
    c = sub.add_parser('confirm'); c.add_argument('--out', required=True); c.add_argument('--top', type=int, default=6)
    c.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    (search if a.stage == 'search' else confirm)(a)


if __name__ == '__main__':
    main()
