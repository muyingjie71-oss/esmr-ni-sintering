"""Heating profile for the highest conversion of the fresh bed (Extended Data Fig. 3).

    python scripts/run_fresh_optimization.py --out runs/fresh --workers 4

The total power is 3.52 kW and the ten zone shares are non-negative and sum to one. The objective is the CH4
conversion of the fresh bed. Limits: the hottest pellet shell and the hottest wall point not above those of the
fresh bed under uniform heating (650.27 and 701.86 C), and every shell within 483-682 C. Each round moves 4 %,
2 % or 1 % of the total power from one zone to an adjacent zone and takes the candidate that raises the
conversion most (by more than 1e-6 % points). When no candidate improves, the next smaller step is used; the
search ends when the 1 % step finds none. Starts: the optimized profile of Fig. 3a, uniform heating, and a random
profile (+-30 % around uniform, seed 20261103) scaled toward uniform by 1, 0.5 and 0.25 until its fresh bed is
within the limits. A start without such a profile is recorded and not searched, and a start that reaches the
profile of another stops. The final profiles that differ from each other (L1 distance above 0.02) and the
optimized profile of Fig. 3a are then aged for 871 h at 3.52 kW with the limits of the profile search, and J is
computed for those that stay within them (esmr.histories.evaluate_profile).

Results in <out>: summary.json (starts, accepted steps, final profiles, ageing runs), steps.csv (conversion
after every accepted step), evaluations.jsonl (every solve); states under <out>/states and <out>/ageing.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # run from a checkout without installing the package
    sys.path.insert(0, str(ROOT))

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import esmr
from esmr import histories as H
from esmr import shares
from esmr.model import DATA, TEMPERATURE_DATA_RANGE_C
from esmr.report import profile, read_json, write_csv, write_json

STEPS = (.04, .02, .01)              # power moved between adjacent zones per move
MAX_MOVES = 40                       # accepted moves per step size (a safety stop, not reached)
RANDOM_SEED = 20261103
RANDOM_SCALES = (1., .5, .25)        # the random start scaled toward uniform heating until its fresh bed is admissible
DIGITS = 7                           # profiles are compared after rounding the shares to 7 decimals
LO, HI = TEMPERATURE_DATA_RANGE_C


def starts(q_opt):
    """Start name -> candidate start profiles; the first admissible candidate is used."""
    rng = np.random.default_rng(RANDOM_SEED)
    u = np.full(10, .1); r0 = shares.normalize(u * (1 + rng.uniform(-.3, .3, 10)))
    return {'optimized': [shares.normalize(q_opt)], 'uniform': [shares.normalize(u)],
            'random': [shares.normalize(u + lam * (r0 - u)) for lam in RANDOM_SCALES]}


def solve_fresh(q, folder, grid='main'):
    """Fresh bed at 3.52 kW under profile q (run in a worker process)."""
    t = time.time()
    try:
        s = H.solve_frozen(H.make_model(esmr.BASE_POWER_W, q, grid=grid), np.ones(esmr.GRIDS[grid][:2]), None, Path(folder)).summary
        return dict(q=list(map(float, q)), X=s['X'], Tmax=s['particle_Tmax_C'], Tmin=s['particle_Tmin_C'], wall=s['wall_max_C'],
                    ok=True, sec=time.time() - t, folder=str(folder))
    except Exception as e:  # noqa: BLE001  (an unsolved profile is not admissible)
        return dict(q=list(map(float, q)), X=None, ok=False, error=repr(e)[:300], traceback=traceback.format_exc()[-1200:],
                    sec=time.time() - t)


def search(solve_many, q_opt, T_cap, wall_cap, steps=STEPS, max_moves=MAX_MOVES, log=print, cache=None):
    """Pattern search of all starts in lockstep rounds. solve_many(list of profiles) -> list of result dicts."""
    cache = {} if cache is None else cache

    def admissible(r):
        return bool(r['ok'] and r['Tmax'] <= T_cap + 1e-6 and r['wall'] <= wall_cap + 1e-6 and r['Tmin'] >= LO and r['Tmax'] <= HI)

    def evaluate(qs):
        todo = {}
        for q in qs:
            k = shares.key(q, digits=DIGITS)
            if k not in cache and k not in todo:
                todo[k] = q
        if todo:
            for r in solve_many(list(todo.values())):
                r['admissible'] = admissible(r)
                cache[shares.key(r['q'], digits=DIGITS)] = r
                if not r['ok']:
                    log(f'  solve failed for {[round(v, 4) for v in r["q"]]}: {r.get("error")}')
        out = []
        for q in qs:
            r = dict(cache[shares.key(q, digits=DIGITS)]); r['admissible'] = admissible(r) if r.get('ok') else False; out.append(r)
        return out

    S = starts(q_opt); st = {}
    first = evaluate([x for xs in S.values() for x in xs]); i = 0
    for name, xs in S.items():
        rr = first[i:i + len(xs)]; i += len(xs)
        k = next((j for j, r in enumerate(rr) if r['admissible']), 0)
        st[name] = s = dict(name=name, x=xs[k], step_i=0, moves_at_step=0, moves=0, done=False, merged_into=None, path=[],
                            q_start=xs[k].tolist(), cur=rr[k], start=rr[k], start_scale=RANDOM_SCALES[k] if len(xs) > 1 else None,
                            start_candidates=[_slim(c) for c in rr], admissible_start=bool(rr[k]['admissible']))
        log(f'start {name}: X = {rr[k].get("X")}, admissible {rr[k]["admissible"]}')
        if not rr[k]['admissible']:
            s['done'] = True
    while any(not s['done'] for s in st.values()):
        act = [s for s in st.values() if not s['done']]
        cand = {s['name']: shares.pair_moves(s['x'], steps[s['step_i']]) for s in act}
        res = evaluate([q for c in cand.values() for q in c]); i = 0
        for s in act:
            rr = res[i:i + len(cand[s['name']])]; i += len(cand[s['name']])
            best = max((r for r in rr if r['admissible'] and r['X'] > s['cur']['X'] + 1e-6), key=lambda r: r['X'], default=None)
            if best is not None and s['moves_at_step'] < max_moves:
                s['x'], s['cur'] = shares.normalize(best['q']), best; s['moves'] += 1; s['moves_at_step'] += 1
                s['path'].append(dict(step=steps[s['step_i']], X=best['X'], Tmax=best['Tmax'], Tmin=best['Tmin'], wall=best['wall'],
                                      q=list(map(float, s['x']))))
                log(f'{s["name"]:9s} step {steps[s["step_i"]]:.2f}  X = {best["X"]:.5f}')
            else:
                s['step_i'] += 1; s['moves_at_step'] = 0
                if s['step_i'] >= len(steps):
                    s['done'] = True
                    log(f'{s["name"]:9s} ends: X = {s["cur"]["X"]:.5f} after {s["moves"]} moves')
        seen = {}
        for s in st.values():                                  # two starts on the same profile: the later one stops
            k = shares.key(s['x'], digits=DIGITS)
            if k in seen and not s['done']:
                s['done'] = True; s['merged_into'] = seen[k]; log(f'{s["name"]} reached the profile of {seen[k]}')
            seen.setdefault(k, s['name'])
    return st, cache


def ageing_runs(st, q_opt):
    """(label, profile) of the ageing runs: the optimized profile of Fig. 3a and every distinct final profile."""
    final = sorted([s for s in st.values() if s['admissible_start'] and not s['merged_into']], key=lambda s: -s['cur']['X'])
    distinct = []
    for s in final:
        if all(np.abs(s['x'] - d['x']).sum() > .02 for d in distinct):
            distinct.append(s)
    qn = shares.normalize(q_opt); runs = [('optimized_fig3a', list(map(float, qn)))]
    for s in st.values():
        s['ageing'] = None
    for s in distinct:
        if np.abs(s['x'] - qn).sum() <= .02:                  # within one 1 % move of the optimized profile: use its run
            s['ageing'] = 'optimized_fig3a'; continue
        s['ageing'] = f'from_{s["name"]}'
        runs.append((s['ageing'], list(map(float, s['x']))))
    return runs


def age(label, q, folder, grid='main'):
    folder = Path(folder); done = folder / 'result.json'
    if done.exists():
        return read_json(done)
    try:
        r, _ = H.evaluate_profile(q, read_json(DATA / f'uniform_history_{grid}.json'), folder, grid=grid)
        r.update(failed=False)
    except Exception as exc:  # noqa: BLE001
        r = dict(q=list(q), failed=True, feasible=False, J=None, error=repr(exc)[:300])
    r.update(label=label)
    write_json(done, r)
    return r


def _slim(r):
    return {k: r.get(k) for k in ('X', 'Tmax', 'Tmin', 'wall', 'ok', 'admissible', 'error')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grid', default='main', choices=['main', 'coarse'])
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    base = read_json(DATA / f'uniform_history_{a.grid}.json')
    T_cap, wall_cap = base['Tmax_C'][0] + base['margin_K'], base['wall_max_C'][0] + base['margin_K']
    log_path = out / 'evaluations.jsonl'; cache = {}; used = []
    if log_path.exists():                                     # resume: reuse every solved profile
        for line in log_path.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            used.append(r['index'])
            if r.get('ok'):
                cache[shares.key(r['q'], digits=DIGITS)] = r
    folders = out / 'states'
    used += [int(p.name[1:5]) for p in folders.glob('e[0-9][0-9][0-9][0-9]*')] if folders.exists() else []
    counter = [max(used, default=0)]
    pool = ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) if a.workers > 1 else None

    def solve_many(qs):
        idx = []
        for _ in qs:
            counter[0] += 1; idx.append(counter[0])
        dirs = [str(folders / f'e{i:04d}') for i in idx]
        grids = [a.grid] * len(qs)
        res = list(pool.map(solve_fresh, qs, dirs, grids)) if pool else [solve_fresh(*z) for z in zip(qs, dirs, grids)]
        with open(log_path, 'a') as fh:
            for i, r in zip(idx, res):
                fh.write(json.dumps(dict(r, index=i, folder=os.path.relpath(r['folder'], out)) if r['ok'] else dict(r, index=i)) + '\n')
        return res

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    try:
        q_opt = profile('optimized')
        st, cache = search(solve_many, q_opt, T_cap, wall_cap, log=log, cache=cache)
        runs = ageing_runs(st, q_opt)
        jobs = [(lab, q, out / 'ageing' / lab, a.grid) for lab, q in runs]
        aged = list(pool.map(age, *zip(*jobs))) if pool and len(jobs) > 1 else [age(*j) for j in jobs]
    finally:
        if pool is not None:
            pool.shutdown()
    qn = shares.normalize(q_opt)
    summary = dict(grid=a.grid, limits=dict(particle_Tmax_C=T_cap, wall_max_C=wall_cap, shells_C=[LO, HI]), steps=list(STEPS),
                   states_solved=len(cache), minutes=(time.time() - t0) / 60,
                   starts={n: dict(q_start=s['q_start'], start=_slim(s['start']), start_scale=s['start_scale'],
                                   start_candidates=s['start_candidates'], admissible_start=s['admissible_start'],
                                   q_final=list(map(float, s['x'])), final=_slim(s['cur']), moves=s['moves'],
                                   merged_into=s['merged_into'], L1_to_optimized=float(np.abs(s['x'] - qn).sum()),
                                   ageing=s['ageing'], path=s['path']) for n, s in st.items()},
                   ageing=aged)
    write_json(out / 'summary.json', summary)
    rows = []
    for n, s in summary['starts'].items():
        rows.append(dict(start=n, step=0, move_size=None, X=s['start']['X'], **{f'q{i + 1}': v for i, v in enumerate(s['q_start'])}))
        for k, p in enumerate(s['path'], start=1):
            rows.append(dict(start=n, step=k, move_size=p['step'], X=p['X'], **{f'q{i + 1}': v for i, v in enumerate(p['q'])}))
    write_csv(out / 'steps.csv', rows)
    for n, s in summary['starts'].items():
        print(f"{n}: X {s['start']['X']} -> {s['final']['X']} after {s['moves']} moves")
    for r in aged:
        print(f"ageing {r['label']}: J = {r.get('J')}, limit exceeded at node {(r.get('violation') or {}).get('node_index')}")
    print('results:', out / 'summary.json', out / 'steps.csv')


if __name__ == '__main__':
    main()
