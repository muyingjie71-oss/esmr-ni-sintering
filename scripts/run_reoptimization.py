"""Heating profile re-optimized at every node of a constant-production run (Extended Data Fig. 5).

    python scripts/run_constant_production.py --times standard --out runs/cp         # limits: uniform heating
    python scripts/run_reoptimization.py --constant-production runs/cp --out runs/reopt --workers 4

At every node of the 871 h grid the ten zone shares are searched again for the current Ni field. Starting
from the shares of the previous node (the optimized profile at t = 0), a pattern search moves 2 % and then
1 % of the power between two adjacent zones (at most 8 accepted moves per step size), each move raising
the CH4 conversion at the power of the previous node. A candidate is admissible when its hottest pellet
shell and hottest wall point are not above those of the uniformly heated constant-production run at the
same node and every shell lies within 483-682 C. The power is then set so that the conversion equals the
target; if that state breaks a limit, the search is repeated once at the new power. The Ni field is
advanced with the chosen shares and power to the next node.

Results in <out>: reoptimization.csv (one row per node), reoptimization.json (every search pass);
states and the list of solved candidates (evaluations.jsonl) under <out>/node##. A rerun resumes.
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
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import esmr
from esmr import histories as H
from esmr import shares
from esmr.model import DATA, INITIAL_STATE, TEMPERATURE_DATA_RANGE_C
from esmr.report import energy_kWh, profile, read_json, write_csv, write_json

TARGET_X = 49.394791933262994
STEPS = (.02, .01)                   # power moved between adjacent zones per move
MAX_MOVES = 8                        # accepted moves per step size and node
LO, HI = TEMPERATURE_DATA_RANGE_C
KEYS = ('X', 'particle_Tmax_C', 'particle_Tmin_C', 'wall_max_C', 'outlet_gas_C', 'net_H2_mol_s', 'in_range', 'mean_Ni_pct',
        'zone_Ni_pct', 'zone_T_C')


def solve_candidate(q, P, health, seed, folder, grid):
    """One frozen solve of a candidate (run in a worker process)."""
    t = time.time()
    try:
        s = H.solve_frozen(H.make_model(P, q, grid=grid), np.asarray(health, float), esmr.load_state(seed), Path(folder))
        m = s.summary
        return dict(q=list(q), P=P, X=m['X'], Tmax=m['particle_Tmax_C'], Tmin=m['particle_Tmin_C'], wall=m['wall_max_C'],
                    folder=str(folder), ok=True, sec=time.time() - t)
    except Exception as e:  # noqa: BLE001  (an unsolved candidate is not admissible)
        return dict(q=list(q), P=P, ok=False, error=repr(e)[:300], traceback=traceback.format_exc()[-1500:], sec=time.time() - t)


def limits_from(folder):
    """Hottest shell and wall of the uniform constant-production run, and both runs' powers, by node."""
    caps, powers = {}, {}
    with open(Path(folder) / 'history.csv', newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['status'] != 'OK':
                continue
            powers.setdefault(r['heating'], {})[round(float(r['t_h']), 6)] = float(r['P_W'])
            if r['heating'] == 'uniform':
                caps[int(r['node'])] = dict(t_h=float(r['t_h']), particle_Tmax_C=float(r['particle_Tmax_C']), wall_max_C=float(r['wall_max_C']))
    return caps, powers


def reoptimize(q_start, P_start, caps, out, times=esmr.TIMES_H, workers=1, grid='main', steps=STEPS, max_moves=MAX_MOVES,
               max_candidates=None):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    nx, nr, _ = esmr.GRIDS[grid]

    def rel(p):
        return os.path.relpath(p, out)

    def full(p):
        return str(out / p)
    pool = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) if workers > 1 else None
    x = shares.normalize(q_start); P_prev = float(P_start); seed = str(INITIAL_STATE); nodes = []
    try:
        for k, t in enumerate(times):
            nd = out / f'node{k:02d}'; nf = nd / 'node.json'; nd.mkdir(parents=True, exist_ok=True)
            if nf.exists():
                rec = read_json(nf)
            else:
                h = np.ones((nx, nr)) if k == 0 else np.load(out / f'node{k - 1:02d}' / 'health_next.npy')
                capT, capW = caps[k]['particle_Tmax_C'], caps[k]['wall_max_C']
                log = nd / 'evaluations.jsonl'; cache = {}; used = []
                if log.exists():
                    for line in log.read_text().splitlines():
                        try:
                            r = json.loads(line)
                        except ValueError:                     # a line cut by an interrupted run: that solve is redone
                            continue
                        used.append(r['index'])
                        if r.get('ok'):
                            r['folder'] = full(r['folder'])
                            cache[shares.key(r['q'], r['P'])] = r
                ev = nd / 'eval'
                used += [int(p.name[1:5]) for p in ev.glob('e[0-9][0-9][0-9][0-9]*')] if ev.exists() else []
                counter = [max(used, default=0)]

                def evaluate(qs, P):
                    qs = [shares.normalize(q) for q in qs]; todo = {}
                    for q in qs:
                        kk = shares.key(q, P)
                        if kk not in cache and kk not in todo:
                            todo[kk] = q
                    if todo:
                        args, index = [], []
                        for q in todo.values():
                            counter[0] += 1; index.append(counter[0])
                            args.append((list(map(float, q)), P, h.tolist(), seed, str(ev / f'e{counter[0]:04d}'), grid))
                        results = pool.map(solve_candidate, *zip(*args)) if pool else [solve_candidate(*z) for z in args]
                        for i, r in zip(index, results):
                            r['feasible'] = bool(r['ok'] and r['Tmax'] <= capT + 1e-9 and r['wall'] <= capW + 1e-9
                                                 and r['Tmin'] >= LO and r['Tmax'] <= HI)
                            cache[shares.key(r['q'], r['P'])] = r
                            line = dict(r, index=i, folder=rel(r['folder'])) if r['ok'] else dict(r, index=i)
                            with open(log, 'a') as fh:
                                fh.write(json.dumps(line) + '\n')
                    return [cache[shares.key(q, P)] for q in qs]

                def search(x0, P):
                    cur = evaluate([x0], P)[0]; xx = shares.normalize(x0); moves = 0
                    for step in steps:
                        it = 0
                        while it < max_moves:
                            cands = shares.pair_moves(xx, step)
                            if max_candidates:
                                cands = cands[:max_candidates]
                            res = evaluate(cands, P)
                            ok = [r for r in res if r['feasible']]
                            if not cur.get('feasible'):
                                best = max(ok, key=lambda r: r['X'], default=None)
                            else:
                                best = max((r for r in ok if r['X'] > cur['X'] + 1e-6), key=lambda r: r['X'], default=None)
                            if best is None:
                                break
                            xx, cur = shares.normalize(best['q']), best; it += 1; moves += 1
                    return xx, cur, moves

                passes, P_eval = [], P_prev
                for n_pass in (1, 2):
                    xq, cur, moves = search(x, P_eval)
                    r = H.power_with_fallback(list(xq), h, TARGET_X, P_eval, esmr.load_state(cur.get('folder') or seed),
                                              nd / f'root_pass{n_pass}', grid=grid)
                    if r['status'] != 'OK':
                        passes.append(dict(n_pass=n_pass, q=xq.tolist(), status=r['status'], error=r.get('error'))); break
                    s = r['state'].summary
                    excess = max(s['particle_Tmax_C'] - capT, s['wall_max_C'] - capW, LO - s['particle_Tmin_C'], s['particle_Tmax_C'] - HI)
                    passes.append(dict(n_pass=n_pass, q=xq.tolist(), P_eval=P_eval, X_at_P_eval=cur.get('X'),
                                       admissible_at_P_eval=cur.get('feasible'), moves=moves, P_W=r['P_W'], state=rel(r['folder']),
                                       root_mode=r['root_mode'], cap_excess_K=excess, summary={k_: s[k_] for k_ in KEYS}))
                    x = xq
                    if excess <= 1e-9:
                        break
                    P_eval = r['P_W']                         # second pass: search again at the power just found
                last = passes[-1]
                if 'P_W' not in last:
                    nodes.append(dict(node=k, t_h=t, status='ROOT_FAILED', passes=passes)); break
                rec = dict(node=k, t_h=t, status='OK', P_W=last['P_W'], **last['summary'], state=last['state'], q=last['q'],
                           root_mode=last['root_mode'], cap_particle_Tmax_C=capT, cap_wall_max_C=capW,
                           cap_excess_K=last['cap_excess_K'], passes=passes, evaluations=len(cache))
                write_json(nf, rec)
            nodes.append(rec); x = shares.normalize(rec['q']); P_prev = rec['P_W']; seed = full(rec['state'])
            if not rec['in_range'] or k == len(times) - 1:
                break
            hn = nd / 'health_next.npy'
            if not hn.exists():
                h1 = H.make_model(rec['P_W'], x, grid=grid).advance(esmr.load_state(full(rec['state'])), times[k + 1] - t)
                np.save(nd / 'health_next.tmp.npy', h1); os.replace(nd / 'health_next.tmp.npy', hn)
    finally:
        if pool is not None:
            pool.shutdown()
    return nodes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--constant-production', required=True, help='output folder of run_constant_production.py')
    ap.add_argument('--start-power', type=float, help='power search start at t = 0, W (default: fresh-bed power of the optimized profile)')
    ap.add_argument('--grid', default='main', choices=['main', 'coarse'])
    ap.add_argument('--workers', type=int, default=1, help='processes solving the candidates of a search step')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); t0 = time.time()
    caps, powers = limits_from(a.constant_production)
    if any(k not in caps for k in range(len(esmr.TIMES_H))):
        raise SystemExit('the uniform constant-production run must reach 871 h')
    P_start = a.start_power or read_json(DATA / 'optimized_heating.json')['P_fresh_constant_production_W']
    nodes = reoptimize(profile('optimized'), P_start, caps, out, workers=a.workers, grid=a.grid)
    rows = []
    for n in nodes:
        if n.get('status') != 'OK':
            rows.append(dict(node=n['node'], t_h=n['t_h'], status=n['status'])); continue
        t = round(n['t_h'], 6); pu = powers.get('uniform', {}).get(t); ps = powers.get('optimized', {}).get(t)
        rows.append(dict(node=n['node'], t_h=n['t_h'], status='OK', P_W=n['P_W'], X_pct=n['X'], particle_Tmax_C=n['particle_Tmax_C'],
                         cap_Tmax_C=n['cap_particle_Tmax_C'], wall_max_C=n['wall_max_C'], cap_wall_C=n['cap_wall_max_C'],
                         cap_excess_K=n['cap_excess_K'], in_range=n['in_range'], passes=len(n['passes']), evaluations=n['evaluations'],
                         P_uniform_W=pu, P_optimized_W=ps, saving_vs_uniform_pct=100 * (pu - n['P_W']) / pu if pu else None,
                         saving_vs_optimized_pct=100 * (ps - n['P_W']) / ps if ps else None,
                         **{f'q{i + 1}': v for i, v in enumerate(n['q'])}))
    write_csv(out / 'reoptimization.csv', rows)
    ok = [r for r in rows if r['status'] == 'OK']
    write_json(out / 'reoptimization.json', dict(target_X=TARGET_X, start_power_W=P_start, steps=list(STEPS), max_moves=MAX_MOVES,
                                                 energy_871h_kWh=energy_kWh(ok, esmr.TIMES_H[-1]) if ok else None, nodes=nodes,
                                                 minutes=(time.time() - t0) / 60))
    if ok:
        print(f"871 h: {ok[-1]['P_W']:.2f} W, {ok[-1]['saving_vs_optimized_pct']} % below the fixed optimized profile")
    print('results:', out / 'reoptimization.csv')


if __name__ == '__main__':
    main()
