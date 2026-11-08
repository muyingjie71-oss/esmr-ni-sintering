"""Heaters with 2, 3 or 5 independently powered sections of adjacent zones (Fig. 5c).

    python scripts/run_heater_sections.py --constant-production runs/cp --out runs/sections --workers 4

Candidate profiles:
  2 sections  zones 1-5 and 6-10, with 55, 60, ... 90 % of the power in the first section (first_0.55 ...)
  3 sections  zones 1-3, 4-7 and 8-10; 5 sections: pairs of adjacent zones. For each, the optimized shares
              summed per section and split equally inside it (optimized_sectioned), and the profiles with
              5 % (3 sections) or 4 % (5 sections) of the power moved to the adjacent section (move_S1_to_S2 ...)
  10 sections the optimized profile itself (optimized)
Each candidate ages the bed for 871 h at 3.52 kW under the thermal limits of the profile search (hottest shell
and hottest wall point not above those under uniform heating at the same node, every shell within
483-682 C), and J is computed for the feasible ones (esmr.histories.evaluate_profile). The best feasible
candidate with 2, 3 and 5 sections is then run at constant production to 871 h, the power search starting
at 0.97 x 3.52 kW. The fresh bed of each best candidate at 3.52 kW gives the outlet approach to equilibrium
of Fig. 5c; 1 section (uniform heating) and 10 sections take their constant-production powers from
run_constant_production.py (--constant-production).

Results in <out>: candidates.csv, constant_production.csv, summary.json; states under <out>/candidates and <out>/cp.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # run from a checkout without installing the package
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import esmr
from esmr import histories as H
from esmr import shares
from esmr.model import DATA
from esmr.outlet import metrics as outlet_metrics
from esmr.report import node_row, profile, read_json, write_csv, write_json

TARGET_X = 49.394791933262994
FIRST_SECTION = [.55, .60, .65, .70, .75, .80, .85, .90]
MOVE = {3: .05, 5: .04}
CP_START = .97                       # first guess of the constant-production power search, x 3.52 kW
T_REF = esmr.TIMES_H[-1]


def candidates(q_opt):
    """(sections, label, shares) of every candidate, in the order of the tables."""
    out = [(2, f'first_{s:.2f}', shares.normalize([s / 5] * 5 + [(1 - s) / 5] * 5)) for s in FIRST_SECTION]
    for n in (3, 5):
        c0 = shares.sectioned(q_opt, shares.SECTIONS[n])
        out += [(n, 'optimized_sectioned', c0)] + [(n, lab, q) for lab, q in shares.section_moves(c0, shares.SECTIONS[n], MOVE[n])]
    out.append((10, 'optimized', list(q_opt)))
    return out


def cid(n, label):
    return f'{n}sections_{label}'


def run_candidate(n, label, q, folder, grid, times=esmr.TIMES_H, utest=True):
    """Fixed-power history and J of one candidate; the result is also written to <folder>/result.json."""
    folder = Path(folder); done = folder / 'result.json'
    if done.exists():
        r = read_json(done)
        if not r.get('failed'):
            return r
    t0 = time.time()
    try:
        r, _ = H.evaluate_profile([float(v) for v in q], read_json(DATA / f'uniform_history_{grid}.json'), folder, grid=grid,
                                  times=times, utest=utest)
        r.update(failed=False)
    except Exception as exc:  # noqa: BLE001
        r = dict(q=[float(v) for v in q], failed=True, feasible=False, J=None, error=repr(exc), traceback=traceback.format_exc())
    r.update(sections=n, label=label, minutes=(time.time() - t0) / 60)
    write_json(done, r)
    return r


def run_cp(n, label, q, folder, grid, times=esmr.TIMES_H):
    """Constant-production history of the best candidate with n sections."""
    res = H.history_constant_production(q, TARGET_X, times, P_start=CP_START * esmr.BASE_POWER_W, folder=folder, grid=grid)
    rows = []
    for r in res:
        if r['status'] != 'OK':
            rows.append(dict(sections=n, label=label, node=r['node'], t_h=r['t_h'], status=r['status'], root_mode=r.get('root_mode')))
            continue
        row = node_row(r['state'], sections=n, label=label, node=r['node'], t_h=r['t_h'], status='OK')
        row.update(particle_Tmin_C=r['state'].summary['particle_Tmin_C'], root_iterations=r['root_iterations'], root_mode=r['root_mode'])
        rows.append(row)
    return rows


def reference_powers(folder):
    """Constant-production powers of uniform and optimized heating (run_constant_production.py) by node time."""
    ref = {}
    if folder is None:
        return ref
    with open(Path(folder) / 'history.csv', newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['status'] == 'OK':
                ref.setdefault(r['heating'], {})[round(float(r['t_h']), 6)] = dict(P_W=float(r['P_W']), in_range=r['in_range'] == 'True')
    return ref


def pool_map(fn, jobs, workers):
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=mp.get_context('spawn')) as pool:
            return list(pool.map(fn, *zip(*jobs)))
    return [fn(*j) for j in jobs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--constant-production', help='output folder of run_constant_production.py (1 and 10 sections)')
    ap.add_argument('--grid', default='main', choices=['main', 'coarse'])
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); t0 = time.time(); q_opt = profile('optimized')
    cands = candidates(q_opt)
    res = pool_map(run_candidate, [(n, lab, q, out / 'candidates' / cid(n, lab), a.grid) for n, lab, q in cands], a.workers)
    rows = []
    for r in res:
        v = r.get('violation') or {}
        rows.append(dict(sections=r['sections'], label=r['label'], failed=r.get('failed'), feasible=r.get('feasible'), nodes=r.get('nodes'),
                         violation_node=v.get('node_index'), violation_excess_K=v.get('excess_K'), J_pp=r.get('J'), X_test_pct=r.get('X_test'),
                         **{f'q{i + 1}': x for i, x in enumerate(r.get('q') or [])}))
    write_csv(out / 'candidates.csv', rows)

    best = {}
    for r in res:
        n = r['sections']
        if n in (2, 3, 5) and r.get('feasible') and r.get('J') is not None and (n not in best or r['J'] > best[n]['J']):
            best[n] = r
    cp_rows = pool_map(run_cp, [(n, best[n]['label'], best[n]['q'], out / 'cp' / f'{n}sections', a.grid) for n in sorted(best)], a.workers)
    ref = reference_powers(a.constant_production)
    table = []
    for part in cp_rows:
        for row in part:
            if row['status'] == 'OK':
                t = round(row['t_h'], 6)
                pu = ref.get('uniform', {}).get(t, {}).get('P_W'); ps = ref.get('optimized', {}).get(t, {}).get('P_W')
                row.update(P_uniform_W=pu, P_optimized_W=ps, saving_vs_uniform_pct=100 * (pu - row['P_W']) / pu if pu else None,
                           extra_vs_optimized_pct=100 * (row['P_W'] / ps - 1) if ps else None)
            table.append(row)
    write_csv(out / 'constant_production.csv', table)

    # Fig. 5c: one entry per number of sections
    fresh_U = H.solve_frozen(H.make_model(esmr.BASE_POWER_W, esmr.UNIFORM, grid=a.grid),
                             np.ones(esmr.GRIDS[a.grid][:2]), None, out / 'fresh_uniform')
    entries = [dict(sections=1, label='uniform', state=fresh_U, J=0., cp=ref.get('uniform'))]
    for n in (2, 3, 5):
        if n in best:
            st = esmr.load_state(out / 'candidates' / cid(n, best[n]['label']) / 'history' / 'node00')
            cp = {round(r['t_h'], 6): dict(P_W=r['P_W'], in_range=r['in_range']) for part in cp_rows for r in part
                  if r.get('sections') == n and r['status'] == 'OK'}
            entries.append(dict(sections=n, label=best[n]['label'], state=st, J=best[n]['J'], cp=cp))
    r10 = next(r for r in res if r['sections'] == 10)
    entries.append(dict(sections=10, label='optimized', state=esmr.load_state(out / 'candidates' / cid(10, 'optimized') / 'history' / 'node00'),
                        J=r10.get('J'), cp=ref.get('optimized')))
    summary = dict(target_X=TARGET_X, sections=[])
    for e in entries:
        m = outlet_metrics(e['state']); cp = e['cp'] or {}
        times = sorted(t for t in cp if t <= T_REF + 1e-6)
        item = dict(sections=e['sections'], profile=e['label'], J_pp=e['J'], X_fresh_P0_pct=m['X'], approach_fresh_P0_K=m['approach_K'],
                    feasible_candidates=sum(1 for r in res if r['sections'] == e['sections'] and r.get('feasible')),
                    candidates=sum(1 for r in res if r['sections'] == e['sections']))
        if times and abs(times[-1] - T_REF) < 1e-6:
            P0_, P1_ = cp[times[0]]['P_W'], cp[times[-1]]['P_W']
            item.update(P_start_W=P0_, P_871_W=P1_, extra_871_pct=100 * (P1_ / P0_ - 1),
                        constant_production_in_range=all(cp[t]['in_range'] for t in times))
        summary['sections'].append(item)
    summary['minutes'] = (time.time() - t0) / 60
    write_json(out / 'summary.json', summary)
    for item in summary['sections']:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in item.items()})
    print('results:', out / 'candidates.csv', out / 'constant_production.csv', out / 'summary.json')


if __name__ == '__main__':
    main()
