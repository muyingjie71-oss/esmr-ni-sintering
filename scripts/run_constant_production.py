"""Constant production with uniform and optimized heating, up to 20,628 h (Fig. 4).

    python scripts/run_constant_production.py --out runs/cp --workers 2
    python scripts/run_constant_production.py --out runs/cp --times standard     # to 871 h only

Both heatings hold the CH4 conversion of the fresh, uniformly heated bed at 3.52 kW (49.39 %). At every node
the power is found for the current Ni field and held over the next step, during which the Ni field ages
(esmr.histories.history_constant_production). The nodes follow the 871 h grid and continue beyond it with
a factor of 4/3 between nodes up to 20,628 h. A history stops at the first node with a pellet shell outside
483-682 C; that node is kept.

Results in <out>:
  history.csv    one row per node and heating; 'state' is the folder of the accepted state
  summary.json   run length: the time at which a shell leaves 483-682 C, interpolated in log t between the
                 last node inside and the first node outside, and the time at which the power reaches
                 1.05 x 3.52 kW; electricity over the time covered by both runs and over 871 h; the
                 hottest shell and the hottest wall point of optimized minus uniform heating at equal times
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # run from a checkout without installing the package
    sys.path.insert(0, str(ROOT))

import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import esmr
from esmr import histories as H
from esmr.report import energy_kWh, node_row, profile, read_json, run_length, write_csv, write_json
from esmr.model import DATA

TARGET_X = 49.394791933262994        # fresh bed, uniform heating, 3.52 kW
POWER_LIMIT = 1.05                    # run-length criterion (ii): power reaches 1.05 x 3.52 kW
T_REF = esmr.TIMES_H[-1]


def start_powers():
    """First guess of the power search at t = 0: 3.52 kW for uniform heating; for the optimized profile the
    power at which its fresh bed reaches the target conversion."""
    return dict(uniform=esmr.BASE_POWER_W, optimized=read_json(DATA / 'optimized_heating.json')['P_fresh_constant_production_W'])


def run_heating(name, q, P_start, times, folder, grid='main'):
    """Constant-production history of one heating; returns the table rows (states stay on disk)."""
    folder = Path(folder)
    res = H.history_constant_production(q, TARGET_X, times, P_start=P_start, folder=folder / name, grid=grid)
    rows = []
    for r in res:
        if r['status'] != 'OK':
            rows.append(dict(heating=name, node=r['node'], t_h=r['t_h'], status=r['status'], root_mode=r.get('root_mode'),
                             error=r.get('error')))
            continue
        s = r['state']
        row = node_row(s, heating=name, node=r['node'], t_h=r['t_h'], status='OK')
        row.update(P_over_P0=s.summary['P_W'] / esmr.BASE_POWER_W, particle_Tmin_C=s.summary['particle_Tmin_C'],
                   root_iterations=r['root_iterations'], root_mode=r['root_mode'],
                   state=os.path.relpath(r['state_folder'], folder))
        rows.append(row)
    return rows


def compare(rows, P_limit_W):
    by = {}
    for r in rows:
        by.setdefault(r['heating'], []).append(r)
    out = {name: run_length(nodes, P_limit_W) for name, nodes in by.items()}
    if {'uniform', 'optimized'} <= set(by):
        U = [n for n in by['uniform'] if n['status'] == 'OK']; S = [n for n in by['optimized'] if n['status'] == 'OK']
        end = min(U[-1]['t_h'], S[-1]['t_h'])
        eU, eS = energy_kWh(U, end), energy_kWh(S, end)
        rU, rS = energy_kWh(U, T_REF), energy_kWh(S, T_REF)
        at = {round(n['t_h'], 6): n for n in U}
        out.update(common_end_h=end,
                   energy_common_kWh=dict(uniform=eU, optimized=eS, saving_pct=100 * (eU - eS) / eU if eU else None),
                   energy_871h_kWh=dict(uniform=rU, optimized=rS, saving_pct=100 * (rU - rS) / rU if rU else None),
                   optimized_minus_uniform=[dict(t_h=n['t_h'],
                                                 particle_Tmax_K=n['particle_Tmax_C'] - at[round(n['t_h'], 6)]['particle_Tmax_C'],
                                                 wall_max_K=n['wall_max_C'] - at[round(n['t_h'], 6)]['wall_max_C'])
                                            for n in S if round(n['t_h'], 6) in at])
        if out['uniform'].get('t_exit_h') and out['optimized'].get('t_exit_h'):
            out['run_length_gain_pct'] = 100 * (out['optimized']['t_exit_h'] / out['uniform']['t_exit_h'] - 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--heatings', default='uniform,optimized', help='comma-separated: uniform, optimized')
    ap.add_argument('--times', default='long', choices=['long', 'standard'], help='long: to 20,628 h; standard: to 871 h')
    ap.add_argument('--start-power', action='append', default=[], metavar='HEATING=W',
                    help='first guess of the power search at t = 0 (default: see start_powers)')
    ap.add_argument('--grid', default='main', choices=['main', 'coarse'])
    ap.add_argument('--workers', type=int, default=1, help='processes; 2 runs both heatings side by side')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); t0 = time.time()
    times = H.LONG_TIMES_H if a.times == 'long' else esmr.TIMES_H
    P_start = start_powers()
    for item in a.start_power:
        k, v = item.split('='); P_start[k] = float(v)
    names = [n.strip() for n in a.heatings.split(',') if n.strip()]
    jobs = [(n, profile(n), P_start[n], times, out, a.grid) for n in names]
    if a.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(min(a.workers, len(jobs)), mp_context=mp.get_context('spawn')) as pool:
            parts = list(pool.map(run_heating, *zip(*jobs)))
    else:
        parts = [run_heating(*j) for j in jobs]
    rows = [r for p in parts for r in p]
    write_csv(out / 'history.csv', rows)
    summary = dict(target_X=TARGET_X, start_power_W={n: P_start[n] for n in names}, times=a.times, grid=a.grid,
                   power_limit_W=POWER_LIMIT * esmr.BASE_POWER_W)
    summary.update(compare(rows, POWER_LIMIT * esmr.BASE_POWER_W))
    summary['minutes'] = (time.time() - t0) / 60
    write_json(out / 'summary.json', summary)
    for n in names:
        m = summary[n]
        print(f"{n}: {m['nodes']} nodes to {m['t_last_h']:.0f} h, last power {m['P_last_W']:.1f} W, "
              + (f"left 483-682 C at {m['t_exit_h']:.0f} h" if m['left_range'] else 'stayed within 483-682 C'))
    print('results:', out / 'history.csv', out / 'summary.json')


if __name__ == '__main__':
    main()
