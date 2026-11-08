"""Ageing history of one heating profile (Figs 1, 3 and 4).

  fixed power (Figs 1-3):         python scripts/run_history.py --profile uniform --out runs/U
  every 5 % of Ni area (Fig. 1c): python scripts/run_history.py --profile uniform --area-steps --out runs/U_area
  constant production:            python scripts/run_history.py --profile optimized --mode constant_production --out runs/cp_S
                                  (Fig. 4 compares both heatings: scripts/run_constant_production.py)

The history stops at the first node with a pellet shell outside 483-682 C (the data range of the
sintering law) unless --no-stop is given. With --uniform-test the final Ni field is also solved
under uniform heating at 3.52 kW (X_U of the objective J, Methods). Each state is saved under
<out>/states; a rerun resumes from the saved states. Results: <out>/history.csv, <out>/summary.json.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # run from a checkout without installing the package
    sys.path.insert(0, str(ROOT))

import argparse
import time

import esmr
from esmr import histories as H
from esmr.report import node_row, profile, write_csv, write_json

TARGET_X = 49.394791933262994        # fresh bed, uniform heating, 3.52 kW: target of constant production


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', default='uniform', help="uniform, optimized or a JSON file with ten zone shares")
    ap.add_argument('--mode', default='fixed_power', choices=['fixed_power', 'constant_production'])
    ap.add_argument('--times', default='standard', choices=['standard', 'long', 'half', 'double'],
                    help='standard: 12 nodes to 871 h; long: on to 20,628 h; half/double: half or twice the step')
    ap.add_argument('--area-steps', action='store_true', help='steps that lower the bed-averaged Ni area by 5 %% each, to 50 %%')
    ap.add_argument('--power', type=float, default=esmr.BASE_POWER_W, help='fixed power, or first guess at constant production, W')
    ap.add_argument('--target', type=float, default=TARGET_X, help='CH4 conversion held at constant production, %%')
    ap.add_argument('--grid', default='main', choices=['main', 'coarse'])
    ap.add_argument('--precision', default='main', choices=['main', 'strict', 'finer'])
    ap.add_argument('--no-stop', action='store_true', help='continue past nodes outside 483-682 C')
    ap.add_argument('--uniform-test', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); q = profile(a.profile); t0 = time.time()
    times = dict(standard=esmr.TIMES_H, long=H.LONG_TIMES_H, half=H.half_step(esmr.TIMES_H), double=H.double_step())[a.times]
    kw = dict(grid=a.grid, precision=a.precision)
    summary = dict(profile=q, mode=a.mode, times=a.times, grid=a.grid, precision=a.precision)
    if a.mode == 'fixed_power':
        model = H.make_model(a.power, q, **kw)
        if a.area_steps:
            nodes = H.history_to_mean_area(model, folder=out / 'states')
        else:
            nodes = H.history_fixed_power(model, times, folder=out / 'states',
                                          stop=None if a.no_stop else (lambda s: not s.summary['in_range']))
        rows = [node_row(s, node=k) for k, s in enumerate(nodes)]
        last = nodes[-1]
    else:
        res = H.history_constant_production(q, a.target, times, P_start=a.power, folder=out / 'states', **kw)
        rows = []
        for r in res:
            if r['status'] != 'OK':
                rows.append(dict(node=r['node'], t_h=r['t_h'], status=r['status'], root_mode=r.get('root_mode'))); continue
            rows.append(node_row(r['state'], node=r['node'], t_h=r['t_h'], status='OK', root_iterations=r['root_iterations'],
                                 root_mode=r['root_mode']))
        ok = [r for r in res if r['status'] == 'OK']
        last = ok[-1]['state'] if ok else None
        summary['target_X'] = a.target
    write_csv(out / 'history.csv', rows)
    summary.update(nodes=len(rows), last=rows[-1] if rows else None)
    if a.uniform_test and last is not None:
        summary['X_uniform_test'] = H.uniform_test(last.health, last, out / 'states' / 'uniform_test', **kw)
    summary['minutes'] = (time.time() - t0) / 60
    write_json(out / 'summary.json', summary)
    print('history:', out / 'history.csv')
    if 'X_uniform_test' in summary:
        print(f"X under uniform heating of the final Ni field: {summary['X_uniform_test']:.6f} %")


if __name__ == '__main__':
    main()
