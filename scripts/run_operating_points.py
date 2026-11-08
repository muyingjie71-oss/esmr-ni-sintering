"""Aged beds at 55 combinations of CH4 feed and steam-to-carbon ratio (Fig. 5a,b).

    python scripts/run_history.py --profile uniform --out runs/U                        # aged bed
    python scripts/run_constant_production.py --times standard --out runs/cp            # beds aged at constant production
    python scripts/run_operating_points.py --fixed-power runs/U --constant-production runs/cp --out runs/points --workers 4
    python scripts/run_operating_points.py ... --f 1.0 --sc 2.96                         # one combination

For every combination (CH4 feed f = 0.5-1.5 times the baseline 2.40 Nm3/h, S/C 2.5-3.5, inlet 550 C and
6 bar, 1 m tube) the fresh bed under uniform heating at f x 3.52 kW sets the reference conversion X_ref.
Four Ni fields from the baseline, held fixed (no further ageing), are then solved for the power that gives X_ref:
  aged          871 h of uniform heating at 3.52 kW (fixed power), solved under uniform heating
  evenly_aged   retained Ni area 0.5 in every cell, uniform heating
  uniform_cp    871 h of uniform heating at constant production, uniform heating
  optimized_cp  871 h of optimized heating at constant production, optimized heating
Each power search starts at f x 3.52 kW, so its first solve is also the bed at fixed power. Combinations
with a shell outside 483-682 C are computed and flagged, not removed.

Results in <out>: points.csv (one row per combination: extra power of each bed, ratio of the extra powers of the
aged and the evenly aged bed, power saved by optimized_cp against uniform_cp), states.csv (one row per solved
bed); <out>/tasks/<id>.json.
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
from esmr.report import profile, read_json, write_csv, write_json

F_GRID = [round(.5 + .1 * i, 1) for i in range(11)]
SC_GRID = [2.5, 2.75, 2.96, 3.25, 3.5]
EVEN_AREA = .5                       # retained Ni area of the evenly aged bed
NM3_PER_MOL = 0.0224141              # m3 (0 C, 1 atm) per mol
T_REF = esmr.TIMES_H[-1]
BEDS = ('aged', 'evenly_aged', 'uniform_cp', 'optimized_cp')


def task_id(f, SC):
    return f'f{f:.1f}_sc{SC:.2f}'


def _row(name, heating, s, **kw):
    m = s.summary
    return dict(state=name, heating=heating, P_W=m['P_W'], X=m['X'], particle_Tmax_C=m['particle_Tmax_C'],
                particle_Tmin_C=m['particle_Tmin_C'], wall_max_C=m['wall_max_C'], in_range=m['in_range'], **kw)


def run_point(f, SC, beds, folder, grid='main'):
    """One combination: fresh reference and the power searches of the frozen beds.

    beds: list of (name, heating label, ten shares, Ni field)."""
    folder = Path(folder); Pref = esmr.BASE_POWER_W * f
    model = H.make_model(Pref, esmr.UNIFORM, f=f, SC=SC, grid=grid)
    nx, nr = model.nx, model.nr
    fresh = H.solve_frozen(model, np.ones((nx, nr)), None, folder / 'fresh')
    X_ref = fresh.summary['X']
    rows = [_row('fresh', 'uniform', fresh, status='OK')]
    for name, heating, q, h in beds:
        r = H.power_with_fallback(q, h, X_ref, Pref, fresh, folder / name, f=f, SC=SC, grid=grid)
        row = dict(state=name, heating=heating, status=r['status'], solves=r.get('solves'), root_mode=r['root_mode'])
        fp = r.get('first_state')
        if fp is not None and abs(fp.summary['P_W'] - Pref) <= 1e-6 * Pref:
            row.update(X_fixed_power=fp.summary['X'], dX_fixed_power_pp=fp.summary['X'] - X_ref,
                       Tmax_fixed_power_C=fp.summary['particle_Tmax_C'], wall_fixed_power_C=fp.summary['wall_max_C'])
        if r['status'] == 'OK':
            s = r['state'].summary
            row.update(P_W=r['P_W'], extra_power_pct=100 * (r['P_W'] / Pref - 1), X=s['X'], particle_Tmax_C=s['particle_Tmax_C'],
                       particle_Tmin_C=s['particle_Tmin_C'], wall_max_C=s['wall_max_C'], in_range=s['in_range'],
                       dTmax_vs_fresh_K=s['particle_Tmax_C'] - fresh.summary['particle_Tmax_C'], mean_Ni_pct=s['mean_Ni_pct'])
        else:
            row.update(error=r.get('error'))
        rows.append(row)
    by = {r['state']: r for r in rows}
    point = dict(f=f, S_C=SC, CH4_feed_Nm3_h=float(model.g.F0[0]) * 3600 * NM3_PER_MOL, P_ref_W=Pref, X_ref_pct=X_ref,
                 fresh_in_range=fresh.summary['in_range'])
    if all(by[k]['status'] == 'OK' for k in BEDS if k in by) and all(k in by for k in BEDS):
        point.update(ratio_aged_to_evenly_aged=by['aged']['extra_power_pct'] / by['evenly_aged']['extra_power_pct'],
                     saving_optimized_pct=100 * (by['uniform_cp']['P_W'] - by['optimized_cp']['P_W']) / by['uniform_cp']['P_W'],
                     all_states_in_range=bool(all(r.get('in_range') for r in rows)))
    return dict(point=point, rows=rows)


def _job(f, SC, beds, folder, grid):
    out = Path(folder) / 'tasks' / f'{task_id(f, SC)}.json'
    if out.exists():
        r = read_json(out)
        if not r.get('failed'):
            return r
    t0 = time.time()
    try:
        r = run_point(f, SC, beds, Path(folder) / 'tasks' / task_id(f, SC), grid)
        r.update(failed=False)
    except Exception as exc:  # noqa: BLE001  (recorded, the other combinations go on)
        r = dict(failed=True, error=repr(exc), traceback=traceback.format_exc(), point=dict(f=f, S_C=SC))
    r.update(id=task_id(f, SC), minutes=(time.time() - t0) / 60)
    write_json(out, r)
    return r


def aged_beds(fixed_power, constant_production, grid='main'):
    """The four frozen beds from the outputs of run_history.py and run_constant_production.py."""
    nat = esmr.load_state(Path(fixed_power) / 'states' / 'node11')
    if abs(nat.model_time_h - T_REF) > 1e-6:
        raise ValueError('node 11 of the fixed-power history is not at 871 h')
    cp = Path(constant_production); at = {}
    with open(cp / 'history.csv', newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['status'] == 'OK' and abs(float(r['t_h']) - T_REF) < 1e-6:
                at[r['heating']] = esmr.load_state(cp / r['state'])
    if not {'uniform', 'optimized'} <= set(at):
        raise ValueError('the constant-production histories do not reach 871 h')
    return [('aged', 'uniform', esmr.UNIFORM, nat.health),
            ('evenly_aged', 'uniform', esmr.UNIFORM, np.full(nat.health.shape, EVEN_AREA)),
            ('uniform_cp', 'uniform', esmr.UNIFORM, at['uniform'].health),
            ('optimized_cp', 'optimized', profile('optimized'), at['optimized'].health)]


def tables(results, out):
    points, states = [], []
    for r in sorted(results, key=lambda r: (r['point']['S_C'], r['point']['f'])):
        if r.get('failed'):
            points.append(dict(id=r['id'], **r['point'], failed=True, error=r.get('error', '')[:200])); continue
        p = r['point']; by = {s['state']: s for s in r['rows']}
        row = dict(id=r['id'], **{k: p[k] for k in ('f', 'S_C', 'CH4_feed_Nm3_h', 'P_ref_W', 'X_ref_pct', 'fresh_in_range')},
                   all_states_in_range=p.get('all_states_in_range'), fresh_Tmax_C=by['fresh']['particle_Tmax_C'],
                   fresh_Tmin_C=by['fresh']['particle_Tmin_C'])
        for s in BEDS:
            b = by.get(s, {})
            row.update({f'{s}_status': b.get('status'), f'{s}_P_W': b.get('P_W'), f'{s}_extra_pct': b.get('extra_power_pct'),
                        f'{s}_dX_fixed_power_pp': b.get('dX_fixed_power_pp'), f'{s}_root_mode': b.get('root_mode'),
                        f'{s}_Tmax_C': b.get('particle_Tmax_C'), f'{s}_Tmin_C': b.get('particle_Tmin_C'), f'{s}_in_range': b.get('in_range')})
        row.update(ratio_aged_to_evenly_aged=p.get('ratio_aged_to_evenly_aged'), saving_optimized_pct=p.get('saving_optimized_pct'))
        points.append(row)
        for s in r['rows']:
            states.append(dict(id=r['id'], f=p['f'], S_C=p['S_C'], **{k: v for k, v in s.items() if k != 'error'}))
    write_csv(out / 'points.csv', points); write_csv(out / 'states.csv', states)
    return points


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fixed-power', required=True, help='output folder of run_history.py --profile uniform (871 h, fixed power)')
    ap.add_argument('--constant-production', required=True, help='output folder of run_constant_production.py')
    ap.add_argument('--f', type=float, nargs='*', default=F_GRID, help='CH4 feed relative to the baseline')
    ap.add_argument('--sc', type=float, nargs='*', default=SC_GRID, help='steam-to-carbon ratios')
    ap.add_argument('--grid', default='main', choices=['main', 'coarse'])
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); t0 = time.time()
    beds = aged_beds(a.fixed_power, a.constant_production, a.grid)
    jobs = [(f, SC, beds, out, a.grid) for SC in a.sc for f in a.f]
    if a.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(min(a.workers, len(jobs)), mp_context=mp.get_context('spawn')) as pool:
            results = list(pool.map(_job, *zip(*jobs)))
    else:
        results = [_job(*j) for j in jobs]
    points = tables(results, out)
    good = [p for p in points if p.get('ratio_aged_to_evenly_aged') is not None]
    inside = [p for p in good if p.get('all_states_in_range')]
    summary = dict(points=len(points), complete=len(good), all_states_in_range=len(inside),
                   ratio_all=[min(p['ratio_aged_to_evenly_aged'] for p in good), max(p['ratio_aged_to_evenly_aged'] for p in good)] if good else None,
                   ratio_in_range=[min(p['ratio_aged_to_evenly_aged'] for p in inside), max(p['ratio_aged_to_evenly_aged'] for p in inside)] if inside else None,
                   saving_in_range_pct=[min(p['saving_optimized_pct'] for p in inside), max(p['saving_optimized_pct'] for p in inside)] if inside else None,
                   failed=[p['id'] for p in points if p.get('failed')], minutes=(time.time() - t0) / 60)
    write_json(out / 'summary.json', summary)
    print(summary)
    print('results:', out / 'points.csv', out / 'states.csv')


if __name__ == '__main__':
    main()
