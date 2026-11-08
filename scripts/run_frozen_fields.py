"""Tests on frozen Ni fields (Figs 1-3, Extended Data Fig. 2): the reactor is solved with a prescribed Ni field, without ageing.

  compare   aged against evenly aged bed and fresh bed: conversion at fixed power, and the power and
            peak catalyst temperature at which each bed restores the fresh conversion (Fig. 1)
  activity  all reaction rates multiplied by a common factor, for the fresh, aged and evenly aged beds
            (Fig. 2 and the activity equivalence of Fig. 3)
  zones     zone sensitivity S_k = X(s_k + 0.05) - X(s_k - 0.05), per 10 % of the fresh Ni area of zone k
  switch-off  all Ni of one zone removed from the fresh bed (Fig. 2a,b, Extended Data Fig. 2a,b): conversion
            and gas temperature along the tube, outlet conversion lost, change of the CH4 converted in each zone.
            The model needs a positive Ni area, so the zone gets 1e-12 instead of 0; this changes the outlet
            conversion by less than 1e-10 % points from solves with the zone exactly inert

The aged Ni field is read from a saved state, normally node11 of a uniform fixed-power history
(scripts/run_history.py --profile uniform --out runs/U  ->  runs/U/states/node11).

  python scripts/run_frozen_fields.py compare  --aged runs/U/states/node11 --out runs/compare
  python scripts/run_frozen_fields.py activity --aged runs/U/states/node11 --factors 0.5 1 2 4 8 16 --out runs/activity
  python scripts/run_frozen_fields.py zones    --base evenly_aged --heating optimized --out runs/zones_S
  python scripts/run_frozen_fields.py switch-off --zones 3 10 --out runs/switch_off
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # run from a checkout without installing the package
    sys.path.insert(0, str(ROOT))

import argparse

import numpy as np

import esmr
from esmr import histories as H
from esmr.report import profile, write_csv, write_json

P0 = esmr.BASE_POWER_W


def fields(aged_dir):
    aged = esmr.load_state(aged_dir)
    mean = float((aged.health @ aged.environment['annular_weights']).mean())
    return aged, dict(fresh=np.ones(aged.health.shape), aged=aged.health, evenly_aged=np.full(aged.health.shape, mean)), mean


def compare(a):
    out = Path(a.out); q = profile(a.heating); aged, F, mean = fields(a.aged)
    model = H.make_model(P0, q); rows = []
    fresh = H.solve_frozen(model, F['fresh'], None, out / 'fresh')
    for name in ('aged', 'evenly_aged'):
        s = H.solve_frozen(model, F[name], aged, out / name)
        r = H.power_for_conversion(q, F[name], fresh.summary['X'], P0, s, out / f'restoring_power_{name}')
        row = dict(bed=name, bed_mean_Ni_pct=100 * (mean if name != 'fresh' else 1.), X=s.summary['X'],
                   conversion_lost_pp=fresh.summary['X'] - s.summary['X'], wall_max_C=s.summary['wall_max_C'],
                   particle_Tmax_C=s.summary['particle_Tmax_C'], restoring_status=r['status'])
        if r['status'] == 'OK':
            rs = r['state'].summary
            row.update(restoring_power_W=r['P_W'], extra_power_pct=100 * (r['P_W'] / P0 - 1),
                       peak_rise_K=rs['particle_Tmax_C'] - fresh.summary['particle_Tmax_C'],
                       wall_rise_K=rs['wall_max_C'] - fresh.summary['wall_max_C'], in_range=rs['in_range'])
        rows.append(row); print(row, flush=True)
    rows.insert(0, dict(bed='fresh', bed_mean_Ni_pct=100., X=fresh.summary['X'], conversion_lost_pp=0.,
                        wall_max_C=fresh.summary['wall_max_C'], particle_Tmax_C=fresh.summary['particle_Tmax_C']))
    write_csv(out / 'COMPARE.csv', rows)


def activity(a):
    out = Path(a.out); aged, F, mean = fields(a.aged); q_S = profile('optimized'); rows = []
    for x in a.factors:
        variant = None if x == 1 else {'label': f'activity x{x:g}', 'sintering': {'activity_reference': float(x)}}
        d = out / f'x{x:g}'; res = dict(factor=x)
        seed = None
        for name, q, h in (('uniform_fresh', esmr.UNIFORM, F['fresh']), ('uniform_aged', esmr.UNIFORM, F['aged']),
                           ('uniform_evenly_aged', esmr.UNIFORM, np.full(aged.health.shape, .5)),
                           ('optimized_fresh', q_S, F['fresh']), ('optimized_evenly_aged', q_S, np.full(aged.health.shape, .5))):
            try:
                s = H.solve_frozen(H.make_model(P0, q, variant=variant), h, seed, d / name)
                res[name + '_X'] = s.summary['X']; res[name + '_outlet_C'] = s.summary['outlet_gas_C']
                if name == 'uniform_fresh':
                    seed = s
            except Exception as exc:  # noqa: BLE001
                res[name + '_error'] = repr(exc)[:200]
        rows.append(res); print(res, flush=True)
    write_csv(out / 'ACTIVITY.csv', rows)


def zones(a):
    out = Path(a.out); q = profile(a.heating); delta = 0.05
    if a.base == 'evenly_aged':
        base = np.full((40, 6), .5)
    else:
        base = esmr.load_state(a.aged).health
    model = H.make_model(P0, q)
    b = H.solve_frozen(model, base, esmr.load_state(a.aged) if a.aged else None, out / 'base')
    rows = []
    for k in range(10):
        X = {}
        for sign, tag in ((-1, 'minus'), (1, 'plus')):
            h = base.copy(); h[4 * k:4 * k + 4, :] += sign * delta
            X[tag] = H.solve_frozen(model, h, b, out / f'Z{k + 1}_{tag}').summary['X']
        rows.append(dict(zone=k + 1, X_minus=X['minus'], X_plus=X['plus'], X_base=b.summary['X'],
                         S_pp_per_10pct_Ni=(X['plus'] - X['minus']) / (2 * delta) * 0.1))
        print(rows[-1], flush=True)
    write_csv(out / 'ZONES.csv', rows)
    write_json(out / 'zones.json', dict(base=a.base, heating=a.heating, delta=delta, rows=rows))


ZERO_AREA = 1e-12                    # retained Ni area of a switched-off zone (the model needs s > 0)


def profiles(state):
    """Conversion, gas temperature and CH4 converted per zone along the tube."""
    e = state.steady.last; F = np.asarray(e.F_edges); nx = len(F) - 1
    X = 100 * (1 - F[:, 0] / F[0, 0])
    return dict(z_edges_m=np.linspace(0, state.conditions['length_m'], nx + 1).tolist(), X_pct=X.tolist(),
                X_zone_bounds_pct=X[::nx // 10].tolist(), Tgas_C=(np.asarray(e.Tgas_edges_K) - 273.15).tolist(),
                zone_converted_pp=(-100 * np.diff(F[:, 0]) / F[0, 0]).reshape(10, nx // 10).sum(1).tolist())


def switch_off(a):
    out = Path(a.out); q = profile(a.heating)
    model = H.make_model(P0, q); nx, nr = model.nx, model.nr
    fresh = H.solve_frozen(model, np.ones((nx, nr)), None, out / 'fresh')
    ref = profiles(fresh); prof = dict(fresh=ref); rows = []
    for z in a.zones:
        h = np.ones((nx, nr)); h[(z - 1) * nx // 10:z * nx // 10, :] = ZERO_AREA
        s = H.solve_frozen(model, h, fresh, out / f'Z{z}_off')
        p = profiles(s); prof[f'Z{z}_off'] = p
        change = np.asarray(p['zone_converted_pp']) - ref['zone_converted_pp']
        rows.append(dict(zone=z, X=s.summary['X'], outlet_loss_pp=fresh.summary['X'] - s.summary['X'],
                         outlet_gas_C=s.summary['outlet_gas_C'], outlet_gas_rise_K=s.summary['outlet_gas_C'] - fresh.summary['outlet_gas_C'],
                         loss_in_zone_pp=-float(change[z - 1]), gain_in_other_zones_pp=float(change.sum() - change[z - 1]),
                         **{f'change_Z{k + 1}_pp': float(v) for k, v in enumerate(change)}))
        print(rows[-1], flush=True)
    write_csv(out / 'SWITCH_OFF.csv', rows)
    write_json(out / 'switch_off.json', dict(heating=a.heating, zone_area=ZERO_AREA, X_fresh=fresh.summary['X'], rows=rows, profiles=prof))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='test', required=True)
    for name in ('compare', 'activity', 'zones', 'switch-off'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True)
        p.add_argument('--aged', required=name in ('compare', 'activity'), help='saved state with the aged Ni field')
        p.add_argument('--heating', default='uniform', help='uniform, optimized or a JSON file')
        if name == 'activity':
            p.add_argument('--factors', type=float, nargs='+', default=[0.5, 1, 2, 4, 8, 16])
        if name == 'zones':
            p.add_argument('--base', default='evenly_aged', choices=['evenly_aged', 'aged'])
        if name == 'switch-off':
            p.add_argument('--zones', type=int, nargs='+', default=list(range(1, 11)), help='zones 1-10')
    a = ap.parse_args()
    {'compare': compare, 'activity': activity, 'zones': zones, 'switch-off': switch_off}[a.test](a)


if __name__ == '__main__':
    main()
