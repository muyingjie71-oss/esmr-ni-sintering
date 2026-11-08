"""Sensitivity tests of Extended Data Fig. 6: the main comparisons repeated with one model change at a time.

For each variant:
 1. time to 50 % bed-averaged Ni loss under uniform heating at 3.52 kW (geometric probe clock,
    log-time interpolation), t50;
 2. fixed-power histories under uniform and optimized heating on the 12-node grid scaled by t50/871 h,
    with the thermal limits of the profile search checked at every node;
 3. J from the two final Ni fields, both solved under uniform heating;
 4. the aged bed (end of the uniform history) against the evenly aged bed with the same bed-averaged Ni
    area: conversion lost at fixed power, and peak catalyst temperature rise at the power that restores
    the fresh conversion, both relative to the fresh bed of the same variant.

  python scripts/run_sensitivity.py --out runs/sensitivity              # all 17 (S0 is the baseline)
  python scripts/run_sensitivity.py --out runs/sensitivity --only S9 H3
Results: <out>/<id>/result.json and <out>/SENSITIVITY.csv.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # run from a checkout without installing the package
    sys.path.insert(0, str(ROOT))

import argparse
import math
import time

import numpy as np

import esmr
from esmr import histories as H
from esmr.report import profile, read_json, write_csv, write_json

VARIANTS = [
    ('S0', 'baseline', {}),
    ('S1', 'rate fit with E = 142 / 410 kJ/mol', {'sintering': {'fit': 'published_slopes'}}),
    ('S2', 'rate fit with E = 144.5 / 427.8 kJ/mol', {'sintering': {'fit': 'free_slopes'}}),
    ('S3', 'exponent of F/Fref 0.5', {'sintering': {'atmosphere_exponent': 0.5}}),
    ('S4', 'exponent of F/Fref 0', {'sintering': {'atmosphere_exponent': 0.0}}),
    ('S5', 'activity s^0.5', {'sintering': {'activity_exponent': 0.5}}),
    ('S6', 'activity s^1.5', {'sintering': {'activity_exponent': 1.5}}),
    ('S7', 'rate fit with E = 142 / 410 kJ/mol, switch at 575 C', {'sintering': {'fit': 'published_slopes', 'T_switch_C': 575.0}}),
    ('S8', 'rate fit with E = 142 / 410 kJ/mol, switch at 625 C', {'sintering': {'fit': 'published_slopes', 'T_switch_C': 625.0}}),
    ('S9', 'activity s^2', {'sintering': {'activity_exponent': 2.0}}),
    ('H1', 'wall-to-bed coefficient x0.7', {'transport': {'hw': 0.7}}),
    ('H2', 'wall-to-bed coefficient x1.3', {'transport': {'hw': 1.3}}),
    ('H3', 'radial bed conductivity x0.7', {'transport': {'kr': 0.7}}),
    ('H4', 'radial bed conductivity x1.3', {'transport': {'kr': 1.3}}),
    ('H5', 'gas-solid coefficient x0.7', {'transport': {'hgs': 0.7}}),
    ('K1', 'all reaction rates x0.5', {'sintering': {'activity_reference': 0.5}}),
    ('K2', 'all reaction rates x2', {'sintering': {'activity_reference': 2.0}}),
]
PROBE = [0.] + list(np.geomspace(esmr.TIMES_H[1], esmr.TIMES_H[1] * 1e5, 21))     # 0.87 h ... 87,122 h


def run_variant(vid, label, variant, q_S, d):
    P0, U = esmr.BASE_POWER_W, esmr.UNIFORM
    variant = dict(variant, label=label) if variant else None
    mU = H.make_model(P0, U, variant=variant); mS = H.make_model(P0, q_S, variant=variant)
    out = dict(id=vid, label=label, variant=variant)
    probe = H.history_fixed_power(mU, PROBE, folder=d / 'probe_U', stop=lambda s: s.summary['mean_Ni_pct'] <= 50.)
    out['probe'] = [dict(t_h=s.model_time_h, mean_Ni_pct=s.summary['mean_Ni_pct'], particle_Tmax_C=s.summary['particle_Tmax_C'],
                         in_range=s.summary['in_range']) for s in probe]
    if probe[-1].summary['mean_Ni_pct'] > 50.:
        out.update(status='NI50_NOT_REACHED'); return out
    k = len(probe) - 1
    t1, m1 = probe[k - 1].model_time_h, probe[k - 1].summary['mean_Ni_pct']
    t2, m2 = probe[k].model_time_h, probe[k].summary['mean_Ni_pct']
    a = (m1 - 50.) / (m1 - m2)
    t50 = t1 + a * (t2 - t1) if t1 <= 0 else math.exp(math.log(t1) + a * (math.log(t2) - math.log(t1)))
    times = [t * t50 / esmr.TIMES_H[-1] for t in esmr.TIMES_H]
    out.update(t50_h=t50, times_h=times)
    hU = H.history_fixed_power(mU, times, folder=d / 'hist_U')
    hS = H.history_fixed_power(mS, times, folder=d / 'hist_S')
    dp = [s.summary['particle_Tmax_C'] - u.summary['particle_Tmax_C'] for s, u in zip(hS, hU)]
    dw = [s.summary['wall_max_C'] - u.summary['wall_max_C'] for s, u in zip(hS, hU)]
    out.update(dTmax_optimized_minus_uniform_K=dp, dWall_optimized_minus_uniform_K=dw,
               thermal_limits_met=bool(max(dp) <= 1e-9 and max(dw) <= 1e-9),
               all_states_in_483_682C=bool(all(s.summary['in_range'] for s in hU + hS)),
               history_uniform=[dict(t_h=s.model_time_h, X=s.summary['X'], particle_Tmax_C=s.summary['particle_Tmax_C'],
                                     in_range=s.summary['in_range']) for s in hU],
               history_optimized=[dict(t_h=s.model_time_h, X=s.summary['X'], particle_Tmax_C=s.summary['particle_Tmax_C'],
                                       in_range=s.summary['in_range']) for s in hS])
    tU = H.solve_frozen(mU, hU[-1].health, hU[-1], d / 'utest_U'); tS = H.solve_frozen(mU, hS[-1].health, hS[-1], d / 'utest_S')
    out.update(X_test_uniform=tU.summary['X'], X_test_optimized=tS.summary['X'], J_pp=tS.summary['X'] - tU.summary['X'])
    fr, nat = hU[0].summary, hU[-1]
    mean = float((nat.health @ nat.environment['annular_weights']).mean())
    hom = H.solve_frozen(mU, np.full(nat.health.shape, mean), nat, d / 'evenly_aged')
    cmp = dict(bed_mean_Ni_pct=100 * mean)
    for name, s in (('aged', nat), ('evenly_aged', hom)):
        cmp[name] = dict(dX_pp=s.summary['X'] - fr['X'], dTpeak_K=s.summary['particle_Tmax_C'] - fr['particle_Tmax_C'],
                         dTmean_K=s.summary['particle_Tmean_C'] - fr['particle_Tmean_C'])
        r = H.power_for_conversion(U, s.health, fr['X'], P0, s, d / f'restoring_power_{name}', variant=variant)
        cmp[name]['restoring_power'] = dict(status=r['status'], P_W=r.get('P_W'), solves=r.get('solves'), error=r.get('error'))
        if r['status'] == 'OK':
            rs = r['state'].summary
            cmp[name]['restoring_power'].update(dTpeak_K=rs['particle_Tmax_C'] - fr['particle_Tmax_C'], in_range=rs['in_range'])
    n, h = cmp['aged'], cmp['evenly_aged']
    cmp['ratio_conversion_lost'] = n['dX_pp'] / h['dX_pp'] if h['dX_pp'] else None
    cmp['ratio_peak_rise_fixed_power'] = n['dTpeak_K'] / h['dTpeak_K'] if h['dTpeak_K'] else None
    a_, b_ = n['restoring_power'].get('dTpeak_K'), h['restoring_power'].get('dTpeak_K')
    cmp['ratio_peak_rise_restoring_power'] = a_ / b_ if (a_ is not None and b_) else None
    out.update(aged_vs_evenly_aged=cmp, status='OK')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True); ap.add_argument('--only', nargs='*')
    ap.add_argument('--profile', default='optimized', help='heating profile compared with uniform heating')
    a = ap.parse_args()
    out = Path(a.out); q_S = profile(a.profile); rows = []
    for vid, label, variant in VARIANTS:
        if a.only and vid not in a.only:
            continue
        d = out / vid; t0 = time.time()
        res = read_json(d / 'result.json') if (d / 'result.json').exists() else None
        if res is None:
            res = run_variant(vid, label, variant, q_S, d); res['minutes'] = (time.time() - t0) / 60
            write_json(d / 'result.json', res)
        c = res.get('aged_vs_evenly_aged', {})
        rows.append(dict(id=vid, label=label, status=res['status'], t50_h=res.get('t50_h'), J_pp=res.get('J_pp'),
                         ratio_conversion_lost=c.get('ratio_conversion_lost'),
                         ratio_peak_rise_restoring_power=c.get('ratio_peak_rise_restoring_power'),
                         thermal_limits_met=res.get('thermal_limits_met'), all_states_in_483_682C=res.get('all_states_in_483_682C')))
        print(rows[-1], flush=True)
    write_csv(out / 'SENSITIVITY.csv', rows)


if __name__ == '__main__':
    main()
