"""Main comparisons at industrial pressure and outlet temperature (Extended Data Table 3).

Only steady states are solved; the Ni fields are those aged at the baseline (6 bar, 3.52 kW, 871 h), so
the sintering law is not used above 682 C. Inlet 550 C, S/C 2.96, H2/CH4 0.04 as in the baseline.
  B  21 bar, 1 m tube, baseline feed, outlet of the fresh bed under uniform heating at 800 C
  A  21 bar, 1 m tube, outlet 900 C
  C  6 bar, 1 m tube, outlet 900 C (separates the effects of pressure and temperature)
  D  21 bar, 3 m tube, three times the baseline feed, outlet 900 C
For each condition the total power is set so that the fresh bed under uniform heating reaches the outlet
temperature; at that power the fresh bed under optimized heating and, under uniform heating, the aged,
evenly aged and optimized-aged Ni fields are solved. At A all reaction rates are also scaled by 1/16 to 4.

  python scripts/run_industrial.py --out runs/industrial
Results: <out>/INDUSTRIAL_SUMMARY.json and .csv; states under <out>/states (a rerun resumes).
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
from scipy.optimize import brentq

import esmr
from esmr import histories as H
from esmr.report import profile, write_csv, write_json

P0 = esmr.BASE_POWER_W
FRESH_X = 49.394791933262994
U = esmr.UNIFORM
QS = profile('optimized')


def x_equilibrium(F_in, T_K, P_bar):
    """CH4 conversion (%) of the feed at reforming and shift equilibrium, equilibrium constants of the kinetic model."""
    K1 = math.exp(30.114 - 26830.0 / T_K); K2 = math.exp(4400.0 / T_K - 4.036)
    n0 = np.asarray(F_in, float) / F_in[0]

    def comp(x, w):
        return np.array([n0[0] - x, n0[1] - x - w, n0[2] + 3 * x + w, n0[3] + x - w, n0[4] + w])

    def shift_w(x):
        lo, hi = -n0[4] + 1e-14, min(n0[3] + x, n0[1] - x) - 1e-14
        g = lambda w: (lambda n: n[4] * n[2] - K2 * n[3] * n[1])(comp(x, w))
        return brentq(g, lo, hi) if hi > lo else lo

    def r(x):
        n = comp(x, shift_w(x)); p = n / n.sum() * P_bar
        return math.log(p[3] * p[2] ** 3 / (p[0] * p[1])) - math.log(K1)
    x = brentq(r, 1e-9, min(n0[0], n0[1]) - 1e-9)
    return 100.0 * x / n0[0]


def approach(state):
    """Outlet approach to equilibrium (temperature form) and the conversion gap to equilibrium at the outlet."""
    e = state.steady.last
    Fin = np.asarray(e.F_edges[0], float); F = np.asarray(e.F_edges[-1], float)
    P = e.P_edges_Pa[-1] / 1e5; T = e.Tgas_edges_K[-1]
    p = F / F.sum() * P
    Q1 = p[3] * p[2] ** 3 / (p[0] * p[1])
    Teq = 26830.0 / (30.114 - math.log(Q1))
    X = 100.0 * (1 - F[0] / Fin[0])
    try:
        Xeq = x_equilibrium(Fin, T, P)
    except Exception:  # noqa: BLE001
        Xeq = float('nan')
    return dict(outlet_gas_C=float(T - 273.15), T_eq_C=float(Teq - 273.15), approach_K=float(T - Teq), p_out_bar=float(P),
                X_eq_at_outlet_pct=Xeq, gap_to_equilibrium_pp=Xeq - X)


def solve(q, P, f, health, seed, folder, Pin=6., L=1., variant=None):
    m = H.make_model(float(P), q, f=f, Pin_bar=Pin, length_m=L, variant=variant)
    h = np.ones((m.nx, m.nr)) if health is None else np.asarray(health, np.float64)
    s = H.solve_frozen(m, h, seed, folder)
    keep = ('X', 'P_W', 'particle_Tmax_C', 'wall_max_C', 'in_range', 'mean_Ni_pct')
    return s, dict({k: s.summary[k] for k in keep}, **approach(s))


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def march_T(q, f, T_target, P_start, seed, d, tag, Pin, L, health=None, slope0=0.045, tol=1.0, maxstep=0.2, max_solves=16):
    """Change the total power until the outlet gas temperature is T_target (within tol K); each solve is seeded by
    the previous one, and a failed solve halves the power step (at most four times)."""
    pts, P, cur, halvings, n = [], float(P_start), seed, 0, 0
    while n < max_solves:
        folder = Path(d) / f'it{n:02d}'; n += 1
        try:
            st, s = solve(q, P, f, health, cur, folder, Pin, L)
        except Exception as exc:  # noqa: BLE001
            log(f'  {tag}: solve failed at P = {P:.1f} W ({exc!r}); halving the step')
            if not pts or halvings >= 4:
                raise
            halvings += 1; P = pts[-1][0] + .5 * (P - pts[-1][0]); continue
        T = s['outlet_gas_C']; pts.append((P, T))
        log(f'  {tag}: P {P:.1f} W -> outlet {T:.2f} C, X {s["X"]:.3f} %, approach {s["approach_K"]:.2f} K')
        cur = st
        if abs(T - T_target) <= tol:
            return dict(P_W=P, state=st, summary=s)
        slope = slope0
        if len(pts) >= 2 and pts[-1][0] != pts[-2][0]:
            sl = (pts[-1][1] - pts[-2][1]) / (pts[-1][0] - pts[-2][0])
            if sl > 0:
                slope = sl
        dP = (T_target - T) / slope
        P = P + max(min(dP, maxstep * P), -maxstep * P)
    raise RuntimeError(f'{tag}: outlet temperature {T_target} C not reached in {max_solves} solves')


def frozen_set(cond, P, f, Pin, L, seed, nat, s50, d):
    hom = np.full(nat.health.shape, float((nat.health @ nat.environment['annular_weights']).mean()))
    out = {}
    try:
        _, out['S_fresh'] = solve(QS, P, f, None, seed, d / 'S_fresh', Pin, L)
    except Exception as exc:  # noqa: BLE001  continue through a 50/50 blend of the two profiles
        log(f'{cond}: optimized fresh bed failed directly ({exc!r}); continuing through a 50/50 blend')
        blend = np.maximum(.5 * np.asarray(U) + .5 * np.asarray(QS), 0.); blend = (blend / blend.sum()).tolist()
        sb, _ = solve(blend, P, f, None, seed, d / 'S_blend50', Pin, L)
        _, out['S_fresh'] = solve(QS, P, f, None, sb, d / 'S_fresh_b', Pin, L)
    for name, h in (('U_NAT50', nat.health), ('U_HOM50', hom), ('U_S50', s50.health)):
        _, out[name] = solve(U, P, f, h, seed, d / name, Pin, L)
        log(f'{cond} {name}: X {out[name]["X"]:.4f} %, approach {out[name]["approach_K"]:.2f} K')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); S = out / 'states'; rep = dict(checks={}, conditions={}, activity={})
    fresh, s0 = solve(U, P0, 1., None, None, S / 'base_fresh')
    rep['checks']['baseline_fresh_X'] = dict(value=s0['X'], reference=FRESH_X, ok=abs(s0['X'] - FRESH_X) < 1e-6)
    if not rep['checks']['baseline_fresh_X']['ok']:
        raise SystemExit('the baseline fresh bed is not reproduced on this computer')
    hU = H.history_fixed_power(H.make_model(P0, U), folder=S / 'history_U')
    hS = H.history_fixed_power(H.make_model(P0, QS), folder=S / 'history_S')
    nat, s50 = hU[-1], hS[-1]
    _, s50u = solve(U, P0, 1., s50.health, s50, S / 'S50_under_U')
    rep['checks']['aged_871h_uniform_X'] = nat.summary['X']; rep['checks']['J_baseline'] = s50u['X'] - nat.summary['X']
    # 21 bar: continuation in pressure, then power to 800 C (B) and 900 C (A)
    st = fresh
    for Pin in (12., 21.):
        st, _ = solve(U, P0, 1., None, st, S / f'L21_P{Pin:g}bar', Pin=Pin)
    cond = {}
    cond['B'] = march_T(U, 1., 800., P0, st, S / 'B_to800', 'B', 21., 1.) | dict(Pin=21., L=1., f=1.)
    cond['A'] = march_T(U, 1., 900., cond['B']['P_W'], cond['B']['state'], S / 'A_to900', 'A', 21., 1.) | dict(Pin=21., L=1., f=1.)
    cond['C'] = march_T(U, 1., 900., P0, fresh, S / 'C_to900', 'C', 6., 1., slope0=0.04) | dict(Pin=6., L=1., f=1.)
    st = cond['A']['state']; PA = cond['A']['P_W']
    for L in (1.5, 2.0, 2.5, 3.0):                       # 3 m tube: continuation in length with power and feed per metre fixed
        st, _ = solve(U, PA * L, L, None, st, S / f'D_L{L:g}m', Pin=21., L=L)
    cond['D'] = march_T(U, 3., 900., PA * 3., st, S / 'D_to900', 'D', 21., 3., slope0=0.015) | dict(Pin=21., L=3., f=3.)
    rows = []
    for c, r in cond.items():
        states = frozen_set(c, r['P_W'], r['f'], r['Pin'], r['L'], r['state'], nat, s50, S / f'{c}_frozen')
        xf = r['summary']['X']; lN, lH = xf - states['U_NAT50']['X'], xf - states['U_HOM50']['X']
        rep['conditions'][c] = dict(P_W=r['P_W'], Pin_bar=r['Pin'], length_m=r['L'], feed_factor=r['f'], U_fresh=r['summary'], **states,
                                    derived=dict(loss_aged_pp=lN, loss_evenly_aged_pp=lH, ratio_aged_to_evenly_aged=lN / lH if lH else None,
                                                 J_pp=states['U_S50']['X'] - states['U_NAT50']['X'],
                                                 gain_optimized_fresh_pp=states['S_fresh']['X'] - xf))
        for n in ('U_fresh', 'S_fresh', 'U_NAT50', 'U_HOM50', 'U_S50'):
            s = r['summary'] if n == 'U_fresh' else states[n]
            rows.append(dict(condition=c, P_W=r['P_W'], heating_or_field=n, **{k: s[k] for k in (
                'X', 'outlet_gas_C', 'approach_K', 'X_eq_at_outlet_pct', 'gap_to_equilibrium_pp', 'particle_Tmax_C', 'wall_max_C')}))
    for chain in ((2, 4), (0.5, 0.25, 0.125, 0.0625)):     # activity at condition A, each factor seeded by the previous one
        seed_U, seed_S = cond['A']['state'], None
        for x in chain:
            var = {'label': f'activity x{x:g}', 'sintering': {'activity_reference': float(x)}}
            sU, rU = solve(U, PA, 1., None, seed_U, S / f'A_activity_x{x:g}_U', 21., 1., var)
            sS, rS = solve(QS, PA, 1., None, seed_S or sU, S / f'A_activity_x{x:g}_S', 21., 1., var)
            rep['activity'][f'{x:g}'] = dict(U_fresh=rU, S_fresh=rS); seed_U, seed_S = sU, sS
            log(f'activity x{x:g}: uniform X {rU["X"]:.3f} %, approach {rU["approach_K"]:.2f} K | optimized X {rS["X"]:.3f} %')
    write_json(out / 'INDUSTRIAL_SUMMARY.json', rep); write_csv(out / 'INDUSTRIAL_SUMMARY.csv', rows)
    for c, v in rep['conditions'].items():
        d = v['derived']
        log(f"{c}: P {v['P_W']:.0f} W | loss aged {d['loss_aged_pp']:.3f} pp, evenly aged {d['loss_evenly_aged_pp']:.3f} pp, "
            f"ratio {d['ratio_aged_to_evenly_aged']:.2f} | J {d['J_pp']:.3f} pp")


if __name__ == '__main__':
    main()
