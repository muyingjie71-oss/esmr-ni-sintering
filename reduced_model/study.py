"""Outlet-equilibrium rule for five endothermic, equilibrium-limited reactions (reduced model, see generic_reactor.py).
usage: python3 study.py <out GENERIC_RESULTS.json> [reaction names, default all five]
NH3 and PDH give dH and dS at T0 = Tr (317.4 and 586.1 C, the equilibrium temperatures at 50 % conversion) and carry them to other
temperatures with the constant cp (Kirchhoff, generic_reactor.py); SMR, DRM and RWGS use NIST Shomate data."""
import json, os, sys, time
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generic_reactor import Reaction, Reactor, w_uniform, w_early, nist_cp, R

ES = 142e3          # sintering activation energy (Sehested, low-temperature branch)
TAIL = 0.3

RX = {
    'SMR': Reaction('CH4 + H2O -> CO + 3H2', ['CH4', 'H2O', 'CO', 'H2'], [-1, -1, 1, 3], [1, 3, 0, .04], 6., 823.15, 120e3),
    'DRM': Reaction('CH4 + CO2 -> 2CO + 2H2', ['CH4', 'CO2', 'CO', 'H2'], [-1, -1, 2, 2], [1, 1, 0, .02], 1., 873.15, 150e3),
    'RWGS': Reaction('CO2 + H2 -> CO + H2O', ['CO2', 'H2', 'CO', 'H2O'], [-1, -1, 1, 1], [1, 3, 0, 0], 5., 673.15, 80e3),
    'NH3': Reaction('NH3 -> 1/2 N2 + 3/2 H2', ['NH3', 'N2', 'H2'], [-1, .5, 1.5], [1, 0, 0], 40., 523.15, 100e3,
                    thermo='const', dH=54.5e3, dS=116., cp=[45., 30.5, 29.5], T0=317.4 + 273.15),
    'PDH': Reaction('C3H8 -> C3H6 + H2', ['C3H8', 'C3H6', 'H2'], [-1, 1, 1], [1, 0, 0], 1., 773.15, 110e3,
                    thermo='const', dH=129e3, dS=141., cp=[135., 115., 29.8], T0=586.1 + 273.15),
}


def Fx(rx, X):
    return rx.F0 + rx.nu * X * rx.F0[rx.key]


def p_for_eq(rx, X):
    """power at which the outlet, at equilibrium, has conversion X (energy balance along a path: react at Tin, heat products)"""
    F = Fx(rx, X); Te = rx.Teq(F)
    Ts = np.linspace(rx.Tin, Te, 60)
    sens = np.trapezoid([float(np.dot(F, rx.cp(T))) for T in Ts], Ts)
    return rx.dH(rx.Tin) * X * rx.F0[rx.key] + sens, Te


def calibrate(rx, rr, X_eq=.5, ate_target=20.):
    P, Te = p_for_eq(rx, X_eq)
    a1 = np.ones_like(rr.zc)
    Tr = Te

    def ate(lk):
        pr = rr.solve(P, w_uniform(), a1, np.exp(lk), Tr); return rr.ate(pr) - ate_target
    lk = brentq(ate, np.log(1e-6), np.log(1e3), xtol=1e-4)
    return P, np.exp(lk), Tr, Te


def root_P(rr, w, a, k0, Tr, Xt, P0):
    f = lambda P: rr.solve(P, w, a, k0, Tr)['Xout'] - Xt
    lo, hi = .6 * P0, 1.6 * P0
    return brentq(f, lo, hi, xtol=1e-6 * P0)


def history(rx, rr, w, k0, Tr, Xt, P_start, taus):
    s = np.ones_like(rr.zc); x = np.zeros_like(rr.zc); out = []; P = P_start
    for i, tau in enumerate(taus):
        P = root_P(rr, w, s, k0, Tr, Xt, P)
        pr = rr.solve(P, w, s, k0, Tr)
        out.append(dict(tau=tau, P=P, s_mean=float(s.mean()), s_out=float(s[-5:].mean()), ATE=rr.ate(pr), Tmax=pr['Tmax'], Tout=pr['Tout'],
                        s=s.copy()))
        if i + 1 < len(taus):
            T = np.interp(rr.zc, pr['z'], pr['T'])
            x = x + np.exp(-ES / R * (1. / T - 1. / Tr)) * (taus[i + 1] - tau)
            s = (1. + x) ** (-1. / 7.)
    return out


def tail_lambda(rr, pr, tail=TAIL):
    z = pr['z']; m = z >= 1 - tail - 1e-9
    ate = np.array([pr['T'][i] - rr.rx.Teq(pr['F'][i]) for i in np.where(m)[0]])
    ok = ate > .05
    if ok.sum() < 5:
        return None
    c = np.polyfit(z[m][ok], np.log(ate[ok]), 1)
    return -1. / c[0]


def run(name):
    rx = RX[name]; rr = Reactor(rx); t0 = time.time()
    P, k0, Tr, Te = calibrate(rx, rr)
    one = np.ones_like(rr.zc)
    res = dict(name=name, reaction=rx.name, p_bar=rx.p, Tin_C=rx.Tin - 273.15, E_kJ=rx.E / 1e3, P_ref=P, k0=k0, Teq_50_C=Te - 273.15)
    profiles = {'uniform': w_uniform(), 'stop_flat': w_early(TAIL, shape='flat'), 'stop_front': w_early(TAIL, shape='triangle')}
    fresh = {}
    for k, w in profiles.items():
        pr = rr.solve(P, w, one, k0, Tr)
        fresh[k] = dict(X=pr['Xout'], ATE=rr.ate(pr), Tout_C=pr['Tout'] - 273.15, Tmax_C=pr['Tmax'] - 273.15, lam=tail_lambda(rr, pr) if k != 'uniform' else None)
    res['fresh_at_Pref'] = fresh
    Xt = fresh['uniform']['X']; res['X_target'] = Xt
    # pick the stop-early shape with the lower peak temperature as the design
    design = min(('stop_flat', 'stop_front'), key=lambda k: fresh[k]['Tmax_C']); res['design'] = design; wd = profiles[design]
    Pd0 = root_P(rr, wd, one, k0, Tr, Xt, P)
    res['fresh_power_saving_pct'] = 100 * (P - Pd0) / P
    # constant-production histories (time in units of 1/ks(Tr)); find tau50 of uniform heating first
    taus = list(np.geomspace(1e-3, 1e5, 33))
    hU = history(rx, rr, profiles['uniform'], k0, Tr, Xt, P, [0.] + taus)
    sm = np.array([h['s_mean'] for h in hU]); tt = np.array([h['tau'] for h in hU])
    i50 = int(np.argmax(sm <= .5)); tau50 = float(np.exp(np.interp(.5, sm[i50:i50 - 2:-1], np.log(tt[i50:i50 - 2:-1])))) if i50 > 0 else None
    clock = [0.] + list(tau50 * np.geomspace(1e-3, 1, 12)[1:]) + list(tau50 * (4 / 3) ** np.arange(1, 12))   # same clock shape as the eSMR study
    hU = history(rx, rr, profiles['uniform'], k0, Tr, Xt, P, clock)
    hD = history(rx, rr, wd, k0, Tr, Xt, Pd0, clock)
    i871 = 11
    res['tau50'] = tau50
    res['history'] = [dict(t_over_t50=h['tau'] / tau50, P_U=h['P'], P_D=g['P'], ATE_U=h['ATE'], ATE_D=g['ATE'], smean_U=h['s_mean'], smean_D=g['s_mean'],
                           sout_U=h['s_out'], sout_D=g['s_out'], Tmax_U_C=h['Tmax'] - 273.15, Tmax_D_C=g['Tmax'] - 273.15) for h, g in zip(hU, hD)]
    r871 = res['history'][i871]
    res['at_t50'] = dict(extra_U_pct=100 * (r871['P_U'] / P - 1), extra_D_pct=100 * (r871['P_D'] / Pd0 - 1), saving_pct=100 * (r871['P_U'] - r871['P_D']) / r871['P_U'],
                         ATE_U=r871['ATE_U'], ATE_D=r871['ATE_D'], sout_U=r871['sout_U'], sout_D=r871['sout_D'])
    last = res['history'][-1]
    res['at_end'] = dict(t_over_t50=last['t_over_t50'], extra_U_pct=100 * (last['P_U'] / P - 1), extra_D_pct=100 * (last['P_D'] / Pd0 - 1),
                         saving_pct=100 * (last['P_U'] - last['P_D']) / last['P_U'], ATE_U=last['ATE_U'], ATE_D=last['ATE_D'])
    # uneven vs uniform loss at t50 (uniform heating), and the same uniform loss under the design
    sN = hU[i871]['s']; sH = np.full_like(sN, float(sN.mean()))
    PN = root_P(rr, profiles['uniform'], sN, k0, Tr, Xt, P); PH = root_P(rr, profiles['uniform'], sH, k0, Tr, Xt, P)
    PHd = root_P(rr, wd, sH, k0, Tr, Xt, Pd0)
    res['loss_location'] = dict(mean_s=float(sN.mean()), s_in=float(sN[:5].mean()), s_out=float(sN[-5:].mean()),
                                extra_uneven_pct=100 * (PN / P - 1), extra_uniform_pct=100 * (PH / P - 1), ratio=(PN - P) / (PH - P),
                                extra_uniform_under_design_pct=100 * (PHd / Pd0 - 1))
    # catalyst equivalence at P_ref: activity multiplier for uniform heating to reach the design's conversion
    Xd = rr.solve(P, wd, one, k0, Tr)['Xout']
    try:
        xm = float(np.exp(brentq(lambda lx: rr.solve(P, profiles['uniform'], one * np.exp(lx), k0, Tr)['Xout'] - (Xd - 5e-4), 0., np.log(1e6), xtol=1e-3)))   # within 0.05 pp
    except ValueError:
        xm = None
    xl = float(np.exp(brentq(lambda lx: rr.solve(P, wd, one * np.exp(lx), k0, Tr)['Xout'] - Xt, np.log(1e-4), 0., xtol=1e-3)))
    res['catalyst_equivalence'] = dict(X_design=Xd, uniform_needs_x_within_005pp=xm, design_matches_uniform_with_x=xl)
    # energy identity: conversion vs outlet temperature at P_ref (all fixed-power fresh/aged states of both heatings)
    pts = []
    for a in (one, sH, sN, one * 4, one * .25):
        for w in (profiles['uniform'], wd):
            pr = rr.solve(P, w, a, k0, Tr); pts.append((pr['Tout'], pr['Xout']))
    pts = np.array(pts); c = np.polyfit(pts[:, 0], pts[:, 1], 1); r = pts[:, 1] - np.polyval(c, pts[:, 0])
    res['energy_identity'] = dict(slope_per_K=c[0], R2=1 - r.var() / pts[:, 1].var(), max_resid=float(np.abs(r).max()))
    res['seconds'] = time.time() - t0
    return res


if __name__ == '__main__':
    outp = sys.argv[1]; names = sys.argv[2:] or list(RX)
    allres = {}
    for n in names:
        r = run(n); allres[n] = r
        f = r['fresh_at_Pref']; a = r['at_t50']; L = r['loss_location']; ce = r['catalyst_equivalence']; e = r['at_end']
        print(f"\n== {n}: {r['reaction']}  p {r['p_bar']} bar  Tin {r['Tin_C']:.0f} C  design {r['design']}  ({r['seconds']:.0f} s)")
        for k, v in f.items():
            print(f"   fresh {k:10s} X {100*v['X']:6.2f} %  ATE {v['ATE']:6.2f} K  Tout {v['Tout_C']:6.1f}  Tmax {v['Tmax_C']:6.1f}  lambda {v['lam']}")
        print(f"   fresh power saving at equal output {r['fresh_power_saving_pct']:.2f} %")
        print(f"   at t50: extra power U {a['extra_U_pct']:.2f} %  design {a['extra_D_pct']:.2f} %  saving {a['saving_pct']:.2f} %  ATE U {a['ATE_U']:.1f} D {a['ATE_D']:.2f}  s_out U {a['sout_U']:.2f} D {a['sout_D']:.2f}")
        print(f"   at {e['t_over_t50']:.0f} t50: extra U {e['extra_U_pct']:.2f} %  design {e['extra_D_pct']:.2f} %  saving {e['saving_pct']:.2f} %  ATE U {e['ATE_U']:.1f} D {e['ATE_D']:.2f}")
        print(f"   uneven vs uniform loss (uniform heating): {L['extra_uneven_pct']:.2f} vs {L['extra_uniform_pct']:.2f} %  ratio {L['ratio']:.2f}  (s in {L['s_in']:.2f} out {L['s_out']:.2f}); same uniform loss under design {L['extra_uniform_under_design_pct']:.3f} %")
        print(f"   catalyst: uniform needs x {ce['uniform_needs_x_within_005pp']}  design matches uniform with x {ce['design_matches_uniform_with_x']:.3f}")
        print(f"   energy identity: slope {100*r['energy_identity']['slope_per_K']:.4f} pp/K  R2 {r['energy_identity']['R2']:.4f}")
    json.dump(allres, open(outp, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
