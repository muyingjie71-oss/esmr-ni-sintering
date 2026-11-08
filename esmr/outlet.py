"""Outlet approach to equilibrium and its decay in an unheated tail (Figs 2, 3e and 5c).

The approach to equilibrium is the outlet gas temperature minus the temperature at which the outlet
composition would be at equilibrium for reaction 1, with the equilibrium constants of the kinetic model
(K1 = exp(-26830/T + 30.114) bar^2).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, curve_fit


def K1(T):
    """Equilibrium constant of reaction 1, CH4 + H2O = CO + 3 H2, in bar^2 (T in K), as in the kinetic model."""
    return np.exp(-26830.0 / T + 30.114)


def approach_profile(state):
    """Approach to equilibrium (K) at every cell edge along the tube."""
    e = state.steady.last
    F = np.asarray(e.F_edges); T = np.asarray(e.Tgas_edges_K); P = np.asarray(e.P_edges_Pa) / 1e5
    z = np.linspace(0, state.conditions['length_m'], len(T)); ate = np.full(len(T), np.nan)
    for i in range(1, len(T)):
        p = F[i] / F[i].sum() * P[i]; Q1 = p[3] * p[2] ** 3 / (p[0] * p[1])
        ate[i] = T[i] - brentq(lambda TT: np.log(K1(TT)) - np.log(Q1), 500., 1400.)
    return z, T, ate


def metrics(state):
    """Power, conversion, outlet gas temperature, outlet approach to equilibrium, hottest wall point, hottest pellet shell and mean Ni area of a state."""
    z, T, ate = approach_profile(state)
    s = state.summary
    return dict(P_W=s['P_W'], X=s['X'], outlet_C=float(T[-1] - 273.15), approach_K=float(ate[-1]),
                wall_max_C=s['wall_max_C'], particle_Tmax_C=s['particle_Tmax_C'], bed_mean_Ni_pct=s['mean_Ni_pct'])


def tail_fit(state, z0):
    """Fit approach(z) = c + A exp(-(z - z0)/lambda) over the unheated tail z >= z0."""
    z, T, ate = approach_profile(state); m = z >= z0 - 1e-9
    f = lambda zz, c, A, lam: c + A * np.exp(-(zz - z0) / lam)
    p, _ = curve_fit(f, z[m], ate[m], p0=(-1.5, ate[m][0] + 1.5, .06), maxfev=20000)
    res = ate[m] - f(z[m], *p)
    return dict(c_K=float(p[0]), A_K=float(p[1]), lambda_m=float(p[2]), rmse_K=float(np.sqrt(np.mean(res ** 2))),
                approach_entry_K=float(ate[m][0]), approach_outlet_K=float(ate[-1]))
