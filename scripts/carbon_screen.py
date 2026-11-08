"""Thermodynamic carbon screen of saved states (Supplementary Note 5): methane cracking to graphite.

a_C = p_CH4 / (Kf(T) p_H2^2), with Kf of C(gr) + 2 H2 = CH4 from NIST-JANAF (log10 Kf interpolated
linearly in 1/T). Gas: bulk gas of each axial cell (mean of its two edges); catalyst temperature: the
hottest pellet shell of each cell and ring. The margin is the temperature at which a_C = 1 for the cell
gas, minus the catalyst temperature (positive: no carbon affinity).

  python scripts/carbon_screen.py runs/U/states/node00 runs/U/states/node11
  python scripts/carbon_screen.py --equilibrated-gas      # feed equilibrated at 500-900 C, 6 and 21 bar
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # run from a checkout without installing the package
    sys.path.insert(0, str(ROOT))

import argparse

import numpy as np
from scipy.optimize import brentq

import esmr

TJ = np.array([700., 800., 900., 1000., 1100., 1200.]); LKF = np.array([0.943, 0.138, -0.500, -1.018, -1.447, -1.807])


def log_kf(T):
    return np.interp(1.0 / T, (1.0 / TJ)[::-1], LKF[::-1])


def t_carbon(r):
    """Temperature at which a_C = 1 for p_CH4/p_H2^2 = r."""
    Tg = np.linspace(700., 1200., 50001); lk = log_kf(Tg)
    return np.interp(-np.log10(r), -lk, Tg)


def screen(folder):
    st = esmr.load_state(folder); e = st.steady.last; nx = len(e.F_edges) - 1
    F = np.array(e.F_edges); P = np.array(e.P_edges_Pa) / 1e5
    y = F / F.sum(1, keepdims=True); pp = y * P[:, None]                 # bar: CH4 H2O H2 CO CO2
    pc = 0.5 * (pp[:-1] + pp[1:]); X = 1 - F[:, 0] / F[0, 0]
    Tcat = st.environment['T_K'].max(axis=2); health = st.environment['health']
    r = pc[:, 0] / pc[:, 2] ** 2
    aC = r[:, None] / 10 ** log_kf(Tcat)
    margin = t_carbon(r)[:, None] - Tcat
    dz = st.conditions['length_m'] / nx; z = (np.arange(nx) + 0.5) * dz
    bad = (aC > 1).any(axis=1)
    L = z[bad].max() + dz / 2 if bad.any() else 0.0
    print(f"\n{folder}: X_out {st.summary['X']:.2f} %, P {st.summary['P_W']:.0f} W, bed-averaged Ni {st.summary['mean_Ni_pct']:.1f} %")
    print(f'  carbon affinity (a_C > 1 in some ring) up to z = {L:.3f} m; max a_C {aC.max():.1f}; min margin {margin.min():.1f} K')
    print('  cell  z      X_cell  pH2    pCH4   Tcat_max_C  aC_max  margin_min_K  Ni_cell')
    for i in sorted(set(list(range(min(12, nx))) + [k for k in (15, 20, 27, nx - 1) if k < nx])):
        print(f'  {i:3d} {z[i]:.3f} {0.5 * (X[i] + X[i + 1]) * 100:7.2f} {pc[i, 2]:.3f} {pc[i, 0]:.3f} {Tcat[i].max() - 273.15:9.1f} '
              f'{aC[i].max():8.2f} {margin[i].min():10.1f} {health[i].mean():.3f}')


def equilibrated_gas():
    """a_C of the feed (per CH4: H2O 2.96, H2 0.04) equilibrated for reforming and shift with the model's K1, K2."""
    Fi = np.array([1.0, 2.96, 0.04, 0.0, 0.0])
    for P in (6.0, 21.0):
        out = []
        for TC in (500, 550, 600, 650, 700, 800, 900):
            T = TC + 273.15; k1 = np.exp(30.114 - 26830.0 / T); k2 = np.exp(4400.0 / T - 4.036)
            comp = lambda a, b: (lambda F: F / F.sum() * P)(Fi + np.array([-a, -a - b, 3 * a + b, a - b, b]))

            def b_of(a):
                lo, hi = -min(Fi[4], Fi[2] + 3 * a), min(Fi[3] + a, Fi[1] - a); e = 1e-12 * (hi - lo)
                return brentq(lambda b: (lambda q: np.log(q[4] * q[2] / (q[3] * q[1])) - np.log(k2))(comp(a, b)), lo + e, hi - e, xtol=1e-15)
            a = brentq(lambda a: (lambda q: np.log(q[3] * q[2] ** 3 / (q[0] * q[1])) - np.log(k1))(comp(a, b_of(a))), 1e-9, 1 - 1e-12, xtol=1e-15)
            q = comp(a, b_of(a)); aC = q[0] / q[2] ** 2 / 10 ** log_kf(T)
            out.append(f'{TC} C: X_eq {100 * a:5.1f} %, a_C {aC:.2f}')
        print(f'P {P:.0f} bar:', '; '.join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('states', nargs='*'); ap.add_argument('--equilibrated-gas', action='store_true')
    a = ap.parse_args()
    if a.equilibrated_gas:
        equilibrated_gas()
    for f in a.states:
        screen(f)


if __name__ == '__main__':
    main()
