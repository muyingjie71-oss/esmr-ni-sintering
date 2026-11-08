"""Outlet metrics of saved states: conversion, outlet temperature, approach to equilibrium, and the decay
of the approach in an unheated tail (Figs 2 and 3e).

The approach to equilibrium is the outlet gas temperature minus the temperature at which the outlet
composition would be at equilibrium for reaction 1, with the equilibrium constants of the kinetic model
(K1 = exp(-26830/T + 30.114) bar^2).

  python scripts/outlet_metrics.py runs/U/states/node00 runs/U/states/node11
  python scripts/outlet_metrics.py --tail-start 0.7 runs/S/states/node00
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # run from a checkout without installing the package
    sys.path.insert(0, str(ROOT))

import argparse

import esmr
from esmr.outlet import metrics, tail_fit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('states', nargs='+'); ap.add_argument('--tail-start', type=float, help='start of the unheated tail, m')
    a = ap.parse_args()
    for folder in a.states:
        st = esmr.load_state(folder)
        m = metrics(st)
        if a.tail_start is not None:
            m.update(tail_fit(st, a.tail_start))
        print(folder, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})


if __name__ == '__main__':
    main()
