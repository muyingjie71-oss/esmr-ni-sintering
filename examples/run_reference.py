"""Reference check: the first two nodes of the uniformly heated fixed-power history (fresh bed and 0.87 h).

    python examples/run_reference.py            # about 3 minutes on one CPU core

Prints each result next to the stored reference (reference_two_nodes.json, Linux, x86-64 CPU, double
precision). On the reference platform the differences are zero; other CPUs and libraries give
differences of order 1e-12 % in conversion and 1e-10 K in temperature.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import esmr  # noqa: E402
from esmr.histories import history_fixed_power, make_model  # noqa: E402

KEYS = ('X', 'P_W', 'particle_Tmax_C', 'particle_Tmean_C', 'wall_max_C', 'mean_Ni_pct')


def main():
    t0 = time.time()
    ref = json.loads((Path(__file__).with_name('reference_two_nodes.json')).read_text())
    nodes = history_fixed_power(make_model(esmr.BASE_POWER_W, esmr.UNIFORM), esmr.TIMES_H[:2])
    worst = 0.
    for state, r in zip(nodes, ref):
        s = state.summary
        print(f"\nnode at {r['t_h']:.4f} h")
        for k in KEYS:
            d = s[k] - r[k]; worst = max(worst, abs(d))
            print(f'  {k:18s} {s[k]:22.15f}   reference {r[k]:22.15f}   difference {d:+.2e}')
        for k in ('zone_Ni_pct', 'zone_T_C'):
            d = max(abs(a - b) for a, b in zip(s[k], r[k])); worst = max(worst, d)
            print(f'  {k:18s} largest difference over the ten zones {d:.2e}')
        print(f"  all shells within 483-682 C: {s['in_range']} (reference {r['in_range']})")
    print(f'\nlargest difference {worst:.2e}; {time.time() - t0:.0f} s')
    return 0 if worst < 1e-6 else 1


if __name__ == '__main__':
    raise SystemExit(main())
