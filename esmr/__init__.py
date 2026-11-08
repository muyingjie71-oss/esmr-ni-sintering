"""Electrified steam methane reformer tube with Ni sintering.

Steady reactor model (pellets, gas, bed and wall) and the ageing of the
catalyst as a sequence of steady states. Computations run in double
precision on the CPU by default; set ESMR_DEVICE=gpu before the first import
to place the pellet solves on a GPU.
"""
import os

os.environ.setdefault('JAX_PLATFORM_NAME', os.environ.get('ESMR_DEVICE', 'cpu'))
os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax  # noqa: E402

jax.config.update('jax_enable_x64', True)

from .model import GRIDS, INITIAL_STATE, Model, State, is_complete, load_state, settings  # noqa: E402,F401
from .sintering import SinteringLaw  # noqa: E402,F401

BASE_POWER_W = 3523.30386509845          # fixed power of the baseline, W
UNIFORM = [0.1] * 10
# Time grid of the ageing histories, model hours; the last node is t_ref, at which the uniformly
# heated bed at the baseline power has lost half of its bed-averaged Ni area.
TIMES_H = [0.0, 0.8712246180056071, 2.613673854016821, 8.712246180056072, 21.78061545014018,
           43.56123090028036, 87.12246180056071, 174.24492360112143, 304.92861630196245,
           435.61230900280356, 653.4184635042053, 871.2246180056071]
