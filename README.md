# esmr-ni-sintering

Reactor model, ageing and optimization scripts for

> C. Liu, Y. Mu, H. Wang. *Stopping the heating before the outlet makes electrified reformers tolerant of catalyst ageing.*

The model is a steady two-dimensional electrified steam methane reformer tube (1 m, 30 mm inner
diameter, ten wall heating zones) with resolved catalytic pellets, coupled to Ni sintering. Ageing is
computed as a sequence of steady states: at each node of a time grid the reactor is solved with the
current Ni field, and the Ni field is advanced to the next node with the sintering rates of that state.
Equations and parameter values are given in Supplementary Note 1 and Extended Data Table 1 of the paper.

## Install

Python 3.10 or newer.

    pip install -e .            # or: pip install -r requirements.txt, then run from this folder

Computations run in double precision on the CPU with JAX. Tested with Python 3.13, NumPy 2.5, SciPy 1.18
and JAX 0.12 on Linux (x86-64). The runs behind the paper solved the pellet problems on a GPU, which is selected by
setting `ESMR_DEVICE=gpu` before `esmr` is first imported.

## Check the installation

    python examples/run_reference.py

solves the first two nodes of the uniformly heated history at the baseline power (fresh bed and 0.87 h,
about 3 minutes) and prints the differences from the stored reference. On the reference platform they are
zero; on other CPUs and library versions they are of order 1e-12 % in conversion.

## Use

```python
import numpy as np, esmr
from esmr.histories import make_model, history_fixed_power, solve_frozen

model = make_model(esmr.BASE_POWER_W, esmr.UNIFORM)        # 3.52 kW, 10 % of the power in each zone
nodes = history_fixed_power(model, esmr.TIMES_H)            # 12 nodes, fresh bed to 871 h
aged = nodes[-1]
print(aged.summary['X'], aged.summary['zone_Ni_pct'])       # CH4 conversion %, retained Ni area per zone %
even = solve_frozen(model, np.full((40, 6), 0.5), aged)     # evenly aged bed, same power, no further ageing
```

A steady state needs a converged state as initial guess; `esmr/data/initial_state` holds the fresh bed
under uniform heating at the baseline power and is used when none is given.

## Terms used in the code

| term | meaning |
|---|---|
| zone shares `q` | fractions of the electric power in the ten wall heating zones, Z1 at the inlet to Z10 at the outlet; they sum to 1. Uniform heating is 0.1 in every zone; the optimized profile of the paper is in `esmr/data/optimized_heating.json` |
| Ni field (`health`) | retained Ni surface area relative to the fresh catalyst, one value per axial cell and radial ring (1 = fresh) |
| fixed power | the total power stays at the baseline of 3.52 kW, and the CH4 conversion falls as the catalyst ages |
| constant production | at every node the power is set so that the CH4 conversion stays at that of the fresh bed |
| frozen Ni field | the reactor is solved with a given Ni field and no further ageing; used to compare beds |
| aged and evenly aged bed | the Ni field after 871 h of uniform heating at fixed power, and a bed with the same mean Ni area (0.5) in every cell |
| node | a time at which the reactor is solved. `esmr.TIMES_H` holds the 12 nodes from 0 to 871 h; at 871 h the uniformly heated bed has lost half of its Ni area |
| J | gain of a heating profile q: the Ni fields after 871 h under q and under uniform heating are both solved under uniform heating at 3.52 kW, and J is the difference in CH4 conversion |
| outlet approach to equilibrium | outlet gas temperature minus the temperature at which the outlet gas would be at equilibrium for steam reforming |
| main and coarse grid | 40 axial cells, 6 radial rings and 12 pellet shells for all results of the paper; 10, 2 and 4 to screen heating profiles |

Units: CH4 conversion `X` in %, differences of conversion in percentage points (pp), temperatures in °C
(names ending in `_C`) or K (`_K`), power in W (`_W`), time in model hours (`t_h`).

## How the code is organized

    esmr/            the model package
    esmr/data/       initial state, sintering fits, optimized profile, reference histories
    scripts/         one script per study of the paper (table below)
    examples/        installation check
    reduced_model/   reduced model of Fig. 5d–f, independent of esmr

The scripts in `scripts/` set up the studies of the paper and call the package `esmr`, which has four
layers:

1. `esmr/histories.py` runs ageing histories at fixed power or constant production, solves frozen Ni
   fields, finds the power that gives a target conversion and computes J.
2. `esmr/model.py` holds one operating point (`Model`: power, zone shares, feed). `Model.solve` finds the
   steady state for a given Ni field, and `Model.advance` ages the Ni field over one time step.
3. `esmr/reactor.py` solves the steady tube: the gas is marched from the inlet to the outlet, the pellets
   of every cell are solved on the way, and the bed and wall heat balances are driven to zero.
4. `esmr/pellet_jax.py` solves the pellet equations in batches. `esmr/pellet_reference.py` holds the same
   equations in NumPy and checks every accepted state.

To follow one steady state from the top, read `examples/run_reference.py`, then
`histories.history_fixed_power`, `Model.solve` and `Reactor.solve`.

| module | content |
|---|---|
| `esmr/histories.py` | fixed-power and constant-production histories, adaptive steps to a bed-averaged Ni area, power control, the objective J |
| `esmr/model.py` | an operating point (`Model`): steady state with a frozen Ni field, Ni update over a time step, saved states |
| `esmr/reactor.py` | gas march (adaptive explicit midpoint with error control), bed and wall heat balances, Broyden solve, acceptance checks |
| `esmr/pellet_jax.py` | the pellet equations of `pellet_reference.py` in JAX, solved in batches by Newton's method with exact coloured Jacobians |
| `esmr/pellet_reference.py` | NumPy finite-volume pellet: Dusty-Gas transport, Xu–Froment kinetics, total-enthalpy balance, Maxwell–Stefan gas film; every accepted state is checked against this residual |
| `esmr/sintering.py` | sintering law fitted to Sehested et al. (2004), catalyst No. 1, and its alternatives; activity of a Ni field |
| `esmr/properties.py` | NIST Shomate thermochemistry; bed heat-transfer correlations (Zehner–Bauer–Schlünder with radiation, Yagi–Kunii form evaluated by Zambon et al., Wakao); gas property and insulation-loss tables |
| `esmr/geometry.py` | tube and packed-bed geometry, inlet flows, power of each heating zone |
| `esmr/search.py` | surrogate-assisted trust-region search over the ten zone shares |
| `esmr/shares.py` | zone shares: normalization, heaters with fewer sections, moves of power between adjacent zones |
| `esmr/outlet.py` | outlet approach to equilibrium and its decay length in an unheated tail |
| `esmr/report.py` | heating profiles by name, result tables, run length and electricity of constant-production runs |
| `esmr/data/` | initial state, sintering fits, optimized profile, uniform-heating reference histories, starting points of the profile search |

## What a run writes

Each script writes into the folder given by `--out`: every accepted steady state in its own subfolder
(`state.json` with the settings, the Ni field and a summary, and `pellets.npz` and `environment.npz` with the
full solution), and the tables of the study (`history.csv`, `summary.json` or similar, described at the top
of each script). Running a script again with the same `--out` reuses the saved states. A saved state is read
with `esmr.load_state(folder)`. Its `summary` holds, among others, the CH4 conversion `X`, the power `P_W`,
the hottest pellet shell `particle_Tmax_C`, the hottest wall point `wall_max_C`, the outlet gas temperature
`outlet_gas_C` and the mean Ni area of each zone `zone_Ni_pct`.

## Results of the paper

Each script prints its usage with `-h`. Some studies start from the results of others: run the uniformly
heated history (`runs/U`) and the constant-production runs (`runs/cp`) first.

| result | command |
|---|---|
| ageing at fixed power, uniform and optimized heating (Figs 1–3) | `python scripts/run_history.py --profile uniform --uniform-test --out runs/U`, and the same with `--profile optimized --out runs/S` |
| history at every 5 % of Ni area (Fig. 1c) | `python scripts/run_history.py --profile uniform --area-steps --out runs/U_area` |
| aged against evenly aged bed, restoring power (Fig. 1) | `python scripts/run_frozen_fields.py compare --aged runs/U/states/node11 --out runs/compare` |
| reaction rates scaled by 0.5–16 (Figs 2 and 3) | `python scripts/run_frozen_fields.py activity --aged runs/U/states/node11 --out runs/activity` |
| all Ni of one zone removed from the fresh bed (Fig. 2a,b, Extended Data Fig. 2a,b) | `python scripts/run_frozen_fields.py switch-off --out runs/switch_off` |
| zone sensitivity of the conversion (Figs 2c and 3, Extended Data Fig. 2c) | `python scripts/run_frozen_fields.py zones --heating optimized --out runs/zones` |
| outlet approach to equilibrium and its decay in the unheated tail (Figs 2, 3e and 5c, Extended Data Fig. 4) | `python scripts/outlet_metrics.py --tail-start 0.7 runs/S/states/node00` (any saved state; the states of `runs/cp` are listed in its `history.csv`) |
| optimized heating profile (Fig. 3a, Supplementary Note 2) | `python scripts/run_profile_search.py search --out runs/search`, then `confirm` |
| highest conversion of the fresh bed (Extended Data Fig. 3) | `python scripts/run_fresh_optimization.py --out runs/fresh --workers 4` |
| constant production to 20,628 h, run length and electricity (Fig. 4) | `python scripts/run_constant_production.py --out runs/cp --workers 2` |
| aged beds at 55 combinations of CH4 feed and S/C (Fig. 5a,b) | `python scripts/run_operating_points.py --fixed-power runs/U --constant-production runs/cp --out runs/points --workers 4` |
| heaters with 2, 3 or 5 sections (Fig. 5c) | `python scripts/run_heater_sections.py --constant-production runs/cp --out runs/sections --workers 4` |
| profile re-optimized at every node (Extended Data Fig. 5) | `python scripts/run_reoptimization.py --constant-production runs/cp --out runs/reopt --workers 4` |
| sensitivity tests (Extended Data Fig. 6) | `python scripts/run_sensitivity.py --out runs/sensitivity` |
| industrial pressure and outlet temperature (Extended Data Table 3) | `python scripts/run_industrial.py --out runs/industrial` |
| carbon affinity of the inlet gas (Supplementary Note 5) | `python scripts/carbon_screen.py --equilibrated-gas runs/U/states/node00` |
| numerical checks (Supplementary Note 3) | `run_history.py --times half` or `double`, `--precision strict` or `finer`, `--grid coarse` |
| reduced model for five reactions (Fig. 5d–f, Extended Data Table 2) | `python reduced_model/study.py runs/GENERIC_RESULTS.json` |

## Run time

One steady state on the main grid takes one to two minutes on one CPU core; the first solve of a process
also compiles the JAX code (about 40 s). Typical times with one process per job: a 12-node history to
871 h 15–30 minutes, a constant-production history to 871 h 25–35 minutes, one sensitivity variant about
45 minutes. The coarse grid of the profile search is much faster. The searches and the studies of Figs 4
and 5 and Extended Data Figs 3 and 5 run several solves in parallel with `--workers`; for other jobs, run
several scripts side by side. Each process needs about 1.5 GB of memory. In the runs behind the paper (a
Linux compute node, several processes side by side), the search for the highest fresh conversion took about
1.6 h, constant production to 20,628 h about 1.7 h per heating, one of the 55 operating points about
35 minutes, one heater-section candidate 4–55 minutes and the re-optimized run about 4 hours.

## Reproducibility

The reactor is solved in double precision throughout. A state is accepted when the heat balance of every
cell closes within 0.005 W, the energy balance of the tube within 0.02 W and the element balances within
1e-8 mol s-1, and when every pellet passes the check against the NumPy reference residual. Results are
bitwise reproducible on one platform; across CPUs they agree to about 1e-12 in relative terms.

A solve in which a pellet fails that check is not accepted, and the searches treat it as outside the
limits. Whether a pellet passes can depend on the machine, so a search can take a slightly different path on
another machine. In the fresh-bed search of Extended Data Fig. 3, one of the 535 solves failed in the run
behind the paper and passes on the reference platform. With it, the search from the optimized profile takes
two different steps, then returns to the path of the paper and ends at the same profile after the same
number of steps.

## Citation

If you use this code, please cite the paper above (`CITATION.cff` holds the details). This version is archived at Zenodo, https://doi.org/10.5281/zenodo.24056436, and the solved states and ageing histories at https://doi.org/10.5281/zenodo.24056434.

## Licence

MIT, see `LICENSE`.
