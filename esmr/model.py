"""Operating point of the tube, steady states with a frozen Ni field, and the Ni-field update.

    model = Model(P_W=3523.30386509845, q=[0.1] * 10)
    fresh = model.solve(np.ones((40, 6)), seed=load_state(INITIAL_STATE))
    s_next = model.advance(fresh, dt_h=0.8712246180056071)
    aged = model.solve(s_next, seed=fresh, model_time_h=0.8712246180056071)

A steady state needs a converged state as initial guess (`seed`): its solid
and wall temperatures and the unknowns of every pellet. When the seed was
solved at another inlet temperature or power, its temperatures are rescaled
about the inlet temperature by sqrt(P / P_seed) before the solve.
"""
from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from . import geometry as geo
from . import properties as pr
from . import sintering as sn
from .pellet_reference import FilmParameters, PoreParameters
from .reactor import Evaluation, InvalidPhysics, Reactor, SteadyResult

DATA = Path(__file__).resolve().parent / 'data'
INITIAL_STATE = DATA / 'initial_state'                     # fresh bed, uniform heating, baseline power, main grid
GRIDS = {'main': (40, 6, 12), 'coarse': (10, 2, 4)}       # axial cells, radial rings, pellet shells
TEMPERATURE_DATA_RANGE_C = (483., 682.)                    # temperature range of the sintering data
ACTIVE_PELLET_FRACTION = .25
PELLET_TOLERANCE = 1e-9
PORE = dict(rho_cat_kg_per_m3_pellet=1860.5974372052015, porosity=0.528)
VARIANT_KEYS = {'sintering': {'fit', 'atmosphere_exponent', 'activity_exponent', 'activity_reference', 'T_switch_C'},
                'transport': {'kr', 'hw', 'hgs'}}


def settings(grid='main', precision='main'):
    """Convergence settings of the steady solve.

    cell_W, global_W: limits on the largest cell heat residual and on the energy balance of the tube, W.
    axial_rtol_flow, axial_atol_T_K: error tolerances of the gas march.
    """
    if grid == 'coarse':
        if precision != 'main':
            raise ValueError('the coarse grid has one precision setting')
        return dict(cell_W=.15, global_W=.5, axial_rtol_flow=.001, axial_atol_T_K=.05, max_iter=24)
    if precision not in ('main', 'strict', 'finer'):
        raise ValueError('precision must be main, strict or finer')
    return dict(cell_W=.005 if precision == 'main' else .001,
                global_W=.02 if precision == 'main' else .004,
                axial_rtol_flow=.00005 if precision == 'finer' else .0005,
                axial_atol_T_K=.002 if precision == 'finer' else .02, max_iter=40)


def check_variant(variant):
    """Checked copy of a sensitivity variant: only 'label', 'sintering' and 'transport' entries, with finite non-negative numbers."""
    v = json.loads(json.dumps(variant or {}, sort_keys=True))
    bad = set(v) - {'label', 'sintering', 'transport'}
    for k in VARIANT_KEYS:
        bad |= set(v.get(k, {})) - VARIANT_KEYS[k]
    if bad:
        raise ValueError('unsupported variant field(s): ' + ', '.join(sorted(bad)))
    for x in list(v.get('transport', {}).values()) + [y for k, y in v.get('sintering', {}).items() if k != 'fit']:
        if not (isinstance(x, (int, float)) and math.isfinite(float(x)) and float(x) >= 0):
            raise ValueError('variant numbers must be finite and non-negative')
    return v


@dataclass
class State:
    """An accepted steady state with its Ni field and the pellet environment used for the next Ni update."""
    conditions: dict
    model_time_h: float
    health: np.ndarray            # retained Ni area s, (nx, nr)
    steady: SteadyResult
    pellets: dict                 # (iz, ir) -> pellet unknowns, the warm start of later solves
    environment: dict             # shell T_K, pH2O_bar, pH2_bar, pressure_Pa, weights, sintering rates
    summary: dict

    def save(self, folder):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        st = self.steady
        doc = dict(conditions=self.conditions, model_time_h=self.model_time_h, summary=self.summary,
                   health=np.asarray(self.health).tolist(),
                   thermal_state_K=np.asarray(st.state).tolist(), converged=bool(st.converged), iterations=int(st.iterations),
                   residual_history_W=[float(x) for x in st.residual_history_W],
                   evaluation={k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(st.last).items()})
        _write_json(folder / 'state.json', doc)
        np.savez(folder / 'pellets.npz', **{f'pellet_{z}_{r}': a for (z, r), a in self.pellets.items()})
        np.savez_compressed(folder / 'environment.npz', **self.environment)


def _write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False, default=_scalar) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def _scalar(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v))


def load_state(folder) -> State:
    """Read a state written by State.save."""
    folder = Path(folder)
    doc = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
    e = doc['evaluation']
    for name in ('residual_W', 'F_edges', 'P_edges_Pa', 'Tgas_edges_K', 'H_edges_W', 'q_bed_W', 'q_gas_W'):
        e[name] = np.asarray(e[name], float)
    steady = SteadyResult(np.asarray(doc['thermal_state_K']), Evaluation(**e), doc['converged'], doc['iterations'],
                          doc['residual_history_W'])
    with np.load(folder / 'pellets.npz') as f:
        pellets = {}
        for key in f:
            _, z, r = key.split('_'); pellets[int(z), int(r)] = f[key].copy()
    environment = {}
    if (folder / 'environment.npz').exists():
        with np.load(folder / 'environment.npz', allow_pickle=False) as f:
            environment = {k: f[k].copy() for k in f.files}
    return State(doc['conditions'], float(doc['model_time_h']), np.asarray(doc['health'], np.float64), steady, pellets,
                 environment, doc['summary'])


def is_complete(folder):
    """True when the folder holds a complete saved state."""
    return all((Path(folder) / n).exists() for n in ('state.json', 'pellets.npz', 'environment.npz'))


class Model:
    """The tube at one operating point and heating profile.

    P_W: total electric power; q: shares of the ten heating zones; feed_factor scales the
    reference inlet flow (0.5 is the baseline, 2.40 Nm3/h CH4); SC: steam-to-carbon ratio,
    set through the H2O flow at fixed CH4 flow; variant: changes for the sensitivity tests,
    {'sintering': {...SinteringLaw fields}, 'transport': {'kr': .., 'hw': .., 'hgs': ..}}.
    """

    def __init__(self, P_W, q=None, Tin_C=550., Pin_bar=6., feed_factor=.5, SC=2.96, length_m=1., grid='main',
                 precision='main', variant=None, device='cpu', verbose=True):
        numeric = settings(grid, precision)
        nx, nr, shells = GRIDS[grid]
        q = np.asarray([.1] * 10 if q is None else q, dtype=np.float64)
        if q.shape != (10,) or not np.isfinite(q).all() or (q < 0).any() or abs(float(q.sum()) - 1.) > 1e-12:
            raise ValueError('ten non-negative zone shares summing to one required')
        values = (P_W, Tin_C, Pin_bar, feed_factor, SC, length_m)
        if not all(math.isfinite(float(v)) for v in values) or min(P_W, Pin_bar, feed_factor, SC) <= 0:
            raise ValueError('invalid operating conditions')
        if Tin_C + 273.15 < 310 or Pin_bar < 1.:
            raise ValueError('operating point outside the numerical range of the model')
        if not 0.5 <= length_m <= 3.:
            raise ValueError('tube length must lie between 0.5 and 3 m')
        self.variant = check_variant(variant)
        self.grid, self.precision, self.numeric = grid, precision, numeric
        self.nx, self.nr, self.shells = nx, nr, shells
        base = geo.tube(length_m=float(length_m), ri_m=.015, nx=nx, nr=nr, Tin_K=float(Tin_C + 273.15), Pin_Pa=float(Pin_bar * 1e5))
        f0 = np.asarray(base.F0, float) * float(feed_factor)
        if float(SC) != float(np.asarray(base.F0, float)[1] / np.asarray(base.F0, float)[0]):
            f0[1] = float(SC) * f0[0]
        self.g = geo.with_inlet(base, f0.tolist())
        self.heating = geo.Heating(float(P_W), tuple(q.tolist()))
        self.law = sn.SinteringLaw(**self.variant.get('sintering', {}))
        mult = self.variant.get('transport', {})
        coeff_key = tuple(float(mult.get(k, 1.0)) for k in ('kr', 'hw', 'hgs'))
        coeffs = pr.scaled_thermal_coeffs(*coeff_key)
        self.reactor = Reactor(self.g, self.heating, PoreParameters(**PORE), FilmParameters(), shells, PELLET_TOLERANCE,
                               numeric['axial_rtol_flow'], numeric['axial_atol_T_K'], ACTIVE_PELLET_FRACTION,
                               coeffs, coeff_key, device=device, verbose=verbose)
        self.reactor.options = dict(max_iter=numeric['max_iter'], tol_W=numeric['cell_W'],
                                    max_global_energy_W=numeric['global_W'], max_step_K=30.)
        self.conditions = dict(P_W=float(P_W), q=q.tolist(), Tin_C=float(Tin_C), Tin_K=self.g.Tin_K, Pin_bar=float(Pin_bar),
                               Pin_Pa=self.g.Pin_Pa, feed_factor=float(feed_factor), SC=float(SC), length_m=float(length_m),
                               F0_mol_s=self.g.F0.tolist(), grid=[nx, nr, shells], precision=precision, variant=self.variant)

    # ------------------------------------------------------------ helpers
    def activity(self, s):
        s = sn.validate_area(s)
        if s.shape != (self.nx, self.nr):
            raise ValueError('Ni field shape differs from the grid')
        return sn.activity(s, self.law)

    def mean_health(self, s):
        g = self.g; s = sn.validate_area(s)
        return float(np.sum(s * g.annulus_areas[None, :]) / (self.nx * g.area))

    def _seed(self, seed: State, transform):
        """Initial guess from a converged state: thermal state and all pellet unknowns."""
        r = self.reactor
        if not seed.steady.converged:
            raise ValueError('the initial guess must be a converged state')
        if self.grid == 'coarse':
            return self._seed_interpolated(seed)
        c = seed.conditions
        if list(c['grid']) != [self.nx, self.nr, self.shells]:
            raise ValueError('seed grid differs from the model grid')
        r.cache = {}; r.recent_by_radius = {}
        for (z, ir), x in seed.pellets.items():
            r.cache[z, ir] = x.copy(); r.recent_by_radius[ir] = x.copy()
        initial = np.asarray(seed.steady.state, np.float64).copy()
        if initial.shape != (self.nx * (self.nr + 1),) or not np.isfinite(initial).all():
            raise ValueError('invalid thermal initial guess')
        Tin_K = self.g.Tin_K; P_W = self.heating.total_W
        old_tin = float(c['Tin_K']); old_power = c['P_W']
        explicit = transform is not None
        transform = deepcopy(transform) if transform is not None else {}
        changed = old_tin != Tin_K or (old_power is not None and float(old_power) != P_W)
        scale = float(transform.get('scale', math.sqrt(P_W / float(old_power)) if changed and old_power is not None else 1.))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('positive finite seed scaling required')
        apply = explicit or changed
        if apply:
            initial = Tin_K + scale * (initial - old_tin)
        n = self.shells
        required = {(z, ir) for z in range(self.nx) for ir in range(self.nr)}
        if set(r.cache) != required:
            raise ValueError('the seed must contain every pellet')
        for key in sorted(required):
            x = np.asarray(r.cache[key], np.float64).copy()
            if x.shape != (7 * n + 6,) or not np.isfinite(x).all():
                raise ValueError('malformed pellet seed')
            if apply:
                internal = x[:7 * n].reshape(n, 7)
                internal[:, 6] = (Tin_K + scale * (internal[:, 6] * 1000 - old_tin)) / 1000
                x[-1] = (Tin_K + scale * (x[-1] * 1000 - old_tin)) / 1000
            local_T = np.r_[x[:7 * n].reshape(n, 7)[:, 6], x[-1]] * 1000
            if not np.isfinite(x).all() or local_T.min() < 310 or local_T.max() > 1750:
                raise ValueError('rescaled pellet guess outside the numerical range')
            r.cache[key] = x
        r.recent_by_radius = {ir: r.cache[self.nx - 1, ir].copy() for ir in range(self.nr)}
        if not np.isfinite(initial).all() or initial.min() < 310 or initial.max() > 1750:
            raise ValueError('rescaled thermal guess outside the numerical range')
        return initial

    def _seed_interpolated(self, seed: State):
        """Coarse grid: interpolate the seed temperatures and pellet profiles onto the model grid."""
        r = self.reactor; c = seed.conditions
        og = list(c['grid']); ng = [self.nx, self.nr, self.shells]
        go = geo.tube(length_m=c['length_m'], ri_m=.015, nx=og[0], nr=og[1], Tin_K=c['Tin_K'], Pin_Pa=c['Pin_Pa'])
        gn = self.g
        state = np.asarray(seed.steady.state, float); temp = state[:og[0] * og[1]].reshape(og[:2]); wall = state[og[0] * og[1]:]
        zo = (np.arange(og[0]) + .5) * go.dz; zn = (np.arange(ng[0]) + .5) * gn.dz
        radial = np.array([np.interp(gn.radius_centers, go.radius_centers, row) for row in temp])
        newtemp = np.array([np.interp(zn, zo, radial[:, i]) for i in range(ng[1])]).T
        initial = np.r_[newtemp.ravel(), np.interp(zn, zo, wall)]

        def centers(n):
            u = np.linspace(0, 1, n + 1); edges = 2 * u - u * u
            return ((edges[:-1] ** 3 + edges[1:] ** 3) / 2) ** (1 / 3)
        r.cache = {}
        for z in range(ng[0]):
            iz = int(np.argmin(abs(zo - zn[z])))
            for ir_new in range(ng[1]):
                ir = int(np.argmin(abs(go.radius_centers - gn.radius_centers[ir_new])))
                x = np.asarray(seed.pellets[iz, ir], float); oldcore = x[:7 * og[2]].reshape(og[2], 7)
                core = np.column_stack([np.interp(centers(ng[2]), centers(og[2]), oldcore[:, i]) for i in range(7)])
                xx = np.r_[core.ravel(), x[-6:]]
                if xx.shape != (7 * ng[2] + 6,) or not np.isfinite(xx).all():
                    raise ValueError('malformed interpolated pellet seed')
                r.cache[z, ir_new] = xx
        r.recent_by_radius = {ir: r.cache[ng[0] - 1, ir].copy() for ir in range(ng[1])}
        if initial.shape != (ng[0] * (ng[1] + 1),) or not (initial.min() > 310 and initial.max() < 1750):
            raise ValueError('interpolated thermal guess outside the numerical range')
        return initial

    # ------------------------------------------------------------ solve and age
    def solve(self, health, seed: State, model_time_h=0., seed_transform=None, folder=None) -> State:
        """Steady state with the Ni field `health` held fixed; saved to `folder` if given."""
        if not math.isfinite(float(model_time_h)) or model_time_h < 0:
            raise ValueError('non-negative finite model time required')
        h = np.asarray(health, np.float64).copy()
        if h.shape != (self.nx, self.nr) or not np.isfinite(h).all() or (h <= 0).any() or (h > 1).any():
            raise ValueError('invalid Ni field')
        r = self.reactor
        backup = (deepcopy(r.activity), deepcopy(r.cache), deepcopy(r.recent_by_radius))
        try:
            initial = self._seed(seed, seed_transform)
            r.activity = self.activity(h)
            steady = r.solve(initial)
            r.check_steady(steady)
            env = r.environment(steady)
        except BaseException:
            r.activity, r.cache, r.recent_by_radius = backup
            raise
        state = self._accept(h, steady, env, float(model_time_h))
        if folder is not None:
            state.save(folder)
        return state

    def _accept(self, h, steady, env, model_time_h):
        g = self.g; nx, nr = self.nx, self.nr
        e = steady.last
        if (not steady.converged or float(np.max(np.abs(e.residual_W))) >= self.numeric['cell_W']
                or abs(e.total_energy_residual_W) >= self.numeric['global_W'] or e.atoms_residual_mol_s > 1e-8):
            raise InvalidPhysics('steady state fails the acceptance limits')
        self.reactor.check_steady(steady)
        t = np.asarray(env['T_K'], np.float64)
        steam = np.asarray(env['pH2O_bar'], np.float64)
        hydrogen = np.asarray(env['pH2_bar'], np.float64)
        sw = np.asarray(env['shell_weights'], np.float64)
        rw = np.asarray(g.annulus_areas / g.area, np.float64)
        rate = sn.rate_x(self.law, t, steam, hydrogen)
        loss = rate * h[:, :, None] ** 8 / 7.            # -ds/dt of each shell, per model hour
        if t.shape != (nx, nr, self.shells) or not np.isfinite(rate).all() or not np.isfinite(t).all():
            raise ValueError('invalid pellet environment')
        environment = dict(T_K=t, pH2O_bar=steam, pH2_bar=hydrogen, pressure_Pa=env['pressure_Pa'], shell_weights=sw,
                           annular_weights=rw, health=h, rate_x_per_h=rate, Ni_loss_rate_per_h=loss)

        def zone(values):
            return np.sum(np.sum(values * sw[None, None, :], axis=2) * rw[None, :], axis=1).reshape(10, nx // 10).mean(axis=1)
        celsius = t - 273.15
        lo, hi = TEMPERATURE_DATA_RANGE_C
        F = e.F_edges
        axT = (np.sum(t * sw[None, None, :], axis=2) * rw[None, :]).sum(1) - 273.15
        zone_Ni = 100 * (h @ rw).reshape(10, nx // 10).mean(1)
        conversion = float(1 - e.F_edges[-1, 0] / e.F_edges[0, 0])
        summary = dict(
            X=float(conversion * 100.), P_W=float(self.heating.total_W), net_H2_mol_s=float(F[-1, 2] - F[0, 2]),
            particle_Tmax_C=float(t.max() - 273.15), particle_Tmin_C=float(t.min() - 273.15),
            particle_Tmean_C=float(axT.mean()), wall_max_C=float(float(steady.state[nx * nr:].max()) - 273.15),
            outlet_gas_C=float(e.Tgas_edges_K[-1] - 273.15), outlet_pressure_bar=float(e.P_edges_Pa[-1] / 1e5),
            in_range=bool(((celsius >= lo) & (celsius <= hi)).all()),
            temperature_min_C=float(celsius.min()), temperature_max_C=float(celsius.max()),
            mean_Ni_pct=float(zone_Ni.mean()), zone_Ni_pct=zone_Ni.tolist(),
            zone_T_C=axT.reshape(10, nx // 10).mean(1).tolist(),
            zone_shell_T_C=(zone(t) - 273.15).tolist(), zone_Ni_loss_pct_per_h=(100 * zone(loss)).tolist(),
            bed_mean_Ni=self.mean_health(h), max_cell_residual_W=float(np.max(np.abs(e.residual_W))),
            energy_residual_W=float(e.total_energy_residual_W), iterations=int(steady.iterations))
        pellets = {k: v.copy() for k, v in self.reactor.cache.items()}
        return State(deepcopy(self.conditions), model_time_h, h.copy(), deepcopy(steady), pellets, environment, summary)

    def advance(self, state: State, dt_h) -> np.ndarray:
        """Ni field after dt_h model hours, with the shell rates of `state` held fixed over the step.

        Each shell is updated first; the cell value is then the volume average over its shells.
        """
        if not math.isfinite(float(dt_h)) or dt_h <= 0:
            raise ValueError('positive finite time step required')
        env = state.environment
        old = np.asarray(env['health']).copy()[:, :, None]
        rate = sn.rate_x(self.law, env['T_K'], env['pH2O_bar'], env['pH2_bar'])
        local = sn.advance_area(old, float(dt_h), rate)
        return np.sum(local * env['shell_weights'][None, None, :], axis=2)
