"""Steady electrified reformer tube: bed, wall and gas coupled to the catalytic pellets.

Unknowns of the outer problem: one solid temperature per axial cell and
radial ring, and one wall temperature per axial cell (u, length nx*nr + nx).
For given u the gas is marched along the tube (plug flow) with an adaptive
explicit midpoint scheme; at every substep the six ring pellets are solved
at the local gas state. The bed and wall heat balances then give the
residual of u, which is driven to zero by a Broyden method started from a
finite-difference Jacobian of the reacting gas propagation.

The bed of each cell holds catalytic and inert pellets of equal size; a
share `active_fraction` of the particle surface is catalytic, the rest
exchanges only sensible heat with the gas.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as j
import numpy as np
from jax import lax
from jax.scipy.linalg import expm
from scipy import sparse
from scipy.optimize import brentq

from . import properties as pr
from .pellet_jax import JaxPellet, STOICH, broyden_update, outer_linear_step, R as R_PELLET
from .pellet_reference import GasBedState, PelletWithFilm
from .properties import ATOMS, MW, R, shomate_h


class InvalidPhysics(RuntimeError):
    """A state that cannot be computed or fails a check: non-physical gas, failed pellet or gas march, unclosed balance."""
    pass


def _in_enthalpy_jump(F, T, miss, tol_W=1e-6):
    """True when every gas enthalpy that the bisection could not match within tol_W lies in the jump of a Shomate fit
    at its switch temperature (properties.SHOMATE_JUMPS_J_MOL). The H2 fit jumps up by 0.18 J mol-1 at 1000 K, so a gas
    enthalpy can fall between the two segments; its temperature is then the switch temperature, which the bisection
    returns. Only a state that would otherwise be rejected takes this path."""
    bad = miss > tol_W
    near = np.abs(T[bad, None] - pr.SWITCH[None, :]) < 1e-6
    allowed = np.sum(F[bad] * np.abs(pr.SHOMATE_JUMPS_J_MOL)[None, :] * near, axis=1) + tol_W
    return bool(near.any(axis=1).all() and (miss[bad] <= allowed).all())


@dataclass
class GasState:
    """Gas at one axial position: molar flows of CH4, H2O, H2, CO and CO2, total enthalpy flow and pressure."""
    F: np.ndarray        # molar flows, mol s-1
    H_W: float           # total enthalpy flow sum(F_i h_i(T)), W
    P_Pa: float

    @property
    def vector(self):
        return np.r_[self.F, self.H_W, self.P_Pa]

    @classmethod
    def from_vector(cls, v):
        return cls(np.asarray(v[:5], float).copy(), float(v[5]), float(v[6]))

    @staticmethod
    def from_inlet(F, Tgas, P):
        F = np.asarray(F, float)
        return GasState(F, float(F @ shomate_h(Tgas)), float(P))

    def gas_temperature(self, bounds_K=(310., 1800.)):
        if np.any(self.F < -1e-13) or np.sum(self.F) <= 0 or self.P_Pa < 1.e5:
            raise InvalidPhysics('non-physical composition or pressure in the gas march')

        def r(T): return float(self.F @ shomate_h(T) - self.H_W)
        lo, hi = bounds_K
        if r(lo) * r(hi) > 0:
            raise InvalidPhysics('gas enthalpy implies a temperature outside the solver range')
        return float(brentq(r, lo, hi, xtol=1e-8))


@dataclass
class AxialResult:
    """One march of the gas through the tube: outlet gas, change of the molar flows and heat exchanged in each ring, steps and balance residuals."""
    outlet: GasState
    deltaF_by_ring: np.ndarray
    q_gas_by_ring_W: np.ndarray
    q_bed_by_ring_W: np.ndarray
    accepted: int
    rejected: int
    smallest_step_m: float
    max_step_m: float
    mass_residual_kg_s: float
    atoms_residual_mol_s: float
    energy_residual_W: float
    flow_residual_mol_s: float
    source_evaluations: int


@dataclass
class Evaluation:
    """One evaluation of the outer residual: heat residual of every cell, gas at the cell edges, heat flows and balance checks."""
    residual_W: np.ndarray
    F_edges: np.ndarray
    P_edges_Pa: np.ndarray
    Tgas_edges_K: np.ndarray
    H_edges_W: np.ndarray
    q_bed_W: np.ndarray
    q_gas_W: np.ndarray
    total_energy_residual_W: float
    gas_bed_flux_mismatch_W: float
    mass_residual_kg_s: float
    atoms_residual_mol_s: float
    accepted_substeps: int
    rejected_substeps: int
    max_local_gas_energy_residual_W: float = 0.
    max_local_atom_residual_mol_s: float = 0.
    max_local_flow_residual_mol_s: float = 0.
    global_closure_identity_error_W: float = 0.


@dataclass
class SteadyResult:
    """Solid and wall temperatures at the end of the outer solve, with the last evaluation and the iteration history."""
    state: np.ndarray
    last: Evaluation
    converged: bool
    iterations: int
    residual_history_W: list


def ergun_dPdz(g, F, T, P):
    """Pressure gradient of the bed, Ergun equation, Pa m-1."""
    y = np.asarray(F) / np.sum(F)
    rho = P * float(y @ np.asarray(pr.MW_TABLE)) / (R * T)
    if not np.isfinite(rho) or rho <= 0:
        raise InvalidPhysics('invalid gas density')
    vel = g.mass_flux_kg_m2s / rho; mu = float(pr.gas_viscosity(T)); ep = g.bed_void; d = g.dp_m
    return -(150 * mu * (1 - ep) ** 2 * vel / (d * d * ep ** 3) + 1.75 * (1 - ep) * rho * vel ** 2 / (d * ep ** 3))


# Compiled code is shared between reactors with identical static inputs; no physical state is shared.
_PELLET_CODE = {}
_THERMAL_CODE = {}
_TEMPERATURE_CODE = {}
_PROPAGATION_CODE = {}
_AXIAL_CODE = {}
_EXPM_BATCH = jax.jit(jax.vmap(lambda x: expm(x, max_squarings=32)))


class Reactor:
    """Steady tube for one geometry and heating (see the module docstring); solve() drives the cell heat balances to zero."""
    def __init__(self, g, heating, pore, film, n_shells, pellet_tolerance, rtol_flow, atol_T_K,
                 active_fraction=.25, thermal_coeffs=pr.thermal_coeffs, coeff_key=(1., 1., 1.),
                 device='cpu', chunk=32, verbose=True):
        self.g, self.heating, self.pore, self.film = g, heating, pore, film
        self.n_shells = int(n_shells); self.solver_tolerance = float(pellet_tolerance)
        self.rtol_flow = float(rtol_flow); self.atol_T_K = float(atol_T_K)
        self.phi = float(active_fraction); self.chunk = int(chunk); self.coeffs = thermal_coeffs
        self.verbose = verbose
        self.ns = g.nx * g.nr; self.n = self.ns + g.nx
        self.activity = None              # (nx, nr) relative catalyst activity
        self.cache = {}                   # (iz, ir) -> pellet unknowns, warm start of the next solve
        self.recent_by_radius = {}
        self.options = dict(max_iter=40, tol_W=.005, max_global_energy_W=.02, max_step_K=30.)
        self.reference = PelletWithFilm(pore, self.n_shells)
        code_key = (device, self.n_shells, repr(asdict(pore)))
        if code_key not in _PELLET_CODE:
            _PELLET_CODE[code_key] = JaxPellet(self.reference, np.array(pr.TG), np.array(pr.MU), device)
        self.pellet = _PELLET_CODE[code_key]
        self.device = self.pellet.device
        self.stats = dict(pellet_batches=0, pellets=0, cold_continuations=0, polishes=0)
        self.atoms = j.asarray([[1., 0., 0., 1., 1.], [4., 2., 2., 0., 0.], [0., 1., 0., 1., 2.]])
        geom_key = (device, repr(asdict(g)), tuple(coeff_key))
        if geom_key not in _THERMAL_CODE:
            _THERMAL_CODE[geom_key] = jax.jit(self._thermal_impl)
        if code_key not in _TEMPERATURE_CODE:
            _TEMPERATURE_CODE[code_key] = jax.jit(jax.vmap(self._temperature_one))
        propagation_key = (device, g.nx, g.nr)
        if propagation_key not in _PROPAGATION_CODE:
            _PROPAGATION_CODE[propagation_key] = jax.jit(self._propagate_impl)
        self._thermal = _THERMAL_CODE[geom_key]
        self._temperature = _TEMPERATURE_CODE[code_key]
        self._propagators = _EXPM_BATCH
        self._linear_propagation = _PROPAGATION_CODE[propagation_key]
        axial_key = (str(self.device), repr(asdict(g)), self.solver_tolerance, repr(asdict(pore)), repr(asdict(film)),
                     self.phi, tuple(coeff_key))
        self.axial = AxialMarch(self)
        if axial_key in _AXIAL_CODE:
            self.axial.integrate_compiled = _AXIAL_CODE[axial_key]
        else:
            _AXIAL_CODE[axial_key] = self.axial.integrate_compiled
        self.L = None; self.loss = None; self._thermal_state = None; self.last_outer_jacobian = None

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ------------------------------------------------------------ pellets
    def _temperature_one(self, state):
        F = state[:5]; H = state[5]

        def value(T): return j.dot(F, self.pellet.enthalpy(j.asarray([T]))[0]) - H
        lo, hi = lax.fori_loop(0, 48, lambda _, b: (
            j.where(value((b[0] + b[1]) / 2) < 0, (b[0] + b[1]) / 2, b[0]),
            j.where(value((b[0] + b[1]) / 2) < 0, b[1], (b[0] + b[1]) / 2)), (j.array(310.), j.array(1800.)))
        return (lo + hi) / 2

    def temperatures(self, states):
        """Gas temperatures of gas states [F (5), H_W, P_Pa], by bisection on the enthalpy."""
        a = np.asarray(states, float).reshape(-1, 7); T = np.array(self._temperature(a))
        h = np.array(self.pellet.enthalpy(j.asarray(T)))
        miss = np.abs(np.sum(a[:, :5] * h, axis=1) - a[:, 5])
        if (np.min(a[:, :5]) < -1e-13 or np.min(a[:, 6]) < 1e5 or
                (np.max(miss) > 1e-6 and not _in_enthalpy_jump(a[:, :5], T, miss))):
            raise ValueError('enthalpy inversion outside the material or temperature range')
        return T

    def backgrounds(self, gas_states, bed):
        gs = np.asarray(gas_states, float).reshape(-1, 7); bed = np.asarray(bed, float).reshape(-1)
        T = self.temperatures(gs); Y = gs[:, :5] / gs[:, :5].sum(axis=1, keepdims=True)
        coeff = jax.vmap(self.coeffs, in_axes=(0, None))(j.asarray((T + bed) / 2), self.g.mass_flux_kg_m2s)
        hf = np.array(coeff[1]); n = len(T)
        return np.column_stack((T, bed, gs[:, 6], Y, np.full(n, self.film.film_thickness_m), hf,
                                np.full(n, self.film.solid_contact_h_W_m2K)))

    def guesses(self, zs, rs, bg):
        X = []; cold = []
        for z, ir, b in zip(zs, rs, bg):
            x = self.cache.get((int(z), int(ir)), self.recent_by_radius.get(int(ir)))
            if x is None:
                state = GasBedState(b[0], b[1], b[2], b[3:8], b[8], b[9], b[10])
                x = self.reference.initial(state); cold.append(True)
            else:
                cold.append(False)
            X.append(np.asarray(x, float).copy())
        return np.asarray(X), np.asarray(cold)

    def _solve_batch(self, X, BG, A, cold=None):
        self.stats['pellet_batches'] += 1; self.stats['pellets'] += len(X)
        x, ok, norm, nit = map(np.array, self.pellet.solve(X, BG, A, tol=self.solver_tolerance, max_iter=42))
        bad = ~ok if cold is None else (~ok) | cold
        if np.any(bad):
            # continuation in activity from zero to the target, for pellets without a warm start
            self.stats['cold_continuations'] += int(bad.sum())
            xb = []
            for b in BG[bad]:
                state = GasBedState(b[0], b[1], b[2], b[3:8], b[8], b[9], b[10])
                xb.append(self.reference.initial(state))
            xb = np.array(xb); ab = A[bad]; bb = BG[bad]
            for strength in np.r_[0., np.geomspace(1e-6, 1., 10)]:
                xb, ob, nb, ib = map(np.asarray, self.pellet.solve(xb, bb, ab * strength, tol=self.solver_tolerance, max_iter=42))
                if not ob.all():
                    raise InvalidPhysics('pellet continuation failed')
            x[bad] = xb; ok[bad] = True
        comp = tuple(np.asarray(v) for v in self.pellet.outputs_batch(x, BG, A))
        mismatch = np.abs(comp[2] - comp[3]); limits = np.maximum(2e-5, 2e-8 * np.maximum(np.abs(comp[2]), np.abs(comp[3])))
        polish = (mismatch > limits) | (np.max(np.abs(comp[0]), axis=1) >= self.solver_tolerance)
        if np.any(polish):
            self.stats['polishes'] += int(polish.sum())
            xx, oo, nn, ii = map(np.asarray, self.pellet.solve(x[polish], BG[polish], A[polish],
                                                              tol=min(1e-11, self.solver_tolerance * .01), max_iter=42))
            x[polish] = xx; comp = tuple(np.asarray(v) for v in self.pellet.outputs_batch(x, BG, A))
        rr, N, qg, qb, T, P, Y, q, rate, pe = comp
        norm = np.max(np.abs(rr), axis=1)
        integrated = self.pellet.rho * A[:, None, None] * (rate @ np.array(STOICH).T)
        integrated = np.sum(integrated * np.array(self.pellet.vol)[None, :, None], axis=1) / float(self.pellet.area[-1])
        atoms = np.max(np.abs(N @ np.array(self.atoms).T), axis=1)
        good = (np.isfinite(rr).all(axis=1) & (norm < self.solver_tolerance) & (atoms < 1e-7) &
                (np.max(np.abs(N - integrated), axis=1) < 5e-7) & (np.abs(q[:, -1]) < 1e-2) & (np.abs(pe) <= 80) &
                (np.abs(qg - qb) <= np.maximum(2e-5, 2e-8 * np.maximum(np.abs(qg), np.abs(qb)))))
        if not np.all(good):
            bads = np.flatnonzero(~good)
            raise InvalidPhysics('pellet balances failed: ' + str([(int(k), norm[k], atoms[k], q[k, -1]) for k in bads[:5]]))
        return x, comp

    def solve_points(self, zs, rs, gas_states, bed, update_cache=False):
        zs = np.asarray(zs, int); rs = np.asarray(rs, int); BG = self.backgrounds(gas_states, bed)
        X, cold = self.guesses(zs, rs, BG)
        A = np.asarray([self.activity[z, ir] for z, ir in zip(zs, rs)], float)
        if np.any(A <= 0):
            raise ValueError('zero catalyst activity is not supported')
        outputs = []; states = []
        for start in range(0, len(X), self.chunk):
            end = min(start + self.chunk, len(X)); xx, cc = self._solve_batch(X[start:end], BG[start:end], A[start:end], cold[start:end])
            states.append(xx); outputs.append(cc)
        X = np.concatenate(states); comp = tuple(np.concatenate([c[k] for c in outputs], axis=0) for k in range(len(outputs[0])))
        if update_cache:
            for z, ir, x in zip(zs, rs, X):
                self.cache[int(z), int(ir)] = x.copy(); self.recent_by_radius[int(ir)] = x.copy()
        N, qg, qb = comp[1], comp[2], comp[3]
        iq = BG[:, 9] * (BG[:, 1] - BG[:, 0])        # sensible heat of the inert pellets
        mixed = np.column_stack((self.phi * N, self.phi * qg + (1 - self.phi) * iq, self.phi * qb + (1 - self.phi) * iq))
        return mixed, X, comp, BG

    # ------------------------------------------------------------ bed and wall heat balance
    def _thermal_impl(self, u):
        g = self.g; nx, nr = g.nx, g.nr; ns = nx * nr
        Ts = u[:ns].reshape(nx, nr); Tw = u[ns:]
        kr = jax.vmap(self.coeffs, in_axes=(0, None))(Ts.ravel(), g.mass_flux_kg_m2s)[2].reshape(nx, nr)
        hw = jax.vmap(self.coeffs, in_axes=(0, None))((Tw + Ts[:, -1]) / 2, g.mass_flux_kg_m2s)[3]
        L = j.zeros((ns + nx, ns + nx), dtype=j.float64)
        for r in range(nr - 1):                                   # radial conduction
            a = j.arange(nx) * nr + r; b = a + 1
            w = 2 * kr[:, r] * kr[:, r + 1] / (kr[:, r] + kr[:, r + 1]) * 2 * j.pi * g.radius_edges[r + 1] * g.dz / (g.radius_centers[r + 1] - g.radius_centers[r])
            L = L.at[a, a].add(w).at[b, b].add(w).at[a, b].add(-w).at[b, a].add(-w)
        a = j.arange(nx) * nr + nr - 1; b = ns + j.arange(nx); ar = 2 * j.pi * g.ri_m * g.dz      # outer ring to wall
        w = 1 / (1 / (hw * ar) + (g.ri_m - g.radius_centers[-1]) / (kr[:, -1] * ar))
        L = L.at[a, a].add(w).at[b, b].add(w).at[a, b].add(-w).at[b, a].add(-w)
        for r in range(nr):                                       # axial conduction, one third of the radial value
            a = j.arange(nx - 1) * nr + r; b = a + nr
            w = 2 * kr[:-1, r] * kr[1:, r] / (kr[:-1, r] + kr[1:, r]) / 3 * g.annulus_areas[r] / g.dz
            L = L.at[a, a].add(w).at[b, b].add(w).at[a, b].add(-w).at[b, a].add(-w)
        a = ns + j.arange(nx - 1); b = a + 1; w = float(pr.WALL_AXIAL_CONDUCTANCE_W_K) * .3 / g.dz     # axial wall conduction
        L = L.at[a, a].add(w).at[b, b].add(w).at[a, b].add(-w).at[b, a].add(-w)
        xp, fp = pr.TI, pr.QI                                     # insulation loss, linear inter- and extrapolation
        loss = j.interp(Tw, xp, fp)
        loss = j.where(Tw < xp[0], fp[0] + (Tw - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]), loss)
        loss = j.where(Tw > xp[-1], fp[-1] + (Tw - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), loss)
        return L, j.maximum(0., loss) * g.dz / float(pr.REFERENCE_LENGTH_M)

    def evaluate(self, u, power=None):
        """Residual of the bed and wall heat balances, W, after marching the gas through the tube."""
        g = self.g; u = np.asarray(u, float)
        power = self.heating.total_W if power is None else power
        if power != self.heating.total_W:
            raise ValueError('the heating is fixed during a solve')
        if u.shape != (self.n,) or not np.isfinite(u).all() or u.min() < 310 or u.max() > 1750:
            raise ValueError('invalid thermal state')
        L, loss = self._thermal(j.asarray(u)); self._thermal_state = u.copy()
        self.L = sparse.csr_matrix(np.array(L)); self.loss = np.r_[np.zeros(self.ns), np.array(loss) / (u[self.ns:] - g.ambient_K)]
        e = self._march(u, power)
        pc = np.array(self.heating.cells(g))
        rr = L @ j.asarray(u)
        rr = rr.at[:self.ns].add(j.asarray(e.q_bed_W.ravel())).at[self.ns:].add(loss - j.asarray(pc))
        if np.max(np.abs(np.array(rr) - e.residual_W)) > 1e-8:
            raise ValueError('the two assemblies of the heat balance differ')
        e.residual_W = np.array(rr)
        return e

    def _march(self, u, power_W):
        u = np.asarray(u, float)
        g = self.g
        if u.shape != (self.n,) or not np.all(np.isfinite(u)) or np.min(u) < 310 or np.max(u) > 1750:
            raise InvalidPhysics('invalid solid or wall temperature state')
        Ts = u[:self.ns].reshape(g.nx, g.nr)
        Tw = u[self.ns:]
        pc = self.heating.cells(g)
        inlet = GasState.from_inlet(g.F0, g.Tin_K, g.Pin_Pa)
        F = [inlet.F.copy()]; P = [inlet.P_Pa]; H = [inlet.H_W]; Tg = [g.Tin_K]
        qb = np.zeros((g.nx, g.nr)); qg = np.zeros_like(qb)
        n_ok = n_bad = 0
        worst_gas_energy = worst_atom = worst_flow = 0.
        initial_step_m = .002
        for iz in range(g.nx):
            r = self.axial.host_integrate(iz, inlet, Ts[iz], g, initial_step_m=initial_step_m)
            initial_step_m = min(g.dz, max(.002, r.max_step_m))    # an oversized proposal is rejected by the error control
            inlet = r.outlet
            F.append(inlet.F.copy()); P.append(inlet.P_Pa); H.append(inlet.H_W)
            Tg.append(inlet.gas_temperature())
            qb[iz] = r.q_bed_by_ring_W; qg[iz] = r.q_gas_by_ring_W
            n_ok += r.accepted; n_bad += r.rejected
            worst_gas_energy = max(worst_gas_energy, abs(r.energy_residual_W))
            worst_atom = max(worst_atom, abs(r.atoms_residual_mol_s))
            worst_flow = max(worst_flow, abs(r.flow_residual_mol_s))
            if (abs(r.energy_residual_W) > 1.e-7 or abs(r.atoms_residual_mol_s) > 1.e-8 or abs(r.flow_residual_mol_s) > 1.e-9):
                raise InvalidPhysics('cell material or enthalpy balance failed')
        residual = self.L @ u
        residual[:self.ns] += qb.ravel()
        residual[self.ns:] += self.loss[self.ns:] * (Tw - g.ambient_K) - pc
        loss = float(np.sum(self.loss[self.ns:] * (Tw - g.ambient_K)))
        en = float(power_W - loss - (H[-1] - H[0]))
        film_mismatch = float(np.sum(qg - qb))
        ff = np.asarray(F)
        mass = float((ff[-1] - ff[0]) @ MW)
        atom = float(np.max(np.abs(ATOMS @ (ff[-1] - ff[0]))))
        # finite-volume identity: residual sum + global energy imbalance = bed-to-pellet minus pellet-to-gas heat
        identity_error = float(en + np.sum(residual) - np.sum(qb - qg))
        if abs(identity_error) > max(2.e-6, 1.e-8 * abs(power_W)):
            raise InvalidPhysics('global energy bookkeeping identity failed')
        return Evaluation(residual, ff, np.asarray(P), np.asarray(Tg), np.asarray(H), qb, qg, en, film_mismatch, mass, atom,
                          n_ok, n_bad, worst_gas_energy, worst_atom, worst_flow, identity_error)

    # ------------------------------------------------------------ Jacobian of the reacting gas propagation
    def _propagate_impl(self, M, J):
        nr = self.g.nr; ns = self.ns; ng = 7

        def body(G, inputs):
            z, m = inputs; start = z * nr
            Q = m[ng + nr:, :ng] @ G
            Q = lax.dynamic_update_slice(Q, lax.dynamic_slice(Q, (0, start), (nr, nr)) + m[ng + nr:, ng:ng + nr], (0, start))
            new = m[:ng, :ng] @ G
            new = lax.dynamic_update_slice(new, lax.dynamic_slice(new, (0, start), (ng, nr)) + m[:ng, ng:ng + nr], (0, start))
            return new, Q * 1000.
        _, rows = lax.scan(body, j.zeros((ng, ns)), (j.arange(self.g.nx), M))
        return J.at[:ns, :ns].add(rows.reshape(ns, ns))

    def preconditioner(self, e):
        """Finite-difference linearization of each cell's gas propagation (cell-midpoint state), propagated
        down the tube with matrix exponentials and added to the conduction matrix."""
        start = time.perf_counter(); g = self.g; nr = g.nr; ns = self.ns
        u = self._thermal_state; Ts = u[:ns].reshape(g.nx, nr)
        scales = np.r_[np.full(5, g.F0.sum()), 1000., 1e6]; steps = np.r_[np.full(5, 1e-5), .002, .0001]
        base = np.column_stack(((e.F_edges[:-1] + e.F_edges[1:]) / 2, (e.H_edges_W[:-1] + e.H_edges_W[1:]) / 2,
                                (e.P_edges_Pa[:-1] + e.P_edges_Pa[1:]) / 2))
        zs = []; rs = []; gas = []; bed = []
        for z in range(g.nx):
            for mode in range(8):
                st = base[z].copy()
                if mode:
                    st[mode - 1] += steps[mode - 1] * scales[mode - 1]
                for ir in range(nr):
                    zs.append(z); rs.append(ir); gas.append(st); bed.append(Ts[z, ir])
            for ir in range(nr):
                zs.append(z); rs.append(ir); gas.append(base[z]); bed.append(Ts[z, ir] + .2)
        values, _, _, _ = self.solve_points(zs, rs, np.asarray(gas), np.asarray(bed), False)
        area = g.particle_area_density * g.annulus_areas
        blocks = []; offset = 0

        def grad(st, rows):
            T = float(self.temperatures(st)[0]); F = st[:5]; P = st[6]
            dp = ergun_dPdz(g, F, T, P)
            return np.r_[np.sum(rows[:, :5] * area[:, None], axis=0), rows[:, 5] @ area, dp] / scales, rows[:, 6] * area / 1000.
        for z in range(g.nx):
            rows = values[offset:offset + 8 * nr].reshape(8, nr, 7); alter = values[offset + 8 * nr:offset + 9 * nr]; offset += 9 * nr
            f0, q0 = grad(base[z], rows[0]); A = np.empty((7, 7)); C = np.empty((nr, 7)); B = np.empty((7, nr)); D = np.empty((nr, nr))
            for k, h in enumerate(steps):
                st = base[z].copy(); st[k] += h * scales[k]; f, q = grad(st, rows[k + 1]); A[:, k] = (f - f0) / h; C[:, k] = (q - q0) / h
            for ir in range(nr):
                rr = rows[0].copy(); rr[ir] = alter[ir]; f, q = grad(base[z], rr); B[:, ir] = (f - f0) / .2; D[:, ir] = (q - q0) / .2
            block = np.zeros((7 + 2 * nr, 7 + 2 * nr)); block[:7, :7] = A; block[:7, 7:7 + nr] = B; block[7 + nr:, :7] = C
            block[7 + nr:, 7:7 + nr] = D; blocks.append(block * g.dz)
        M = self._propagators(j.asarray(np.array(blocks)))
        J, loss = self._thermal(j.asarray(u)); _, lp = self._thermal(j.asarray(np.r_[u[:ns], u[ns:] + .01]))
        _, lm = self._thermal(j.asarray(np.r_[u[:ns], u[ns:] - .01]))
        J = J.at[j.arange(ns, ns + g.nx), j.arange(ns, ns + g.nx)].add((lp - lm) / .02)
        J = self._linear_propagation(M, J)
        if not np.isfinite(np.array(J)).all():
            raise ValueError('non-finite preconditioner')
        self.log(f'  Jacobian: {time.perf_counter() - start:.1f} s, {len(values)} pellet solves')
        return J

    # ------------------------------------------------------------ steady solve
    def solve(self, u0, **options):
        """Broyden iterations with a backtracking line search on the maximum cell residual."""
        o = dict(self.options); o.update(options)
        u = np.array(u0, float).copy()
        tol = o['tol_W']; globaltol = o['max_global_energy_W']; maxit = o['max_iter']; maxstep = o['max_step_K']
        e = self.evaluate(u); norm = float(np.max(np.abs(e.residual_W))); history = [norm]
        B = self.preconditioner(e); self.last_outer_jacobian = np.array(B)
        for it in range(maxit):
            self.log(f'  iteration {it}: max cell residual {norm:.3g} W, global {e.total_energy_residual_W:.3g} W')
            if norm < tol and abs(e.total_energy_residual_W) < globaltol:
                return SteadyResult(u, e, True, it, history)
            step = np.array(outer_linear_step(B, j.asarray(e.residual_W))); lam = min(1., maxstep / max(float(np.max(np.abs(step))), 1e-30))
            good = False
            for _ in range(8):
                trial = u + lam * step
                try:
                    er = self.evaluate(trial); rn = float(np.max(np.abs(er.residual_W)))
                    good = (rn < norm * (1 - 1e-5 * lam) or (rn < tol and abs(er.total_energy_residual_W) < abs(e.total_energy_residual_W)))
                    if good:
                        break
                except (InvalidPhysics, ValueError, FloatingPointError):
                    pass
                lam *= .5
            if not good:
                raise ValueError('line search failed')
            s = trial - u; B = broyden_update(B, j.asarray(s), j.asarray(er.residual_W - e.residual_W)); u, e, norm = trial, er, rn
            history.append(norm)
            self.last_outer_jacobian = np.array(B)
        return SteadyResult(u, e, norm < tol and abs(e.total_energy_residual_W) < globaltol, maxit, history)

    def check_steady(self, result):
        """Acceptance of a steady state: heat balances, element balance and consistent gas edges."""
        e = result.last
        if (not result.converged or not np.isfinite(result.state).all() or not np.isfinite(e.residual_W).all() or
                np.max(np.abs(e.residual_W)) >= self.options['tol_W'] or not np.isfinite(e.total_energy_residual_W) or
                abs(e.total_energy_residual_W) >= self.options['max_global_energy_W'] or
                e.atoms_residual_mol_s > 1e-8 or abs(e.gas_bed_flux_mismatch_W) > 1e-5):
            raise InvalidPhysics('steady state not accepted')
        for F, T, H in zip(e.F_edges, e.Tgas_edges_K, e.H_edges_W):
            if not np.isfinite(F).all() or np.min(F) < 0 or abs(float(F @ shomate_h(T)) - H) > 1e-6:
                raise InvalidPhysics('inconsistent molar flow, temperature and enthalpy at a cell edge')

    def environment(self, steady):
        """Shell temperatures and partial pressures of all pellets of an accepted state.

        Every pellet is solved again at the cell-mean gas state and checked against
        the NumPy reference residual (pellet_reference.PelletWithFilm).
        """
        self.check_steady(steady)
        g = self.g; e = steady.last; Ts = steady.state[:self.ns].reshape(g.nx, g.nr)
        gas = np.column_stack(((e.F_edges[:-1] + e.F_edges[1:]) / 2, (e.H_edges_W[:-1] + e.H_edges_W[1:]) / 2,
                               (e.P_edges_Pa[:-1] + e.P_edges_Pa[1:]) / 2))
        zs = np.repeat(np.arange(g.nx), g.nr); rs = np.tile(np.arange(g.nr), g.nx)
        _, X, comp, BG = self.solve_points(zs, rs, np.repeat(gas, g.nr, axis=0), Ts.ravel(), False)
        n = self.pellet.n
        for x, b, a in zip(X, BG, self.activity.ravel()):
            bg = GasBedState(b[0], b[1], b[2], b[3:8], b[8], b[9], b[10])
            rr, diag = self.reference.evaluate(x, bg, a)
            pell = self.reference.internal.check(x[:7 * n], diag['film_surface'], a, float(np.max(np.abs(rr[:7 * n]))))
            if not pell.passed or np.max(np.abs(rr)) >= min(1e-8, self.solver_tolerance):
                raise ValueError('a pellet failed the check against the reference residual')
        T = comp[4].reshape(g.nx, g.nr, n); P = comp[5].reshape(T.shape); Y = comp[6].reshape(T.shape + (5,))
        sw = np.array(self.pellet.vol); sw /= sw.sum()
        return dict(T_K=T, pH2O_bar=P * Y[..., 1] / 1e5, pH2_bar=P * Y[..., 2] / 1e5, shell_weights=sw, pressure_Pa=P)


class AxialMarch:
    """Adaptive explicit midpoint march of the gas over one axial cell, with an embedded Euler error
    estimate. Each source evaluation solves the six ring pellets; the whole cell runs as one compiled loop."""

    def __init__(self, reactor):
        self.b = reactor; self.p = reactor.pellet; self.g = reactor.g
        self.area = j.asarray(self.g.particle_area_density * self.g.annulus_areas)
        self.atoms = reactor.atoms
        self.nr = self.g.nr
        self.integrate_compiled = jax.jit(self.integrate)

    def temperature(self, V):
        T = self.b._temperature_one(V)
        good = (j.min(V[:5]) >= -1e-13) & (j.sum(V[:5]) > 0) & (V[6] >= 1e5)
        good = good & (j.abs(j.dot(V[:5], self.p.enthalpy(j.asarray([T]))[0]) - V[5]) < 1e-6)
        return T, good

    def initial(self, BG):
        n = self.p.n
        row = j.concatenate((BG[:, 3:8], BG[:, 2:3] / 1e6, BG[:, 1:2] / 1000), axis=1)
        return j.concatenate((j.tile(row[:, None, :], (1, n, 1)).reshape(len(BG), 7 * n), BG[:, 3:8], BG[:, 1:2] / 1000), axis=1)

    def source(self, V, Tbed, A, X):
        film = self.b.film
        T, valid = self.temperature(V); Y = V[:5] / j.sum(V[:5])
        hf = jax.vmap(self.b.coeffs, in_axes=(0, None))((T + Tbed) / 2, self.g.mass_flux_kg_m2s)[1]
        BG = j.column_stack((j.full(self.nr, T), Tbed, j.full(self.nr, V[6]), j.tile(Y, (self.nr, 1)),
                             j.full(self.nr, film.film_thickness_m), hf, j.full(self.nr, film.solid_contact_h_W_m2K)))
        valid = valid & (j.min(V[:5]) >= 0) & j.all(j.isfinite(BG)) & (T > 300) & (T < 1800)

        def solve(_):
            xx, ok, norm, nit = self.p.newton_batch(X, BG, A, self.b.solver_tolerance, 42)
            bad = ~ok

            def continuation(inp):
                xx, bad = inp; xx = j.where(bad[:, None], self.initial(BG), xx)
                alive = j.ones(self.nr, dtype=bool)

                def step(index, state):
                    xx, alive = state
                    start = j.maximum(A * 1e-6, 1e-8)
                    strength = j.where(index == 0, 0., j.exp(j.log(start) + (index - 1) / 9. * (j.log(A) - j.log(start))))
                    act = j.where(bad, strength, A)
                    xt, ot, nt, it = self.p.newton_batch(xx, BG, act, self.b.solver_tolerance, 42)
                    return xt, alive & ot
                xx, alive = lax.fori_loop(0, 11, step, (xx, alive))
                return xx, j.where(bad, alive, ok)
            xx, ok = lax.cond(j.any(bad), continuation, lambda inp: (inp[0], ok), (xx, bad))
            comp = self.p.outputs_batch(xx, BG, A)
            rr, N, qg, qb, tc, pc, yc, q, rate, pe = comp
            limit = j.maximum(2e-5, 2e-8 * j.maximum(j.abs(qg), j.abs(qb)))
            polish = j.any(j.abs(qg - qb) > limit) | j.any(j.max(j.abs(rr), axis=1) >= self.b.solver_tolerance)

            def refine(x): return self.p.newton_batch(x, BG, A, min(1e-11, self.b.solver_tolerance * .01), 42)[0]
            xx = lax.cond(polish, refine, lambda x: x, xx)
            rr, N, qg, qb, tc, pc, yc, q, rate, pe = self.p.outputs_batch(xx, BG, A)
            integrated = j.sum(self.p.rho * A[:, None, None] * (rate @ STOICH.T) * self.p.vol[None, :, None], axis=1) / self.p.area[-1]
            good = j.all(ok) & j.all(j.isfinite(rr)) & (j.max(j.abs(rr)) < self.b.solver_tolerance)
            good = good & (j.max(j.abs(N @ self.atoms.T)) < 1e-7) & (j.max(j.abs(N - integrated)) < 5e-7)
            good = good & (j.max(j.abs(q[:, -1])) < 1e-2) & (j.max(j.abs(pe)) <= 80)
            good = good & j.all(j.abs(qg - qb) <= j.maximum(2e-5, 2e-8 * j.maximum(j.abs(qg), j.abs(qb))))
            iq = hf * (Tbed - T)
            mol = self.b.phi * N * self.area[:, None]
            qgas = (self.b.phi * qg + (1 - self.b.phi) * iq) * self.area
            qbed = (self.b.phi * qb + (1 - self.b.phi) * iq) * self.area
            # Ergun pressure drop with the tabulated molar masses
            rho = V[6] * j.dot(Y, pr.MW_TABLE) / (R_PELLET * T); vel = self.g.mass_flux_kg_m2s / rho; mu = self.p.viscosity(T)
            ep = self.g.bed_void; dp = self.g.dp_m
            gradp = -(150 * mu * (1 - ep) ** 2 * vel / (dp * dp * ep ** 3) + 1.75 * (1 - ep) * rho * vel ** 2 / (dp * ep ** 3))
            grad = j.concatenate((j.sum(mol, axis=0), j.asarray([j.sum(qgas), gradp])))
            return grad, mol, qgas, qbed, xx, good

        def invalid(_): return j.zeros(7), j.zeros((self.nr, 5)), j.zeros(self.nr), j.zeros(self.nr), X, j.array(False)
        return lax.cond(valid, solve, invalid, None)

    def integrate(self, inlet, Tbed, A, X, initial_h, rtol, atol):
        length = self.g.dz; h = j.minimum(initial_h, length)
        # pos, h, V, X, ringF, qg, qb, nOK, nBAD, minH, maxH, k0, cache, fail, evals
        start = (j.array(0.), h, inlet, X, j.zeros((self.nr, 5)), j.zeros(self.nr), j.zeros(self.nr),
                 j.array(0), j.array(0), j.array(j.inf), j.array(0.), j.zeros(7), j.array(False), j.array(0), j.array(0))

        def cond(s): return (s[0] < length - 1e-14) & (s[13] == 0) & (s[7] + s[8] < 1200)

        def body(s):
            pos, h, V, X, rf, qg, qb, nok, nbad, minh, maxh, kcache, hascache, fail, evals = s
            remaining = length - pos
            # the clipped last step of a cell is a normal step with error control and quadrature
            final_remainder = (h >= remaining) & (remaining > 0.)
            h = j.minimum(h, remaining)
            too_small = (h < 1e-7) & (~final_remainder)

            def cached(_): return kcache, X, j.array(True), j.array(0)

            def fresh(_):
                k, m, q, b, x, good = self.source(V, Tbed, A, X); return k, x, good, j.array(1)
            k0, x0, good0, ne = lax.cond(hascache, cached, fresh, None)
            vmid = V + .5 * h * k0
            k1, m1, q1, b1, x1, good1 = self.source(vmid, Tbed, A, x0)
            v1 = V + h * k1; ve = V + h * k0
            t0, t0ok = self.temperature(V); tm, tmok = self.temperature(vmid); tn, tnok = self.temperature(v1); te, teok = self.temperature(ve)
            guarded = (good1 & tmok & tnok & teok & (j.min(v1[:5]) >= -1e-13) & (j.min(vmid[:5]) >= -1e-13)
                       & (j.abs(v1[0] - V[0]) <= .15 * j.maximum(V[0], 1e-6)) & (j.abs(tn - t0) <= 14))
            errF = j.max(j.abs(v1[:5] - ve[:5])) / (rtol * j.maximum(inlet[0], 1e-8))
            errT = j.abs(tn - te) / atol; err = j.maximum(errF, errT)
            accepted = good0 & (~too_small) & guarded & (err <= 1.)
            errh = h * j.maximum(.2, j.minimum(.8, .8 / j.maximum(j.sqrt(j.maximum(err, 0)), 1e-12)))
            rejecth = j.where(guarded, errh, h * .5)
            grow = h * j.minimum(1.8, j.maximum(1., .85 / j.maximum(j.sqrt(j.maximum(err, 1e-12)), 1e-12)))
            newh = j.where(accepted, grow, rejecth)
            newpos = j.where(accepted, pos + h, pos); newV = j.where(accepted, v1, V)
            rf = rf + j.where(accepted, m1 * h, 0.); qg = qg + j.where(accepted, q1 * h, 0.); qb = qb + j.where(accepted, b1 * h, 0.)
            fail = j.where(~good0, 2, j.where(too_small, 3, fail))
            return (newpos, newh, newV, x1, rf, qg, qb, nok + accepted.astype(j.int32), nbad + (~accepted).astype(j.int32),
                    j.where(accepted, j.minimum(minh, h), minh), j.where(accepted, j.maximum(maxh, h), maxh), k0, ~accepted, fail, evals + ne + 1)
        result = lax.while_loop(cond, body, start)
        return result

    def host_integrate(self, iz, inlet, Tbed, g, initial_step_m=.002):
        gs = np.tile(inlet.vector, (g.nr, 1)); BG = self.b.backgrounds(gs, Tbed)
        X, cold = self.b.guesses(np.full(g.nr, iz), np.arange(g.nr), BG)
        if cold.any():
            raise InvalidPhysics('the gas march needs a warm start for every pellet')
        A = self.b.activity[iz]
        out = self.integrate_compiled(j.asarray(inlet.vector), j.asarray(Tbed), j.asarray(A), j.asarray(X), float(initial_step_m),
                                      self.b.rtol_flow, self.b.atol_T_K)
        out = tuple(np.array(x) for x in out)
        pos, h, V, X, rf, qg, qb, nok, nbad, minh, maxh, k, cache, fail, evals = out
        if fail or pos < g.dz - 1e-14:
            raise InvalidPhysics(f'gas march failed in cell {iz} (code {fail}, position {pos}, steps {nok}/{nbad})')
        for ir, x in enumerate(X):
            self.b.cache[iz, ir] = x.copy(); self.b.recent_by_radius[ir] = x.copy()
        self.b.stats['pellet_batches'] += int(evals); self.b.stats['pellets'] += int(evals) * g.nr
        dF = V[:5] - inlet.F; dH = V[5] - inlet.H_W
        energy = float(dH - qg.sum()); atoms = float(np.max(np.abs(ATOMS @ dF))); flow = float(np.max(np.abs(dF - rf.sum(axis=0))))
        if np.any(np.abs(qg - qb) > np.maximum(1e-6, 2e-8 * np.maximum(np.abs(qg), np.abs(qb)))):
            raise InvalidPhysics('integrated gas and bed heat of a cell differ')
        return AxialResult(GasState.from_vector(V), rf, qg, qb, int(nok), int(nbad), float(minh), float(maxh), float(dF @ MW),
                           atoms, energy, flow, int(evals))
