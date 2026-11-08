"""JAX implementation of the pellet and film equations of pellet_reference.py.

Same discretization and the same equations, written for batched evaluation
with exact derivatives. Newton's method uses a column-coloured Jacobian built
from forward-mode derivatives along non-conflicting structural directions.
Batch layout of the boundary vector bg:
[T_gas, T_bed, pressure, y_bulk(5), film thickness, h_film, h_contact].
"""
from __future__ import annotations

import jax
import jax.numpy as j
import numpy as np
from jax import lax
from jax.scipy.linalg import expm
from scipy.optimize._numdiff import group_columns

from .pellet_reference import FSG_DIFFUSION_VOLUMES, prepared_coefficients
from .properties import HIGH, LOW, MW, SWITCH

jax.config.update('jax_enable_x64', True)

R = 8.31446261815324
STOICH = j.array([[-1., 0., -1.], [-1., -1., -2.], [3., 1., 4.], [1., -1., 0.], [0., 1., 1.]])


class JaxPellet:
    """Batched JAX version of the pellet equations of a reference pellet from pellet_reference.py."""
    def __init__(self, reference, viscosity_T, viscosity_mu, device='cpu'):
        devices = jax.devices(device)
        if not devices:
            raise RuntimeError('requested device is unavailable: ' + device)
        self.device = devices[0]
        p = reference.internal.p; n = reference.n
        pair, knudsen = prepared_coefficients(p)
        self.n = n; self.size = 7 * n + 6
        put = lambda x: jax.device_put(j.asarray(x, dtype=j.float64), self.device)
        self.area = put(reference.internal.area); self.vol = put(reference.internal.vol)
        self.dr = put(reference.internal.dr); self.pair = put(pair); self.knudsen = put(knudsen)
        self.low = put(LOW); self.high = put(HIGH); self.switch = put(SWITCH); self.mw = put(MW)
        self.mu_T = put(viscosity_T); self.mu = put(viscosity_mu)
        self.permeability = p.permeability_m2; self.rho = p.rho_cat_kg_per_m3_pellet
        self.keff = p.k_effective_W_m_K; self.eps_tau = p.porosity / p.tortuosity
        self.pore_radius = p.pore_radius_m
        mg = np.asarray(MW) * 1000.
        self.sqrt_red = put(np.sqrt(2. / (1. / mg[:, None] + 1. / mg[None, :])))
        self.volume_sum = put((FSG_DIFFUSION_VOLUMES[:, None] ** (1 / 3) + FSG_DIFFUSION_VOLUMES[None, :] ** (1 / 3)) ** 2)
        self.lo = put(np.r_[np.tile([0.] * 5 + [.01, .30], n), [0.] * 5 + [.3]])
        self.hi = put(np.r_[np.tile([1.] * 5 + [10., 1.8], n), [1.] * 5 + [1.8]])
        pattern = reference.pattern.toarray().astype(bool); groups = group_columns(reference.pattern)
        for c in range(groups.max() + 1):
            if np.max(pattern[:, groups == c].sum(axis=1)) > 1:
                raise ValueError('conflicting structural derivative colouring')
        seed = np.zeros((self.size, int(groups.max()) + 1)); seed[np.arange(self.size), groups] = 1.
        self.pattern = jax.device_put(j.asarray(pattern), self.device)
        self.groups = jax.device_put(j.asarray(groups), self.device); self.derivative_seed = put(seed)
        self.residual_batch = jax.jit(jax.vmap(self.residual, in_axes=(0, 0, 0)))
        self.outputs_batch = jax.jit(jax.vmap(self.components, in_axes=(0, 0, 0)))
        self.newton_batch = jax.jit(jax.vmap(self.newton, in_axes=(0, 0, 0, None, None)))

    def viscosity(self, T):
        xp, fp = self.mu_T, self.mu
        y = j.interp(T, xp, fp)
        y = j.where(T < xp[0], fp[0] + (T - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]), y)
        y = j.where(T > xp[-1], fp[-1] + (T - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), y)
        return j.maximum(1e-5, y)

    def enthalpy(self, T):
        tt = T[..., None] / 1000.
        c = j.where(T[..., None, None] < self.switch[None, :, None], self.low, self.high)
        return 1000 * (c[..., 0] * tt + c[..., 1] * tt ** 2 / 2 + c[..., 2] * tt ** 3 / 3 + c[..., 3] * tt ** 4 / 4 - c[..., 4] / tt + c[..., 5])

    def cp(self, T):
        tt = T / 1000.
        c = j.where((T < self.switch)[:, None], self.low, self.high)      # scalar T, one switch per species
        return c[:, 0] + c[:, 1] * tt + c[:, 2] * tt ** 2 + c[:, 3] * tt ** 3 + c[:, 4] / tt ** 2

    def kinetics(self, T, p):
        """Xu and Froment (1989) rates of reactions 1-3."""
        h = j.maximum(p[:, 2], 1e-4); ch, w, _, co, co2 = p.T
        k1 = j.exp(-26830 / T + 30.114); k2 = j.exp(4400 / T - 4.036)
        a = 1.173611111e15 * j.exp(-240100 / (R * T)); b = 5.430555556e5 * j.exp(-67130 / (R * T)); c = 2.833333333e14 * j.exp(-243900 / (R * T))
        dc = 40.91 * j.exp((70650 / R) * (1 / T - 1 / 648.)); dh = .0296 * j.exp((82900 / R) * (1 / T - 1 / 648.))
        dm = .179 * j.exp((38280 / R) * (1 / T - 1 / 823.)); dw = .4152 * j.exp((-88680 / R) * (1 / T - 1 / 823.))
        den = 1 + dc * co + dh * h + dm * ch + dw * w / h
        return j.stack((a * (ch * w - co * h ** 3 / k1) / (h ** 2.5 * den ** 2), b * (co * w - co2 * h / k2) / (h * den ** 2),
                        c * (ch * w * w - co2 * h ** 4 / (k1 * k2)) / (h ** 3.5 * den ** 2)), axis=-1)

    def components(self, z, bg, activity):
        n = self.n; raw = z[:7 * n].reshape(n, 7); y = raw[:, :5] / j.sum(raw[:, :5], axis=1, keepdims=True)
        # Keep the unit conversions as separate rounding points, as in the NumPy model; otherwise
        # fusion can turn (1000 x - 1000 y) into 1000 (x - y), which changes stiff energy rows.
        P = lax.optimization_barrier(raw[:, 5] * 1e6)
        T = lax.optimization_barrier(raw[:, 6] * 1000)
        sy = z[7 * n:7 * n + 5]; sy = sy / j.sum(sy); Ts = lax.optimization_barrier(z[-1] * 1000)
        yr = j.concatenate((y[1:], sy[None, :])); Pr = j.concatenate((P[1:], bg[2:3])); Tr = j.concatenate((T[1:], j.asarray([Ts])))
        Pf = (P + Pr) / 2; Tf = (T + Tr) / 2; xf = (y + yr) / 2
        Dij = .00143 * Tf[:, None, None] ** 1.75 / ((Pf[:, None, None] / 101325.) * self.sqrt_red * self.volume_sum) * 1e-4 * self.eps_tau
        Dij = Dij.at[:, j.arange(5), j.arange(5)].set(0.)
        inv = j.where(j.eye(5, dtype=bool)[None, :, :], 0., 1 / j.where(Dij > 0, Dij, 1.))
        dk = (2 / 3) * self.pore_radius * j.sqrt(8 * R * Tf[:, None] / (j.pi * self.mw)) * self.eps_tau; mu = self.viscosity(Tf)
        matrix = -xf[:, :, None] * inv
        diag = 1 / dk + j.sum(xf[:, None, :] * inv, axis=-1)
        matrix = matrix.at[:, j.arange(5), j.arange(5)].set(diag)
        force = (-Pf[:, None] * (yr - y) / self.dr[:, None] - xf * (1 + self.permeability * Pf[:, None] / (mu[:, None] * dk)) * (Pr - P)[:, None] / self.dr[:, None]) / (R * Tf[:, None])
        N = j.linalg.solve(matrix, force[..., None])[..., 0]
        rate = self.kinetics(T, P[:, None] / 1e5 * y); S = self.rho * activity * (rate @ STOICH.T)
        species = (self.area[1:, None] * N - self.area[:-1, None] * j.concatenate((j.zeros((1, 5)), N[:-1]))) / self.vol[:, None] - S
        q = j.sum(N * self.enthalpy(Tf), axis=1) - self.keff * (Tr - T) / self.dr
        energy = (self.area[1:] * q - self.area[:-1] * j.concatenate((j.zeros(1), q[:-1]))) / self.vol
        internal = j.stack((species[:, 0] / 1e4, species[:, 1] / 1e4, species[:, 2] / 1e4, species[:, 3] / 1e4, species[:, 4] / 1e4,
                            energy / 1e9, j.sum(raw[:, :5], axis=1) - 1), axis=1).ravel()
        Ns = N[-1]; Tfilm = (Ts + bg[0]) / 2
        Db = .00143 * Tfilm ** 1.75 / ((bg[2] / 101325.) * self.sqrt_red * self.volume_sum) * 1e-4
        Db = Db.at[j.arange(5), j.arange(5)].set(0.)
        invb = j.where(j.eye(5, dtype=bool), 0., 1 / j.where(Db > 0, Db, 1.))
        mm = Ns[:, None] * invb
        mm = mm.at[j.arange(5), j.arange(5)].set(-j.sum(Ns[None, :] * invb, axis=1))
        mat = -bg[8] / (bg[2] / (R * Tfilm)) * mm
        transported = expm(mat, max_squarings=32) @ sy
        fm = (transported - bg[3:8])[:4]
        pe = j.dot(Ns, self.cp(Tfilm)) / bg[9]
        safe = j.where(j.abs(pe) < 1e-4, 1., pe)
        factor = j.where(j.abs(pe) < 1e-4, 1 + pe / 2 + pe ** 2 / 12 - pe ** 4 / 720, safe / (-j.expm1(-safe)))
        hg = self.enthalpy(j.asarray([bg[0]]))[0]
        gasq = j.dot(Ns, hg) + bg[9] * (Ts - bg[0]) * factor
        bedq = gasq - q[-1]          # the pellet surface is at the bed temperature
        residual = j.concatenate((internal, fm, j.asarray([j.sum(z[7 * n:7 * n + 5]) - 1, (Ts - bg[1]) / 1000])))
        return residual, Ns, gasq, bedq, T, P, y, q, rate, pe

    def residual(self, z, bg, activity):
        return self.components(z, bg, activity)[0]

    def colored_jacobian(self, z, bg, activity):
        _, linear = jax.linearize(lambda x: self.residual(x, bg, activity), z)
        compressed = jax.vmap(linear, in_axes=1, out_axes=1)(self.derivative_seed)
        return j.where(self.pattern, compressed[:, self.groups], 0.)

    def newton(self, x, bg, activity, tol, max_iter):
        x = j.clip(x, self.lo, self.hi); r = self.residual(x, bg, activity)
        init = (j.array(0), x, r, j.array(0))         # code 0 running, -1 line-search or non-finite failure

        def cond(s):
            it, x, r, code = s
            return (it < max_iter) & (j.max(j.abs(r)) >= tol) & (code == 0) & j.all(j.isfinite(r))

        def body(s):
            it, x, r, code = s
            J = self.colored_jacobian(x, bg, activity)
            dx = j.linalg.solve(J, -r)
            low = j.where(dx < 0, .99 * (x - self.lo) / j.maximum(-dx, 1e-300), j.inf)
            high = j.where(dx > 0, .99 * (self.hi - x) / j.maximum(dx, 1e-300), j.inf)
            lam = j.minimum(1., j.minimum(j.min(low), j.min(high)))
            obj = j.dot(r, r)
            ls0 = (j.array(0), lam, x, r, j.array(False))

            def lc(v): return (v[0] < 22) & (~v[4]) & (v[1] >= 1e-13) & j.all(j.isfinite(dx))

            def lb(v):
                k, l, xt, rt, ok = v
                xt = j.clip(x + l * dx, self.lo, self.hi); rt = self.residual(xt, bg, activity)
                ok = j.all(j.isfinite(rt)) & ((j.dot(rt, rt) <= obj * (1 - 1e-4 * l)) | (j.max(j.abs(rt)) < tol))
                return k + 1, j.where(ok, l, l * .5), xt, rt, ok
            ls = lax.while_loop(lc, lb, ls0)
            return it + 1, j.where(ls[4], ls[2], x), j.where(ls[4], ls[3], r), j.where(ls[4], 0, -1)
        it, x, r, code = lax.while_loop(cond, body, init)
        norm = j.max(j.abs(r)); ok = j.isfinite(norm) & (norm < tol)
        return x, ok, norm, it

    def solve(self, x, bg, activity, tol=1e-11, max_iter=42):
        with jax.default_device(self.device):
            args = [jax.device_put(j.asarray(v, dtype=j.float64), self.device) for v in (x, bg, activity)]
            return self.newton_batch(*args, float(tol), int(max_iter))


@jax.jit
def outer_linear_step(B, residual):
    """Quasi-Newton step dx that solves B dx = -residual."""
    return j.linalg.solve(B, -residual)


@jax.jit
def broyden_update(B, step, change):
    """Broyden rank-one update of the approximate Jacobian B after a step and the resulting change of the residual."""
    den = j.dot(step, step)
    return j.where(den > 1e-18, B + j.outer(change - B @ step, step) / j.maximum(den, 1e-300), B)
