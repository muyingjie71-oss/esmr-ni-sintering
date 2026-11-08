"""NumPy finite-volume model of the catalytic pellet and its gas film.

This is the reference implementation of the pellet equations. The reactor
solves the pellets with the JAX version in pellet_jax.py; every accepted
reactor state is then checked against the residual of this module
(see Reactor.environment).

Pellet: steady spherical finite volumes, five species (CH4, H2O, H2, CO, CO2),
the three reactions of Xu and Froment (1989), Dusty-Gas transport with
Fuller-Schettler-Giddings binary diffusivities, Knudsen diffusion and Darcy
flow, and a total-enthalpy balance (species enthalpies include the formation
enthalpy, so no separate reaction heat appears). Unknowns per shell: five raw
mole fractions, pressure/1e6 Pa and temperature/1000 K.

Film: constant-pressure Maxwell-Stefan film integrated exactly with a matrix
exponential, and a film heat flux with the Stefan (Ackermann) correction. The
pellet surface temperature equals the local bed temperature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.optimize._numdiff import group_columns

from .properties import ATOMS, MW, R, gas_viscosity, shomate_cp, shomate_h, shomate_h_many

STOICH3 = np.array([[-1., 0., -1.], [-1., -1., -2.], [3., 1., 4.], [1., -1., 0.], [0., 1., 1.]], float)
FSG_DIFFUSION_VOLUMES = np.array([25.14, 13.1, 6.12, 18.0, 26.7], dtype=float)


def fuller_binary_gas_diffusivities(T_K, P_Pa, M_kg_mol, volumes=FSG_DIFFUSION_VOLUMES):
    """Binary diffusivities D_ij (m2 s-1), Fuller-Schettler-Giddings; shape broadcast(T, P) + (5, 5)."""
    T, P = np.broadcast_arrays(np.asarray(T_K, float), np.asarray(P_Pa, float))
    MWs = np.asarray(M_kg_mol, float)
    V = np.asarray(volumes, float)
    if MWs.shape != (5,) or V.shape != (5,):
        raise ValueError('five species molar masses and diffusion volumes required')
    if not (np.all(np.isfinite(T)) and np.all(np.isfinite(P)) and np.all(np.isfinite(MWs)) and np.all(np.isfinite(V))
            and np.min(T) > 0 and np.min(P) > 0 and np.min(MWs) > 0 and np.min(V) > 0):
        raise ValueError('finite positive T, P, M and diffusion volumes required')
    mg = MWs * 1000.
    redmass = 2. / (1. / mg[:, None] + 1. / mg[None, :])
    v_sum = (V[:, None] ** (1 / 3) + V[None, :] ** (1 / 3)) ** 2
    d_cm2_s = 0.00143 * T[..., None, None] ** 1.75 / ((P[..., None, None] / 101325.) * np.sqrt(redmass) * v_sum)
    Dij = d_cm2_s * 1.0e-4
    for i in range(5):
        Dij[..., i, i] = 0.
    return Dij


def xu_froment_numpy(T_K, p_bar):
    """Intrinsic rates of reactions 1-3 (mol per kg catalyst per s), partial pressures in bar."""
    T = np.asarray(T_K, float)
    p = np.asarray(p_bar, float)
    if p.shape[-1] != 5:
        raise ValueError('five partial pressures needed')
    h2 = np.maximum(p[..., 2], 1e-4)        # regularization for vanishing H2
    K1 = np.exp(-26830 / T + 30.114)
    K2 = np.exp(4400 / T - 4.036)
    K3 = K1 * K2
    k1 = 1.173611111e15 * np.exp(-240100 / (R * T))
    k2 = 5.430555556e5 * np.exp(-67130 / (R * T))
    k3 = 2.833333333e14 * np.exp(-243900 / (R * T))
    kco = 40.91 * np.exp((70650 / R) * (1 / T - 1 / 648))
    kh2 = .0296 * np.exp((82900 / R) * (1 / T - 1 / 648))
    kch4 = .179 * np.exp((38280 / R) * (1 / T - 1 / 823))
    kh2o = .4152 * np.exp((-88680 / R) * (1 / T - 1 / 823))
    ch4, h2o, h2, co, co2 = np.moveaxis(p, -1, 0)
    den = 1 + kco * co + kh2 * h2 + kch4 * ch4 + kh2o * h2o / h2
    r1 = k1 * (ch4 * h2o - co * h2 ** 3 / K1) / (h2 ** 2.5 * den ** 2)
    r2 = k2 * (co * h2o - co2 * h2 / K2) / (h2 * den ** 2)
    r3 = k3 * (ch4 * h2o * h2o - co2 * h2 ** 4 / K3) / (h2 ** 3.5 * den ** 2)
    return np.stack((r1, r2, r3), axis=-1)


@dataclass(frozen=True)
class PoreParameters:
    """Pellet radius, density, pore structure and effective thermal conductivity."""
    pellet_radius_m: float = .0015
    rho_cat_kg_per_m3_pellet: float = 2355.
    porosity: float = .60
    tortuosity: float = 3.0
    pore_radius_m: float = 1.0e-7
    permeability_m2: float = 2.5e-16
    k_effective_W_m_K: float = 3.0

    def __post_init__(self):
        if min(self.pellet_radius_m, self.rho_cat_kg_per_m3_pellet, self.porosity, self.tortuosity,
               self.pore_radius_m, self.k_effective_W_m_K) <= 0:
            raise ValueError('positive pore and thermal quantities required')
        if self.porosity >= 1 or self.permeability_m2 < 0:
            raise ValueError('invalid porosity or permeability')

    def at(self, T_K, P_Pa):
        """Effective binary diffusivities, Knudsen diffusivities and gas viscosity."""
        T = np.asarray(T_K, float)
        P = np.asarray(P_Pa, float)
        T, P = np.broadcast_arrays(T, P)
        bulkD = fuller_binary_gas_diffusivities(T, P, MW)
        Dij = (self.porosity / self.tortuosity) * bulkD
        DK = (2 / 3) * self.pore_radius_m * np.sqrt(8 * R * T[..., None] / (np.pi * MW)) * self.porosity / self.tortuosity
        return Dij, DK, gas_viscosity(T)


def prepared_coefficients(p: PoreParameters):
    """Temperature- and pressure-independent factors of the binary and Knudsen diffusivities."""
    mg = MW * 1000.
    redmass = 2. / (1. / mg[:, None] + 1. / mg[None, :])
    v_sum = (FSG_DIFFUSION_VOLUMES[:, None] ** (1 / 3) + FSG_DIFFUSION_VOLUMES[None, :] ** (1 / 3)) ** 2
    pair = 0.00143 * 101325. * 1e-4 / (np.sqrt(redmass) * v_sum)
    pair *= p.porosity / p.tortuosity
    np.fill_diagonal(pair, 0.)
    knudsen = (2. / 3.) * p.pore_radius_m * np.sqrt(8. * R / (np.pi * MW)) * p.porosity / p.tortuosity
    return np.ascontiguousarray(pair, float), np.ascontiguousarray(knudsen, float)


@dataclass(frozen=True)
class FilmParameters:
    """Thickness and heat-transfer coefficient of the gas film around a pellet, and the pellet-to-pellet contact coefficient."""
    film_thickness_m: float = 3e-5
    film_h_W_m2K: float = 800.
    solid_contact_h_W_m2K: float = 2500.


@dataclass(frozen=True)
class SurfaceState:
    """Gas at the outer surface of a pellet: temperature, pressure and mole fractions of CH4, H2O, H2, CO and CO2."""
    T_K: float
    P_Pa: float
    y: np.ndarray

    def __post_init__(self):
        yy = np.asarray(self.y, float)
        if (yy.shape != (5,) or np.any(yy < 0) or not np.isclose(np.sum(yy), 1, rtol=0, atol=1e-10)
                or self.P_Pa < 1e4 or not (300 < self.T_K < 1800)):
            raise ValueError('invalid five-component surface state')


@dataclass(frozen=True)
class GasBedState:
    """Bulk gas and bed temperature seen by one pellet; h_film from the bed correlation."""
    gas_T_K: float
    solid_T_K: float
    pressure_Pa: float
    gas_y: np.ndarray
    film_thickness_m: float
    film_h_W_m2K: float
    solid_contact_h_W_m2K: float

    def __post_init__(self):
        y = np.asarray(self.gas_y, float)
        if y.shape != (5,) or (y < 0).any() or abs(sum(y) - 1) > 1e-10:
            raise ValueError('five gas mole fractions, non-negative, summing to one required')
        if not (300 < self.gas_T_K < 1800 and 300 < self.solid_T_K < 1800):
            raise ValueError('invalid gas or solid temperature')
        if self.pressure_Pa < 1e4 or min(self.film_thickness_m, self.film_h_W_m2K, self.solid_contact_h_W_m2K) <= 0:
            raise ValueError('invalid film or contact coefficients')


@dataclass
class PelletCheck:
    """Result of the reference check of one pellet solution: pass flag, shell profiles, surface fluxes and the largest residuals."""
    passed: bool
    T_cells_K: np.ndarray
    P_cells_Pa: np.ndarray
    y_cells: np.ndarray
    surface_species_flux_mol_m2s: np.ndarray
    q_total_outer_W_m2: float
    atom_residual_mol_m2_s: float
    max_local_species_residual_mol_m3s: float
    max_local_energy_residual_W_m3: float


class PelletFV:
    """Spherical finite volumes; the shells are refined towards the surface (dr ~ 1 - (1 - u)^2)."""

    def __init__(self, p: PoreParameters = PoreParameters(), n: int = 18, kinetic: Callable = xu_froment_numpy):
        if n < 3:
            raise ValueError('at least 3 shells needed')
        self.p = p; self.n = int(n); self.kinetic = kinetic
        uu = np.linspace(0., 1., n + 1)
        self.edges = p.pellet_radius_m * (2 * uu - uu ** 2)
        self.centers = ((self.edges[:-1] ** 3 + self.edges[1:] ** 3) / 2) ** (1 / 3)
        self.vol = (self.edges[1:] ** 3 - self.edges[:-1] ** 3) / 3      # per 4 pi
        self.area = self.edges ** 2                                     # per 4 pi
        self.dr = np.r_[np.diff(self.centers), self.edges[-1] - self.centers[-1]]
        pattern = sparse.lil_matrix((7 * n, 7 * n), dtype=bool)
        for j in range(n):
            for k in range(max(0, j - 1), min(n, j + 2)):
                pattern[j * 7:(j + 1) * 7, k * 7:(k + 1) * 7] = True
        self.jac_pattern = pattern.tocsr()

    def initial(self, b: SurfaceState):
        z = np.tile(np.r_[np.asarray(b.y, float).copy(), b.P_Pa / 1e6, b.T_K / 1000], (self.n, 1))
        z[:, :5] /= z[:, :5].sum(axis=1)[:, None]
        return z.ravel()

    def unpack(self, z):
        s = np.asarray(z, float).reshape(self.n, 7)
        yraw = s[:, :5]
        y = yraw / yraw.sum(axis=1, keepdims=True)
        P = s[:, 5] * 1e6
        T = s[:, 6] * 1000
        return y, P, T, yraw

    def fv_components(self, z, b: SurfaceState, activity: float):
        y, P, T, yraw = self.unpack(z)
        xL = y; xR = np.r_[y[1:], np.asarray(b.y, float)[None, :]]
        PL = P; PR = np.r_[P[1:], b.P_Pa]; TL = T; TR = np.r_[T[1:], b.T_K]
        dxdr = (xR - xL) / self.dr[:, None]
        dPdr = (PR - PL) / self.dr
        dTdr = (TR - TL) / self.dr
        xf = (xL + xR) / 2
        Pf = (PL + PR) / 2
        Tf = (TL + TR) / 2
        Dij, DK, mu = self.p.at(Tf, Pf)
        A = np.zeros((self.n, 5, 5))         # Dusty-Gas matrix of each face
        for i in range(5):
            A[:, i, i] = 1 / DK[:, i] + np.sum(xf / np.where(Dij[:, i, :] > 0, Dij[:, i, :], np.inf), axis=1)
            for j in range(5):
                if i != j:
                    A[:, i, j] = -xf[:, i] / Dij[:, i, j]
        driv = (-Pf[:, None] * dxdr - xf * (1 + self.p.permeability_m2 * Pf[:, None] / (mu[:, None] * DK)) * dPdr[:, None]) / (R * Tf[:, None])
        N = np.linalg.solve(A, driv[..., None])[..., 0]
        rate = self.kinetic(T, (P / 1e5)[:, None] * y)
        S = (self.p.rho_cat_kg_per_m3_pellet * activity) * (rate @ STOICH3.T)
        N_inner = np.r_[np.zeros((1, 5)), N[:-1]]
        area_inner = self.area[:-1, None]
        area_outer = self.area[1:, None]
        species = (area_outer * N - area_inner * N_inner) / self.vol[:, None] - S
        gas_h = shomate_h_many(Tf)
        qN = np.sum(N * gas_h, axis=1)
        qC = -self.p.k_effective_W_m_K * dTdr
        q = qN + qC
        q_inner = np.r_[0., q[:-1]]
        energy = (self.area[1:] * q - self.area[:-1] * q_inner) / self.vol
        norm = yraw.sum(axis=1) - 1.
        return species, energy, norm, N, q, qN, qC, S, rate, y, P, T

    def check(self, x, b, activity, residual_norm):
        """Element balance, source integral and surface energy flux of a solved pellet."""
        sp, ee, nn, N, q, qN, qC, S, rate, y, P, T = self.fv_components(x, b, activity)
        surf = N[-1]
        source_integrated = np.sum(S * self.vol[:, None], axis=0) / self.area[-1]
        atoms = ATOMS @ surf
        good = (residual_norm < 5.e-7 and np.all(y >= 0) and np.min(P) > 0 and np.max(np.abs(atoms)) < 1e-7 and
                np.max(np.abs(surf - source_integrated)) < 5e-7 and abs(q[-1]) < 1e-2)
        return PelletCheck(bool(good), T, P, y, surf, float(q[-1]), float(np.max(np.abs(atoms))),
                           float(np.max(np.abs(sp))), float(np.max(np.abs(ee))))


def stefan_factor(Pe):
    """Pe / (1 - exp(-Pe)), stable for small Pe."""
    Pe = float(Pe)
    if abs(Pe) < 1e-4:
        return 1 + Pe / 2 + Pe * Pe / 12 - Pe ** 4 / 720
    return Pe / (-np.expm1(-Pe))


def gas_film_enthalpy_flux(N_out, Ts, Tg, hfilm):
    """Total enthalpy flux into the bulk gas, W m-2 (formation enthalpies included)."""
    Tm = .5 * (Ts + Tg)
    cp = shomate_cp(Tm)
    Pe = np.asarray(N_out) @ cp / hfilm
    if abs(Pe) > 80:
        raise ValueError('outside the stable range of the Stefan film correction')
    return float(np.asarray(N_out) @ shomate_h(Tg) + hfilm * (Ts - Tg) * stefan_factor(Pe))


class PelletWithFilm:
    """Pellet unknowns plus five surface mole fractions and the surface temperature/1000 K."""

    def __init__(self, p: PoreParameters = PoreParameters(), n=24):
        self.internal = PelletFV(p, n)
        self.n = n
        k = 7 * n
        pattern = sparse.lil_matrix((k + 6, k + 6), dtype=bool)
        pattern[:k, :k] = self.internal.jac_pattern
        pattern[k - 7:k, k:k + 6] = True
        pattern[k:k + 6, k - 7:k + 6] = True
        self.pattern = pattern.tocsr()
        self.column_groups = group_columns(self.pattern)

    def unpack(self, z, bg: GasBedState):
        k = self.n * 7
        xsraw = z[k:k + 5]
        xs = xsraw / np.sum(xsraw)
        Ts = z[k + 5] * 1e3
        return SurfaceState(Ts, bg.pressure_Pa, xs), xsraw

    def initial(self, bg: GasBedState):
        Ts0 = bg.solid_T_K
        b = SurfaceState(Ts0, bg.pressure_Pa, np.asarray(bg.gas_y, float))
        return np.r_[self.internal.initial(b), np.asarray(bg.gas_y, float), Ts0 / 1e3]

    def evaluate(self, z, bg: GasBedState, activity):
        """Residual of the pellet, film and surface equations, and the film fluxes."""
        b, xsraw = self.unpack(z, bg)
        sp, en, norm, N, q, qN, qC, S, rate, y, P, T = self.internal.fv_components(z[:self.n * 7], b, activity)
        internal_res = np.column_stack((sp / 1e4, en / 1e9, norm)).ravel()
        Ns = N[-1]
        pellet_surface_q = float(q[-1])
        xs = b.y; yb = np.asarray(bg.gas_y, float)
        Tf = (bg.gas_T_K + b.T_K) / 2
        Dij_pore, *_ = self.internal.p.at(Tf, bg.pressure_Pa)
        Db = np.asarray(Dij_pore) / (self.internal.p.porosity / self.internal.p.tortuosity)
        c = bg.pressure_Pa / (R * Tf)
        film_M = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                if i != j:
                    film_M[i, j] = Ns[i] / Db[i, j]
                    film_M[i, i] -= Ns[j] / Db[i, j]
        yb_from_surface = expm(-bg.film_thickness_m / c * film_M) @ xs
        fm = (yb_from_surface - yb)[:4]
        gasq = gas_film_enthalpy_flux(Ns, b.T_K, bg.gas_T_K, bg.film_h_W_m2K)
        # The pellet surface is at the bed temperature; the heat drawn from the bed is an output.
        contactq = gasq - pellet_surface_q
        boundary_res = (b.T_K - bg.solid_T_K) / 1000.
        allres = np.r_[internal_res, fm, np.sum(xsraw) - 1, boundary_res]
        return allres, dict(gasq=gasq, contactq=contactq, species_film_resid=np.max(np.abs(fm)),
                            heatfilm_resid=pellet_surface_q, film_surface=b)
