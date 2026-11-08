"""Gas thermochemistry and packed-bed transport properties.

Species order everywhere: CH4, H2O, H2, CO, CO2.

Two sets of molar masses are kept on purpose. The pellet equations use molar
masses built from the atomic masses, so that the element balances close
exactly. The bed correlations, the Ergun equation and the inlet mass flux use
the rounded tabulated values.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

# ---------------------------------------------------------------- thermochemistry (NumPy)
R = 8.31446261815324                     # J mol-1 K-1
R_SINTERING = 8.314462618                # value used in the sintering-law fit
NU = np.array([[-1., 0.], [-1., -1.], [3., 1.], [1., -1.], [0., 1.]], float)
ATOMS = np.array([[1., 0., 0., 1., 1.], [4., 2., 2., 0., 0.], [0., 1., 0., 1., 2.]], float)   # C, H, O
ATOMIC_MASSES = np.array([.0120107, .00100795, .0159994], float)
MW = ATOMS.T @ ATOMIC_MASSES             # kg mol-1, element-consistent

# NIST Shomate coefficients A..H (Chase 1998), low and high temperature segments.
LOW = np.array([
    [-.703029, 108.4773, -42.52157, 5.862788, .678565, -76.84376, 158.7163, -74.87310],
    [30.092, 6.832514, 6.793435, -2.534480, .082139, -250.8810, 223.3967, -241.8264],
    [33.066178, -11.363417, 11.432816, -2.772874, -.158558, -9.980797, 172.707974, 0.],
    [25.56759, 6.096130, 4.054656, -2.671301, .131021, -118.0089, 227.3665, -110.5271],
    [24.99735, 55.18696, -33.69137, 7.948387, -.136638, -403.6075, 228.2431, -393.5224]], float)
HIGH = np.array([
    [85.81217, 11.26467, -2.114146, .138190, -26.42221, -153.5327, 224.4143, -74.87310],
    [41.96426, 8.622053, -1.499780, .098119, -11.15764, -272.1797, 219.7809, -241.8264],
    [18.563083, 12.257357, -2.859786, .268238, 1.977990, -1.147438, 156.288133, 0.],
    [35.15070, 1.300095, -.205921, .013550, -3.282780, -127.8375, 231.7120, -110.5271],
    [58.16639, 2.720074, -.492289, .038844, -6.447293, -425.9186, 263.6125, -393.5224]], float)
SWITCH = np.array([1300., 1700., 1000., 1300., 1200.])


def shomate_h(T: float) -> np.ndarray:
    """Absolute molar enthalpy of the five species, J mol-1 (formation enthalpy included)."""
    c = np.where((T < SWITCH)[:, None], LOW, HIGH)
    t = T / 1000.
    return 1000. * (c[:, 0] * t + c[:, 1] * t * t / 2 + c[:, 2] * t ** 3 / 3 + c[:, 3] * t ** 4 / 4 - c[:, 4] / t + c[:, 5])


def _segment_h(c, T):
    """Enthalpy of the five species from one segment of the fits (rows of c), one temperature per species, J mol-1."""
    t = T / 1000.
    return 1000. * (c[:, 0] * t + c[:, 1] * t * t / 2 + c[:, 2] * t ** 3 / 3 + c[:, 3] * t ** 4 / 4 - c[:, 4] / t + c[:, 5])


# The two segments of a fit do not meet exactly at the switch temperature. Jump of each enthalpy there, high minus low
# segment, J mol-1: CH4 -12.9 (1300 K), H2O -2.93 (1700 K), H2 +0.180 (1000 K), CO -0.318 (1300 K), CO2 -3.17 (1200 K).
SHOMATE_JUMPS_J_MOL = _segment_h(HIGH, SWITCH) - _segment_h(LOW, SWITCH)


def shomate_h_many(T_K: np.ndarray) -> np.ndarray:
    """shomate_h for a 1-D array of temperatures, shape (n, 5)."""
    values = np.asarray(T_K, float)
    if values.ndim != 1:
        raise ValueError('shomate_h_many requires 1-D temperatures')
    c = np.where(values[:, None, None] < SWITCH[None, :, None], LOW[None, :, :], HIGH[None, :, :])
    t = values[:, None] / 1000.
    return 1000. * (c[:, :, 0] * t + c[:, :, 1] * t * t / 2 + c[:, :, 2] * t ** 3 / 3 +
                    c[:, :, 3] * t ** 4 / 4 - c[:, :, 4] / t + c[:, :, 5])


def shomate_cp(T: float) -> np.ndarray:
    """Molar heat capacity of the five species, J mol-1 K-1."""
    c = np.where((T < SWITCH)[:, None], LOW, HIGH)
    t = T / 1000.
    return c[:, 0] + c[:, 1] * t + c[:, 2] * t ** 2 + c[:, 3] * t ** 3 + c[:, 4] / t ** 2


# ---------------------------------------------------------------- tabulated data
MW_TABLE_VALUES = (.0160425, .0180153, .00201588, .0280101, .0440095)          # kg mol-1
Y_REFERENCE = (.25, .74, .01, 0., 0.)          # reference feed mole fractions (sets the reference mass flux)
TUBE_RADIUS_M = 0.015
PELLET_DIAMETER_M = .003
BED_VOID = 0.45
PELLET_EMISSIVITY = .60
SOLID_CONDUCTIVITY_W_MK = 5.
REFERENCE_LENGTH_M = 3.0                     # tube length for which the insulation loss is tabulated
REFERENCE_TIN_K = 673.15
REFERENCE_MASS_FLUX_KG_M2S = 0.7913163085727307
STEFAN_BOLTZMANN = 5.670374419e-8

# Gas mixture viscosity and conductivity at the mean reforming composition.
GAS_TABLE_T_K = (873.15, 950.0, 973.15, 1073.15, 1173.15)
GAS_TABLE_MU_PA_S = (3.419152627631647e-05, 3.671321482877019e-05, 3.7457114160989504e-05,
                     4.059324916359509e-05, 4.361101724933599e-05)
GAS_TABLE_K_W_MK = (0.1256113745988393, 0.1387745858355413, 0.1427574441135458,
                    0.1600231805077458, 0.1773143975167916)
# Heat lost through the insulation by a 3 m wall at uniform temperature, W.
WALL_LOSS_T_K = (950.0, 1073.15, 1173.15)
WALL_LOSS_W = (218.54162048364003, 277.12809475261525, 330.55716899970366)
# Axial conductance of the tube wall over one 0.3 m heating zone, W K-1.
WALL_AXIAL_CONDUCTANCE_W_K = 0.024697342666869584

MW_TABLE = jnp.array(MW_TABLE_VALUES, jnp.float64)
Y0 = jnp.array(Y_REFERENCE, jnp.float64)
MW0 = jnp.dot(Y0, MW_TABLE)
TG = jnp.asarray(np.array(GAS_TABLE_T_K))
MU = jnp.asarray(np.array(GAS_TABLE_MU_PA_S))
KG = jnp.asarray(np.array(GAS_TABLE_K_W_MK))
TI = jnp.asarray(np.array(WALL_LOSS_T_K))
QI = jnp.asarray(np.array(WALL_LOSS_W))


def linear_extrapolate(x, xp, fp):
    """np.interp, continued linearly beyond both ends of the table."""
    x = np.asarray(x, float); xp = np.asarray(xp, float); fp = np.asarray(fp, float)
    y = np.interp(x, xp, fp)
    return np.where(x < xp[0], fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]),
                    np.where(x > xp[-1], fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), y))


def gas_viscosity(T):
    """Mixture viscosity from the table, linear inter- and extrapolation, Pa s (NumPy)."""
    return np.maximum(1e-5, linear_extrapolate(T, TG, MU))


# ---------------------------------------------------------------- JAX heat capacity
_LOW_J = jnp.array(LOW, dtype=jnp.float64)
_HIGH_J = jnp.array(HIGH, dtype=jnp.float64)
_SWITCH_J = jnp.array(SWITCH, dtype=jnp.float64)


def _coeff(T):
    T = jnp.asarray(T, dtype=jnp.float64)
    mask = (T[..., None] < _SWITCH_J)
    return jnp.where(mask[..., None], _LOW_J, _HIGH_J)


def cp_species_J_molK(T):
    """Shomate heat capacities of CH4, H2O, H2, CO and CO2, J mol-1 K-1."""
    T = jnp.asarray(T, dtype=jnp.float64)
    c = _coeff(T); t = T[..., None] / 1000.0
    A, B, C, D, E = [c[..., i] for i in range(5)]
    return A + B * t + C * t * t + D * t ** 3 + E / (t * t)


def mixture_cp_J_kgK(T, y):
    """Heat capacity of a gas mixture with mole fractions y, J kg-1 K-1."""
    y = jnp.asarray(y, dtype=jnp.float64); y = y / jnp.maximum(jnp.sum(y), 1e-30)
    return jnp.dot(y, cp_species_J_molK(T)) / jnp.dot(y, MW_TABLE)


# ---------------------------------------------------------------- bed heat-transfer correlations (JAX)
def _interp_extrap(x, xp, fp):
    y = jnp.interp(x, xp, fp)
    low = fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]); high = fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return jnp.where(x < xp[0], low, jnp.where(x > xp[-1], high, y))


def gas_props(T, y):
    """Viscosity and thermal conductivity (tables in T), heat capacity of the mixture y and Prandtl number, for the bed correlations."""
    mu = jnp.maximum(1e-5, _interp_extrap(T, TG, MU)); kg = jnp.maximum(.04, _interp_extrap(T, TG, KG))
    cp = mixture_cp_J_kgK(T, y); Pr = jnp.clip(cp * mu / kg, .3, 2.)
    return mu, kg, cp, Pr


def comp_from_beta(beta):
    """Gas composition at CH4 conversion beta, with 55 % of the converted CH4 shifted to CO2."""
    e1 = jnp.clip(beta, 0, .999999) * Y0[0]; e2 = jnp.minimum(.55 * e1, jnp.minimum(e1, jnp.maximum(0., Y0[1] - e1 - 1e-12)))
    n = jnp.array([Y0[0] - e1, Y0[1] - e1 - e2, Y0[2] + 3 * e1 + e2, e1 - e2, e2]); n = jnp.maximum(n, 1e-14)
    return n / jnp.sum(n)


def zbs_static_k(eps, kgv, kpv, omega=7.26e-3):
    """Stagnant bed conductivity, Zehner-Bauer-Schlunder."""
    A = kpv / kgv; B = 1.25 * ((1 - eps) / eps) ** (10 / 9); den = 1 - B / A
    Gamma = 2 / den * (((A - 1) / den ** 2) * (B / A) * jnp.log(A / B) - (B - 1) / den - .5 * (B + 1))
    return ((1 - jnp.sqrt(1 - eps)) * kgv + jnp.sqrt(1 - eps) * (omega * A + (1 - omega) * Gamma) * kgv)


def zbs_radiation_k(T, ep, eb, kgv):
    """Radiative contribution to the bed conductivity."""
    B = 1.25 * ((1 - eb) / eb) ** (10 / 9); base = 4 * STEFAN_BOLTZMANN * T ** 3 * PELLET_DIAMETER_M; a = 2 / ep - 1
    return base * ((1 - jnp.sqrt(1 - eb)) * eb + (jnp.sqrt(1 - eb) / a) * ((B + 1) / B) *
                   (1 / (1 + 1 / (a * SOLID_CONDUCTIVITY_W_MK / jnp.maximum(base, 1e-20)))))


def thermal_coeffs(T, G):
    """Bed transport coefficients at temperature T and mass flux G.

    Returns (overall wall-to-bed U, gas-pellet h, effective radial conductivity,
    wall coefficient h_w, particle Reynolds number).
    """
    DP, EPS_B, KP, RI = PELLET_DIAMETER_M, BED_VOID, SOLID_CONDUCTIVITY_W_MK, TUBE_RADIUS_M
    y = comp_from_beta(.4); mu, kg, cp, Pr = gas_props(T, y); Re = jnp.maximum(1e-8, G * DP / mu)
    k0 = zbs_static_k(EPS_B, kg, KP); krb = k0 + zbs_radiation_k(T, PELLET_EMISSIVITY, EPS_B, kg); tau_b = 1.5 - .5 * EPS_B
    Pe = 1 / (EPS_B * tau_b / (Re * Pr) + 1 / 12.); kr = krb + kg * (Re * Pr) / Pe; N = 2 * RI / DP
    Nu0 = (1.3 + 5 / N) * (krb / kg); Nu_star = .3 * Pr ** (1 / 3) * Re ** .75; Nu_m = .054 * Pr * Re
    Nuw = Nu0 + 1 / (1 / jnp.maximum(Nu_star, 1e-12) + 1 / jnp.maximum(Nu_m, 1e-12)); hw = Nuw * kg / DP
    Uavg = 1 / (1 / hw + RI / (4 * kr)); Nu_gs = 2 + 1.1 * Re ** .6 * Pr ** (1 / 3); hgs = Nu_gs * kg / DP
    return Uavg, hgs, kr, hw, Re


def scaled_thermal_coeffs(kr=1.0, hw=1.0, hgs=1.0):
    """thermal_coeffs with the radial conductivity, wall coefficient and gas-pellet coefficient scaled."""
    if (kr, hw, hgs) == (1.0, 1.0, 1.0):
        return thermal_coeffs
    mult = (float(kr), float(hw), float(hgs))

    def coeffs(T, G):
        Uavg, h_gs, k_r, h_w, Re = thermal_coeffs(T, G)
        k_r, h_w, h_gs = k_r * mult[0], h_w * mult[1], h_gs * mult[2]
        return 1. / (1. / h_w + TUBE_RADIUS_M / (4. * k_r)), h_gs, k_r, h_w, Re
    return coeffs
