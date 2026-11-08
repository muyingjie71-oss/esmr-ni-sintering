"""Ni sintering law and catalyst activity.

State: s = A_Ni / A_Ni,0, the fraction of the initial Ni surface area retained.
Following Sehested et al. (J. Catal. 223, 432-443, 2004) the sintering extent
x = s**-7 - 1 grows linearly in time at fixed conditions, at a rate

    dx/dt = x700/700 h * exp[E/R (1/873.15 K - 1/T)] * (F/F_ref)**m,
    F = p_H2O / p_H2**0.5,

fitted to the Ni area retained by catalyst No. 1 (22 wt% Ni/MgAl2O4) after
700 h at 40 bar and H2O/H2 = 2.5, 483-682 C. The activation energy changes at
600 C. Within a time step the rate of every pellet shell is evaluated from its
own temperature and partial pressures and held constant.

fit = 'retained_area'    baseline: continuous fit to the retained area at 700 h
                          (E = 144.9 / 403.3 kJ/mol below / above 600 C)
fit = 'published_slopes' Arrhenius fit of the rates with E fixed at 142 / 410 kJ/mol
fit = 'free_slopes'      Arrhenius fit of the rates with free E (144.5 / 427.8 kJ/mol)
The two rate fits switch at T_switch_C.

activity = activity_reference * s**activity_exponent
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .properties import R_SINTERING

DATA = Path(__file__).resolve().parent / 'data' / 'sintering_fits.json'
FITS = ('retained_area', 'published_slopes', 'free_slopes')


@lru_cache(maxsize=1)
def rate_fits():
    """Rate parameters of the alternative sintering fits (esmr/data/sintering_fits.json)."""
    return json.loads(DATA.read_text(encoding='utf-8'))


@dataclass(frozen=True)
class SinteringLaw:
    """One version of the sintering law and of the catalyst activity.

    fit                  one of FITS (module docstring)
    atmosphere_exponent  exponent m of F = p_H2O / p_H2**0.5
    activity_exponent    activity = activity_reference * s**activity_exponent
    T_switch_C           temperature at which the activation energy changes in the fits
                         'published_slopes' and 'free_slopes' (the baseline fit switches at 600 C)
    """
    fit: str = 'retained_area'
    atmosphere_exponent: float = 1.0
    activity_exponent: float = 1.0
    activity_reference: float = 1.0
    T_switch_C: float = 600.0

    def __post_init__(self):
        if self.fit not in FITS:
            raise ValueError('unknown sintering fit: ' + str(self.fit))
        if not np.isfinite(self.atmosphere_exponent) or not 0 <= self.atmosphere_exponent <= 1:
            raise ValueError('the exponent of F must lie within [0, 1]')
        vals = [self.activity_reference, self.activity_exponent]
        if not all(np.isfinite(x) and x > 0 for x in vals) or not np.isfinite(self.T_switch_C) or self.T_switch_C <= -273.15:
            raise ValueError('invalid sintering-law coefficients')


def _rate(law, T_K, pH2O_bar, pH2_bar):
    if law.fit == 'retained_area':
        T = np.asarray(T_K, float); w = np.asarray(pH2O_bar, float); h = np.asarray(pH2_bar, float)
        E = np.where(T <= 873.15, 144.89707097033659, 403.2969576359686) * 1000.
        log_rate = 6.176746838644072 - np.log(700.) + E / 8.314462618 * (1 / 873.15 - 1 / T)
        f = w / np.sqrt(np.maximum(h, 1e-12)); fref = (40 * 2.5 / 3.5) / np.sqrt(40 / 3.5)   # F of the experiment
        return np.exp(log_rate) * (f / fref) ** law.atmosphere_exponent
    fits = rate_fits()[law.fit]
    low = fits['low']; high = fits['high']
    T = np.asarray(T_K, float); w = np.asarray(pH2O_bar, float); h = np.asarray(pH2_bar, float)
    R = R_SINTERING
    lo = np.exp(low['lnA_rate_per_h'] - 1000 * low['E_kJ_mol'] / (R * T))
    hi = np.exp(high['lnA_rate_per_h'] - 1000 * high['E_kJ_mol'] / (R * T))
    f = w / np.sqrt(np.maximum(h, 1e-12))
    fref = (40 * 2.5 / 3.5) / np.sqrt(40 / 3.5)
    return np.where(T <= law.T_switch_C + 273.15, lo, hi) * (f / fref) ** law.atmosphere_exponent


def rate_x(law: SinteringLaw, T_K, pH2O_bar, pH2_bar):
    """Growth rate of the sintering extent x, per model hour."""
    T, w, h = np.broadcast_arrays(np.asarray(T_K, float), np.asarray(pH2O_bar, float), np.asarray(pH2_bar, float))
    if (not np.isfinite(T).all() or not np.isfinite(w).all() or not np.isfinite(h).all()
            or np.any(T <= 0) or np.any(w < 0) or np.any(h < 0)):
        raise ValueError('invalid shell temperature or partial pressure')
    rate = _rate(law, T, w, h)
    if not np.isfinite(rate).all() or np.any(rate < 0):
        raise ValueError('invalid sintering rate')
    return rate


def validate_area(s):
    """The Ni field as a float array, checked to lie in (0, 1]."""
    a = np.asarray(s, float)
    if not np.isfinite(a).all() or (a <= 0).any() or (a > 1).any():
        raise ValueError('retained Ni area must be finite with 0 < s <= 1')
    return a


def transformed_x_from_area(s_A):
    """Sintering extent x = s**-7 - 1 of a retained area s."""
    s = np.clip(np.asarray(s_A, dtype=float), 1e-12, None)
    return s ** -7 - 1.0


def area_from_transformed_x(x):
    """Retained area s = (1 + x)**(-1/7) of a sintering extent x."""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    return (1.0 + x) ** (-1.0 / 7.0)


def advance_area(s, dt_h, rate_x_per_h):
    """Retained Ni area after dt_h model hours at a constant rate of the sintering extent."""
    s = validate_area(s); r = np.asarray(rate_x_per_h, float)
    if not np.isfinite(dt_h) or dt_h < 0 or not np.isfinite(r).all() or (r < 0).any():
        raise ValueError('invalid time step or sintering rate')
    if dt_h == 0:
        return np.broadcast_arrays(s, r)[0].copy()
    out = area_from_transformed_x(transformed_x_from_area(s) + np.asarray(r, dtype=float) * dt_h)
    validate_area(out)
    if np.any(out > np.broadcast_arrays(s, out)[0] + 1e-14):
        raise ValueError('a non-negative sintering rate increased the Ni area')
    return out


def activity(s, law: SinteringLaw):
    """Relative catalyst activity of a retained-area field."""
    s = validate_area(s)
    return law.activity_reference * np.asarray(np.clip(np.asarray(s, dtype=float), 0.0, 1.0) ** float(law.activity_exponent))
