"""Tube geometry, inlet flows and the ten-zone heating profile."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np

from . import properties as pr


@dataclass(frozen=True)
class Geometry:
    """Tube, packed bed and inlet of one operating point, in SI units. The defaults are building blocks of tube(); Model sets the conditions of a run."""
    length_m: float = pr.REFERENCE_LENGTH_M
    nx: int = 60                       # axial cells
    nr: int = 6                        # radial rings
    ri_m: float = pr.TUBE_RADIUS_M
    dp_m: float = pr.PELLET_DIAMETER_M
    bed_void: float = pr.BED_VOID
    mass_flux_kg_m2s: float = pr.REFERENCE_MASS_FLUX_KG_M2S * 7.4
    Pin_Pa: float = 18e5
    Tin_K: float = pr.REFERENCE_TIN_K
    ambient_K: float = 298.15
    inlet_molar_flows: tuple = ()      # mol s-1; empty: reference composition at the mass flux

    @property
    def dz(self): return self.length_m / self.nx

    @property
    def area(self): return np.pi * self.ri_m ** 2

    @cached_property
    def radius_edges(self): return np.linspace(0, self.ri_m, self.nr + 1)

    @cached_property
    def radius_centers(self): return .5 * (self.radius_edges[1:] + self.radius_edges[:-1])

    @cached_property
    def annulus_areas(self): return np.pi * np.diff(self.radius_edges ** 2)

    @property
    def wall_area_cell(self): return 2 * np.pi * self.ri_m * self.dz

    @property
    def particle_area_density(self): return 6. * (1 - self.bed_void) / self.dp_m

    @property
    def F0(self):
        if self.inlet_molar_flows:
            return np.array(self.inlet_molar_flows, float)
        return self.mass_flux_kg_m2s * self.area / float(pr.MW0) * np.asarray(pr.Y0)


def tube(**overrides) -> Geometry:
    """Geometry with the tabulated tube radius, pellet diameter and bed void fraction."""
    args = dict(length_m=pr.REFERENCE_LENGTH_M, nx=60, nr=6, ri_m=pr.TUBE_RADIUS_M, dp_m=pr.PELLET_DIAMETER_M,
                bed_void=pr.BED_VOID, mass_flux_kg_m2s=pr.REFERENCE_MASS_FLUX_KG_M2S * 7.4,
                Pin_Pa=18e5, Tin_K=pr.REFERENCE_TIN_K)
    args.update(overrides)
    g = Geometry(**args)
    if g.nx < 1 or g.nr < 1 or g.length_m <= 0 or not 0 < g.bed_void < 1:
        raise ValueError('invalid reactor geometry')
    for name, target in [('ri_m', pr.TUBE_RADIUS_M), ('dp_m', pr.PELLET_DIAMETER_M), ('bed_void', pr.BED_VOID)]:
        if not np.isclose(getattr(g, name), target, rtol=0, atol=1e-14):
            raise ValueError(f'the transport correlations are tabulated for {name} = {target}')
    return g


def with_inlet(g: Geometry, F0, Tin_K=None, Pin_Pa=None) -> Geometry:
    """Same tube with prescribed inlet molar flows; the mass flux follows from the flows."""
    f = np.asarray(F0, float)
    if f.shape != (5,) or (f < 0).any() or f[0] <= 0:
        raise ValueError('invalid inlet composition')
    mass_flux = (g.mass_flux_kg_m2s if np.array_equal(f, g.F0)
                 else float(f @ np.asarray(pr.MW_TABLE) / g.area))
    return replace(g, mass_flux_kg_m2s=mass_flux, Tin_K=float(g.Tin_K if Tin_K is None else Tin_K),
                   Pin_Pa=float(g.Pin_Pa if Pin_Pa is None else Pin_Pa), inlet_molar_flows=tuple(f))


@dataclass(frozen=True)
class Heating:
    """Total electric power and its shares over ten equal heating zones."""
    total_W: float
    fractions: tuple

    def __post_init__(self):
        f = np.asarray(self.fractions, float)
        if (not np.isfinite(self.total_W) or self.total_W <= 0 or f.shape != (10,) or not np.isfinite(f).all()
                or (f < 0).any() or not np.isclose(f.sum(), 1, rtol=0, atol=1e-12)):
            raise ValueError('positive total power and 10 non-negative shares summing to 1 required')
        object.__setattr__(self, 'fractions', tuple(float(x) for x in f))

    def cells(self, g: Geometry):
        """Electric power of each axial wall cell, W."""
        ze = np.linspace(0, g.length_m, 11); ce = np.linspace(0, g.length_m, g.nx + 1)
        overlap = np.maximum(0, np.minimum(ce[1:, None], ze[None, 1:]) - np.maximum(ce[:-1, None], ze[None, :-1]))
        return overlap @ (np.asarray(self.fractions) * self.total_W / (g.length_m / 10))
