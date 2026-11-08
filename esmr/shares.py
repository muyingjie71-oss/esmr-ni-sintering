"""Zone shares of the heating profile: normalization, heaters with fewer sections, and moves between zones."""
from __future__ import annotations

import numpy as np

# Sections of adjacent zones (zone indices 0-9 = Z1-Z10) for heaters with 2, 3 or 5 independently powered sections.
SECTIONS = {2: [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            3: [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]],
            5: [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]}


def normalize(q):
    """Non-negative shares summing to one within 1e-15 (the model requires |sum - 1| <= 1e-12).

    Negative values are clipped, values below 1e-9 set to zero, and the largest share takes up the rounding.
    """
    q = np.clip(np.asarray(q, float), 0, None); q[q < 1e-9] = 0
    q = q / q.sum(); i = int(np.argmax(q)); q[i] = 1. - (q.sum() - q[i])
    if not (abs(q.sum() - 1.) <= 1e-12 and (q >= 0).all()):
        raise ValueError('zone shares cannot be normalized')
    return q


def key(q, P_W=None, digits=6):
    """Dictionary key of a profile (shares rounded to `digits` decimals) and, optionally, a power (rounded to 1 mW)."""
    k = tuple(round(float(v), digits) for v in q)
    return k if P_W is None else (k, round(float(P_W), 3))


def sectioned(q, sections):
    """Profile of a heater with fewer sections: the shares of each section summed and split equally in it."""
    q = np.asarray(q, float); out = np.zeros(10)
    for g in sections:
        out[g] = q[g].sum() / len(g)
    return normalize(out)


def section_moves(center, sections, step):
    """Profiles with `step` of the total power moved between adjacent sections, in both directions."""
    c = np.asarray(center, float); tot = [c[g].sum() for g in sections]; out = []
    for i in range(len(sections) - 1):
        for a, b in ((i, i + 1), (i + 1, i)):
            if tot[a] >= step - 1e-12:
                t = list(tot); t[a] -= step; t[b] += step
                q = np.zeros(10)
                for g, s in zip(sections, t):
                    q[g] = s / len(g)
                out.append((f'move_S{a + 1}_to_S{b + 1}', normalize(q)))
    return out


def pair_moves(x, step):
    """Profiles with `step` of the total power moved between two adjacent zones, in both directions."""
    x = np.asarray(x, float); out = []
    for i in range(9):
        for a, b in ((i, i + 1), (i + 1, i)):
            if x[a] >= step - 1e-12:
                y = x.copy(); y[a] -= step; y[b] += step; out.append(normalize(y))
    return out
