"""Helpers shared by the scripts: heating profiles and result tables."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

from . import UNIFORM
from .model import DATA, TEMPERATURE_DATA_RANGE_C

ROW_KEYS = ('X', 'P_W', 'particle_Tmax_C', 'particle_Tmean_C', 'wall_max_C', 'outlet_gas_C', 'in_range', 'mean_Ni_pct',
            'bed_mean_Ni')


def profile(name):
    """'uniform', 'optimized' or a JSON file holding ten zone shares (a list, or {"q": [...]})."""
    if name == 'uniform':
        return list(UNIFORM)
    if name == 'optimized':
        return json.loads((DATA / 'optimized_heating.json').read_text())['q']
    doc = json.loads(Path(name).read_text())
    return list(map(float, doc['q'] if isinstance(doc, dict) else doc))


def node_row(state, **extra):
    """One table row of a state; t_h defaults to the model time of the state (pass t_h for the frozen solves
    of a constant-production history, whose model time is 0)."""
    s = state.summary
    row = dict(extra)
    row.setdefault('t_h', state.model_time_h)
    row.update({k: s[k] for k in ROW_KEYS if k in s})
    for i, v in enumerate(s['zone_Ni_pct']):
        row[f'Ni_Z{i + 1}_pct'] = v              # zone mean of the retained Ni area
    for i, v in enumerate(s['zone_T_C']):
        row[f'T_cat_Z{i + 1}_C'] = v             # zone mean catalyst temperature
    return row


def crossing_time(t0, y0, t1, y1, y=0.):
    """Time at which y(t) crosses y between two nodes, with y linear in log t (linear in t if t0 = 0)."""
    if y1 == y0:
        return t1
    a = (y - y0) / (y1 - y0)
    if t0 <= 0:
        return t0 + a * (t1 - t0)
    return math.exp(math.log(t0) + a * (math.log(t1) - math.log(t0)))


def range_margin_K(row):
    """Distance of the hottest and the coldest pellet shell from the ends of 483-682 C (negative outside)."""
    lo, hi = TEMPERATURE_DATA_RANGE_C
    return min(hi - row['particle_Tmax_C'], row['particle_Tmin_C'] - lo)


def run_length(nodes, P_limit_W):
    """Run length of a constant-production history (rows with t_h, P_W, in_range and shell temperatures).

    t_exit_h: time at which a pellet shell leaves 483-682 C, interpolated between the last node inside and
    the first node outside the range (None if the history never left it; t_exit_lower_bound_h is then its
    last node). t_limit_h: time at which the power reaches P_limit_W, interpolated in the same way.
    """
    nodes = [n for n in nodes if n.get('status', 'OK') == 'OK']
    out = dict(nodes=len(nodes), t_last_h=nodes[-1]['t_h'] if nodes else None)
    inside = [n for n in nodes if n['in_range']]
    out['t_last_in_range_h'] = inside[-1]['t_h'] if inside else None
    ex = next((i for i, n in enumerate(nodes) if not n['in_range']), None)
    if ex is None:
        out.update(left_range=False, t_exit_h=None, t_exit_lower_bound_h=out['t_last_in_range_h'], exit_side=None)
    else:
        b = nodes[ex]
        side = 'hottest shell above 682 C' if b['particle_Tmax_C'] > TEMPERATURE_DATA_RANGE_C[1] else 'coldest shell below 483 C'
        if ex == 0:
            out.update(left_range=True, exit_side=side, t_exit_lower_bound_h=None, t_exit_node_h=b['t_h'], t_exit_h=b['t_h'])
        else:
            a = nodes[ex - 1]
            out.update(left_range=True, exit_side=side, t_exit_lower_bound_h=a['t_h'], t_exit_node_h=b['t_h'],
                       t_exit_h=crossing_time(a['t_h'], range_margin_K(a), b['t_h'], range_margin_K(b)))
    hit = next((i for i, n in enumerate(nodes) if n['P_W'] >= P_limit_W), None)
    if hit is None:
        out.update(power_limit_reached=False, t_limit_h=None)
    elif hit == 0:
        out.update(power_limit_reached=True, t_limit_h=0.)
    else:
        a, b = nodes[hit - 1], nodes[hit]
        out.update(power_limit_reached=True, t_limit_h=crossing_time(a['t_h'], a['P_W'] - P_limit_W, b['t_h'], b['P_W'] - P_limit_W))
    out['P_last_W'] = nodes[-1]['P_W'] if nodes else None
    return out


def energy_kWh(nodes, t_end):
    """Electric energy up to t_end, trapezoid rule over the nodes of a constant-production history."""
    pts = [(n['t_h'], n['P_W']) for n in nodes if n.get('status', 'OK') == 'OK' and n['t_h'] <= t_end + 1e-9]
    return sum(.5 * (p0 + p1) * (t1 - t0) for (t0, p0), (t1, p1) in zip(pts, pts[1:])) / 1000.


def write_csv(path, rows):
    """Write a list of dicts as CSV; the columns are all keys in the order of first appearance."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def write_json(path, obj):
    """Write JSON through a temporary file, so that an interrupted run never leaves a partial file."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=float) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def read_json(path):
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding='utf-8'))
