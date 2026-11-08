"""Ageing histories, frozen-field tests and power control.

All functions take a `folder`: every accepted state is written there and is
reused when the call is repeated, so long runs can be resumed.
"""
from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from . import BASE_POWER_W, TIMES_H, UNIFORM
from . import sintering as sn
from .model import GRIDS, INITIAL_STATE, Model, State, is_complete, load_state

DEFAULTS = dict(Tin_C=550., Pin_bar=6., length_m=1.)
# Longer runs continue beyond t_ref with a factor of 4/3 between nodes, up to 20,628 h.
LONG_TIMES_H = TIMES_H + [TIMES_H[-1] * (4 / 3) ** k for k in range(1, 12)]


def half_step(times):
    """The time grid with a node added halfway between every two nodes (time-step test, Supplementary Note 3)."""
    out = []
    for a, b in zip(times[:-1], times[1:]):
        out += [a, .5 * (a + b)]
    return out + [times[-1]]


def double_step(times=TIMES_H):
    """Nodes 0, 2, 4, 6, 8 and 11 of the 12-node grid, about twice the time step (time-step test, Supplementary Note 3)."""
    return [times[i] for i in (0, 2, 4, 6, 8, 11)]


def make_model(P_W, q, f=1., SC=2.96, grid='main', precision='main', variant=None, **kw):
    """Model at the baseline inlet; f scales the baseline feed (2.40 Nm3/h CH4)."""
    args = dict(DEFAULTS); args.update(kw)
    return Model(P_W, q=list(map(float, q)), feed_factor=.5 * f, SC=SC, grid=grid, precision=precision, variant=variant, **args)


def _initial():
    return load_state(INITIAL_STATE)


def _seed(seed):
    if seed is None:
        return _initial()
    return seed if isinstance(seed, State) else load_state(seed)


def _fresh_folder(folder):
    """Move an incomplete attempt aside; a failed solve is never resumed in place."""
    folder = Path(folder)
    if folder.exists() and not is_complete(folder):
        shutil.move(str(folder), str(folder.with_name(folder.name + f'_incomplete_{int(time.time())}')))
    return folder


def solve_frozen(model: Model, health, seed, folder=None) -> State:
    """Steady state with a prescribed Ni field (no ageing)."""
    if folder is not None and is_complete(folder):
        return load_state(folder)
    if folder is not None:
        _fresh_folder(folder)
    return model.solve(np.asarray(health, np.float64), _seed(seed), model_time_h=0., folder=folder)


def history_fixed_power(model: Model, times=TIMES_H, seed=None, folder=None, stop=None):
    """Fixed-power ageing history: fresh bed at t = 0, then one Ni update and steady solve per node.

    stop(state) -> True ends the history after that node.
    """
    nodes, parent = [], None
    for k, t in enumerate(times):
        sub = None if folder is None else Path(folder) / f'node{k:02d}'
        if sub is not None and is_complete(sub):
            state = load_state(sub)
        else:
            if sub is not None:
                _fresh_folder(sub)
            if k == 0:
                state = model.solve(np.ones((model.nx, model.nr)), _seed(seed), model_time_h=0., folder=sub)
            else:
                h = model.advance(parent, t - times[k - 1])
                state = model.solve(h, parent, model_time_h=t, folder=sub)
        nodes.append(state); parent = state
        if stop is not None and stop(state):
            break
    return nodes


def step_to_mean_area(model: Model, state: State, target):
    """Time step after which the bed-averaged Ni area falls to `target`, with the shell rates of `state` held."""
    env = state.environment
    rate = sn.rate_x(model.law, env['T_K'], env['pH2O_bar'], env['pH2_bar'])

    def mean_at(dt):
        local = sn.advance_area(np.asarray(state.health)[:, :, None], dt, rate)
        return model.mean_health(np.sum(local * env['shell_weights'][None, None, :], axis=-1))
    dt = .00001
    while mean_at(dt) > target:
        dt *= 2.
        if dt > 1e9:
            raise RuntimeError('no time step reaches the requested Ni area')
    root = float(brentq(lambda t: mean_at(t) - target, 0., dt, xtol=1e-13, rtol=1e-12))
    if root <= 0 or not math.isfinite(root):
        raise RuntimeError('invalid ageing step')
    return root


def history_to_mean_area(model: Model, targets=(.95, .90, .85, .80, .75, .70, .65, .60, .55, .50), seed=None, folder=None):
    """Fixed-power history with adaptive steps: each step lowers the bed-averaged Ni area to the next target."""
    sub = None if folder is None else Path(folder) / 'node00'
    if sub is not None and is_complete(sub):
        parent = load_state(sub)
    else:
        if sub is not None:
            _fresh_folder(sub)
        parent = model.solve(np.ones((model.nx, model.nr)), _seed(seed), model_time_h=0., folder=sub)
    nodes = [parent]
    for k, target in enumerate(targets, start=1):
        sub = None if folder is None else Path(folder) / f'node{k:02d}'
        if sub is not None and is_complete(sub):
            state = load_state(sub)
        else:
            if sub is not None:
                _fresh_folder(sub)
            dt = step_to_mean_area(model, parent, float(target))
            state = model.solve(model.advance(parent, dt), parent, model_time_h=parent.model_time_h + dt, folder=sub)
        nodes.append(state); parent = state
    return nodes


def next_power(pts, xt, p0):
    """Safeguarded secant step towards conversion xt; pts = [(P, X), ...]."""
    lo = [p for p in pts if p[1] < xt]; hi = [p for p in pts if p[1] >= xt]
    (pa, xa), (pb, xb) = pts[-2], pts[-1]
    slope = (xb - xa) / (pb - pa) if pb != pa else 0.
    cand = pb + (xt - xb) / slope if slope > 0 else None
    if lo and hi:
        a = max(lo, key=lambda p: p[0]); b = min(hi, key=lambda p: p[0])
        left, right = min(a[0], b[0]), max(a[0], b[0])
        if cand is None or not (left < cand < right) or abs(cand - pb) > .5 * (right - left):
            lin = a[0] + (xt - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            cand = .5 * (lin + .5 * (left + right)) if cand is None or not (left < cand < right) else lin
        return cand
    if cand is None:
        return pb * 1.03
    return pb + max(min(cand - pb, .15 * pb), -.10 * pb)


def power_for_conversion(q, health, target, P_start, seed, folder=None, tol=.002, max_solves=8, **model_kw):
    """Total power at which a frozen Ni field gives the CH4 conversion `target` (within tol, % points).

    The first solve is at P_start. 'tried' holds every solved state in order, 'folder' the folder of the
    accepted state (iter01, iter02, ... under `folder`).
    """
    pts, tried = [], []
    P = float(P_start); cur_seed = _seed(seed)
    for n in range(max_solves):
        sub = None if folder is None else Path(folder) / f'iter{n + 1:02d}'
        try:
            s = solve_frozen(make_model(P, q, **model_kw), health, cur_seed, sub)
        except Exception as exc:  # noqa: BLE001  (a failed solve ends the search, it is not a data point)
            return dict(status='SOLVE_FAILED', error=repr(exc), P_W=None, points=pts, solves=n, tried=tried)
        X = s.summary['X']
        pts.append((P, X)); tried.append(s)
        if abs(X - target) <= tol:
            return dict(status='OK', P_W=P, state=s, points=pts, solves=n + 1, tried=tried,
                        folder=None if sub is None else str(sub))
        cur_seed = s
        P = P_start * (1 + 1.8 * (target - X) / target) if n == 0 else next_power(pts, target, P_start)
        if not (P > 0 and math.isfinite(P)):
            break
    return dict(status='ROOT_FAILED', P_W=None, points=pts, solves=len(pts), state=tried[-1] if tried else None,
                tried=tried)


ROOT_TOL_RELAXED = .01     # % points, accepted only after the retry from the initial state has failed as well


def power_with_fallback(q, health, target, P_start, seed, folder=None, tol=.002, max_solves=8, **model_kw):
    """power_for_conversion with two fallbacks, used by the long histories and the operating-point studies.

    If the search fails, it is repeated once from the shipped initial state (in <folder>/retry_cold). If that
    fails too, the solved state closest to the target is accepted when its conversion lies within
    ROOT_TOL_RELAXED of the target. 'root_mode' is 'primary', 'cold_start', 'relaxed' or 'failed';
    'first_state' is the first solve of the primary search, at P_start.
    """
    first = power_for_conversion(q, health, target, P_start, seed, folder, tol, max_solves, **model_kw)
    head = first['tried'][0] if first['tried'] else None
    if first['status'] == 'OK':
        return dict(first, root_mode='primary', first_state=head)
    cold = power_for_conversion(q, health, target, P_start, None, None if folder is None else Path(folder) / 'retry_cold',
                                tol, max_solves, **model_kw)
    attempts = dict(first=dict(status=first['status'], points=first['points'], error=first.get('error')),
                    cold=dict(status=cold['status'], points=cold['points'], error=cold.get('error')))
    if cold['status'] == 'OK':
        return dict(cold, root_mode='cold_start', first_state=head, attempts=attempts)
    candidates = [(s, None if folder is None else str(Path(folder) / f'iter{i + 1:02d}')) for i, s in enumerate(first['tried'])]
    candidates += [(s, None if folder is None else str(Path(folder) / 'retry_cold' / f'iter{i + 1:02d}'))
                   for i, s in enumerate(cold['tried'])]
    best = None
    for s, sub in candidates:
        if best is None or abs(s.summary['X'] - target) < abs(best[0].summary['X'] - target):
            best = (s, sub)
    if best is not None and abs(best[0].summary['X'] - target) <= ROOT_TOL_RELAXED:
        s = best[0]
        return dict(status='OK', P_W=s.summary['P_W'], state=s, points=first['points'] + cold['points'], solves=None,
                    folder=best[1], root_mode='relaxed', dX_pp=s.summary['X'] - target, first_state=head,
                    attempts=attempts)
    return dict(status='ROOT_FAILED', P_W=None, state=None, points=first['points'] + cold['points'], solves=None,
                root_mode='failed', first_state=head, attempts=attempts, error=first.get('error') or cold.get('error'))


def history_constant_production(q, target, times=TIMES_H, P_start=BASE_POWER_W, seed=None, folder=None, fallback=True,
                                **model_kw):
    """Constant production: at every node the power is set so that X equals `target`; the Ni field is then
    advanced with that power held over the step. The run stops at the first node outside 483-682 C, which
    is kept. With fallback (default) the power search of a node is retried as in power_with_fallback."""
    nx, nr, _ = GRIDS[model_kw.get('grid', 'main')]
    h = np.ones((nx, nr)); P_prev = float(P_start); out = []; seed = _seed(seed)
    find = power_with_fallback if fallback else power_for_conversion
    for k, t in enumerate(times):
        sub = None if folder is None else Path(folder) / f'node{k:02d}'
        r = find(q, h, target, P_prev, seed, sub, **model_kw)
        if r['status'] != 'OK':
            out.append(dict(node=k, t_h=t, status=r['status'], error=r.get('error'), root_mode=r.get('root_mode')))
            break
        state = r['state']
        out.append(dict(node=k, t_h=t, status='OK', P_W=r['P_W'], root_iterations=r['solves'], root_points=r['points'],
                        root_mode=r.get('root_mode', 'primary'), state=state, state_folder=r.get('folder'),
                        **{k2: v for k2, v in state.summary.items() if k2 != 'P_W'}))
        if not state.summary['in_range'] or k == len(times) - 1:
            break
        h = make_model(r['P_W'], q, **model_kw).advance(state, times[k + 1] - t)
        P_prev = r['P_W']; seed = state
    return out


def uniform_test(health, seed, folder=None, grid='main', **model_kw):
    """CH4 conversion of a frozen Ni field under uniform heating at the baseline power."""
    return solve_frozen(make_model(BASE_POWER_W, UNIFORM, grid=grid, **model_kw), health, seed, folder).summary['X']


def evaluate_profile(q, baseline, folder=None, grid='main', times=TIMES_H, utest=True, **model_kw):
    """Fixed-power history of heating profile q with the thermal constraints of the profile search,
    and J = X_U(s_q) - X_U(s_U) when the history is feasible.

    At every node the hottest pellet shell and the hottest wall point may exceed those of the uniformly
    heated history (`baseline`) by at most baseline['margin_K'], and every shell must stay within 483-682 C.
    """
    margin = baseline['margin_K']
    model = make_model(BASE_POWER_W, q, grid=grid, **model_kw)
    states, dp, dw, violation = [], [], [], None

    def stop(s):
        nonlocal violation
        k = len(states); states.append(s); m = s.summary
        dp.append(m['temperature_max_C'] - baseline['Tmax_C'][k]); dw.append(m['wall_max_C'] - baseline['wall_max_C'][k])
        exc = max(0., 483. - m['temperature_min_C'], m['temperature_max_C'] - 682., dp[-1] - margin, dw[-1] - margin)
        if exc > 0 and violation is None:
            violation = dict(node_index=k, excess_K=exc)
        return violation is not None
    history_fixed_power(model, times, None, None if folder is None else Path(folder) / 'history', stop)
    res = dict(q=list(map(float, q)), nodes=len(states), dp=dp, dw=dw, violation=violation,
               X=[s.summary['X'] for s in states], feasible_history=(violation is None and len(states) == len(times)))
    res['X_test'] = res['J'] = None
    if utest and res['feasible_history']:
        res['X_test'] = uniform_test(states[-1].health, states[-1], None if folder is None else Path(folder) / 'utest',
                                     grid=grid, **model_kw)
        res['J'] = res['X_test'] - baseline['X_test_Uniform']
    res['feasible'] = bool(res['feasible_history'] and res['J'] is not None)
    return res, states
