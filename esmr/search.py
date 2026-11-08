"""Surrogate-assisted trust-region search over the ten zone shares (sum 1, non-negative).

J is close to linear in q and is modelled by a ridge-regularized linear fit; the thermal
constraints are not, and their slack is modelled by a local Gaussian process (linear trend
plus RBF residual) on the nearest evaluated profiles. Candidates are scored by predicted J
times predicted probability of feasibility; every candidate is then evaluated with the full
model (esmr.histories.evaluate_profile), which alone decides feasibility.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr


def project_simplex(V):
    """Euclidean projection of each row onto the non-negative shares that sum to one."""
    V = np.atleast_2d(np.asarray(V, float)); n, d = V.shape
    u = np.sort(V, axis=1)[:, ::-1]; css = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, d + 1); rho = (u - css / ind > 0).sum(1)
    theta = css[np.arange(n), rho - 1] / rho
    W = np.maximum(V - theta[:, None], 0.0)
    return W / W.sum(1, keepdims=True)


def excess(dp, dw, margin):
    """Signed thermal slack over the completed nodes (K).  <=0 means every completed node obeys the cap."""
    vals = [x - margin for x in dp] + [x - margin for x in dw]
    return max(vals) if vals else float('inf')


class Archive:
    """Every evaluated profile of the search with its J and its constraint excess."""
    def __init__(self, margin):
        self.margin = margin
        self.rec = []

    def add(self, r):
        r = dict(r)
        r['s'] = excess(r.get('dp', []), r.get('dw', []), self.margin)
        self.rec.append(r)

    def Q(self, sel=None):
        rs = self.rec if sel is None else [r for r in self.rec if sel(r)]
        return np.array([r['q'] for r in rs], float).reshape(-1, 10), rs

    def feasible(self):
        return [r for r in self.rec if r.get('feasible') and r.get('J') is not None]

    def best(self, n=1):
        return sorted(self.feasible(), key=lambda r: -r['J'])[:n]


class LinearJ:
    """J ~ a + b.q (ridge).  Collinearity from sum(q)=1 is handled by centering + ridge."""
    def __init__(self, Q, J, lam=1e-5):
        self.mu = Q.mean(0); self.ym = float(np.mean(J))
        A = Q - self.mu
        self.w = np.linalg.solve(A.T @ A + lam * np.eye(10), A.T @ (np.asarray(J) - self.ym))
        self.n = len(J)

    def __call__(self, Q):
        return self.ym + (np.atleast_2d(Q) - self.mu) @ self.w

    def loo_rmse(self, Q, J, lam=1e-5):
        e = []
        for i in range(len(J)):
            m = np.arange(len(J)) != i
            e.append(LinearJ(Q[m], np.asarray(J)[m], lam)(Q[i])[0] - J[i])
        return float(np.sqrt(np.mean(np.square(e))))


class LocalGP:
    """Linear trend + RBF-GP residual on the nearest points; returns mean and std of the slack s(q)."""
    def __init__(self, Q, y, ell, sn):
        self.Q = Q; self.ell = ell
        n = len(y); Xa = np.c_[np.ones(n), Q]
        lam = 1e-3
        self.b = np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ y)
        r = y - Xa @ self.b
        self.sf2 = max(float(np.var(r)), 0.25); self.sn2 = sn * sn
        K = self.sf2 * self._k(Q, Q) + self.sn2 * np.eye(n)
        self.cf = cho_factor(K, lower=True)
        self.al = cho_solve(self.cf, r)
        self.Ki_diag = np.diag(cho_solve(self.cf, np.eye(n)))

    def _k(self, A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / self.ell ** 2)

    def loo(self):
        return self.al / self.Ki_diag  # residual of leave-one-out prediction (y_i - mu_-i)

    def predict(self, Qs):
        Qs = np.atleast_2d(Qs); out_m = np.empty(len(Qs)); out_s = np.empty(len(Qs))
        for a in range(0, len(Qs), 2000):
            q = Qs[a:a + 2000]
            ks = self.sf2 * self._k(q, self.Q)
            m = np.c_[np.ones(len(q)), q] @ self.b + ks @ self.al
            v = self.sf2 - np.einsum('ij,ji->i', ks, cho_solve(self.cf, ks.T)) + self.sn2
            out_m[a:a + 2000] = m; out_s[a:a + 2000] = np.sqrt(np.maximum(v, 1e-9))
        return out_m, out_s


def fit_local_gp(Q, s, center, k=260):
    """Gaussian process of the constraint excess on the k evaluated profiles nearest to center; length scale and noise chosen by leave-one-out error."""
    d = np.abs(Q - center).sum(1)
    idx = np.argsort(d)[:min(k, len(s))]
    Qk, sk = Q[idx], np.clip(s[idx], -40.0, 60.0)  # clip: far-infeasible values carry no boundary information
    best = None
    for ell in (0.04, 0.07, 0.12, 0.2):
        for sn in (0.3, 1.0):
            try:
                g = LocalGP(Qk, sk, ell, sn)
            except np.linalg.LinAlgError:
                continue
            near = np.abs(sk) < 8
            e = g.loo()
            score = np.sqrt(np.mean(e[near] ** 2)) if near.sum() >= 5 else np.sqrt(np.mean(e ** 2))
            if best is None or score < best[0]:
                best = (score, g, ell, sn)
    return best[1], dict(ell=best[2], sn=best[3], loo_near_rmse=float(best[0]), n_local=int(len(idx)))


def propose(arch, rng, B, radius, center_rank=0, n_pool=8000, min_sep=0.012, explore_frac=0.3):
    """Return (list of q, info).  Candidates are scored by predicted J x predicted feasibility."""
    Qj, rj = arch.Q(lambda r: r.get('J') is not None)
    jm = LinearJ(Qj, [r['J'] for r in rj])
    top = arch.best(max(center_rank + 1, 1))
    if not top:
        raise RuntimeError('no feasible profile among the starting histories')
    inc = top[min(center_rank, len(top) - 1)]
    c = np.array(inc['q'], float)
    Qa, ra = arch.Q(lambda r: r.get('dp') is not None and len(r.get('dp', [])) > 0)
    s = np.array([r['s'] for r in ra], float)
    gp, gpinfo = fit_local_gp(Qa, s, c)
    # candidate pool: random tangent steps + J-gradient-aimed steps, L1 length <= radius
    g = jm.w - jm.w.mean(); gn = np.abs(g).sum()
    steps = []
    for i in range(n_pool):
        t = rng.uniform(0.15, 1.0)
        noise = rng.normal(size=10); noise -= noise.mean(); noise /= np.abs(noise).sum()
        if i % 5 < 3:      # aimed
            d = 0.65 * g / gn + 0.35 * noise
        else:              # isotropic
            d = noise
        d = d / np.abs(d).sum()
        steps.append(c + radius * t * d)
    pool = project_simplex(np.array(steps))
    # drop near-duplicates of already evaluated points
    Qall = np.array([r['q'] for r in arch.rec], float)
    dmin = np.abs(pool[:, None, :] - Qall[None, :, :]).sum(-1).min(1) if len(Qall) < 3000 else np.full(len(pool), 1.0)
    pool = pool[dmin > 2e-3]
    Jh = jm(pool)
    mu, sd = gp.predict(pool)
    Pf = ndtr(-mu / sd)
    Jbest = inc['J']
    gain = Jh - Jbest
    chosen, used = [], []

    def take(order, limit):
        for i in order:
            if len(chosen) >= limit:
                break
            if all(np.abs(pool[i] - pool[j]).sum() >= min_sep for j in used):
                chosen.append(pool[i].copy()); used.append(i)

    n_exp = int(round(B * explore_frac)); n_ex = B - n_exp
    prio = Pf * np.maximum(gain, 0.0) + 1e-6 * Pf * Jh      # exploit: expected gain
    take(np.argsort(-prio), n_ex)
    mid = np.where((Pf > 0.2) & (Pf < 0.85))[0]               # explore the thermal boundary
    take(mid[np.argsort(-Jh[mid])] if len(mid) else np.argsort(-Pf * Jh), B)
    take(np.argsort(-Pf), B)                                  # fill
    info = dict(center_id=inc['id'], center_J=Jbest, radius=radius, J_model_n=jm.n, gp=gpinfo,
                pool=int(len(pool)), mean_Pf_chosen=float(np.mean([Pf[i] for i in used])) if used else None,
                mean_Jhat_chosen=float(np.mean([Jh[i] for i in used])) if used else None)
    return [list(map(float, q)) for q in chosen], info


def select_for_main(arch, n=6, min_l1=0.06):
    """The n best feasible coarse-grid profiles that differ from each other by at least min_l1 (L1 distance), to be solved on the main grid."""
    out = []
    for r in arch.best(len(arch.feasible())):
        if all(np.abs(np.array(r['q']) - np.array(o['q'])).sum() >= min_l1 for o in out):
            out.append(r)
        if len(out) >= n:
            break
    return out
