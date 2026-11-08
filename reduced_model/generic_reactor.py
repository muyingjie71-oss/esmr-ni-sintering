"""Reduced model of an electrically heated, plug-flow, equilibrium-limited endothermic reactor (one reaction, pseudo-homogeneous,
isobaric, adiabatic except for the electric heat input q'(z)), with temperature-driven catalyst sintering.
Independent of the eSMR model in esmr/: it tests whether the outlet-equilibrium rule holds for other chemistries.

Reaction: sum_i nu_i A_i = 0, key reactant k (nu_k = -1).  Rate per unit length (mol s-1 m-1):
    r = a(z) * k0 * exp(-E/R (1/T - 1/Tr)) * p_k * (1 - Q/K(T))
Energy:  sum_i F_i cp_i dT/dz = q'(z) - dH(T) r
Sintering (same functional form as Sehested's law, x = s^-7 - 1):  dx/dt = ks(T) = exp(-Es/R (1/T - 1/Tr))  (time in units of 1/ks(Tr))
Thermodynamics: 'nist' uses NIST Shomate data per species.  'const' uses constant species cp, with dH and dS given at T0 and carried to
other temperatures by Kirchhoff's law, dH(T) = dH0 + dcp (T - T0), dS(T) = dS0 + dcp ln(T/T0), dcp = sum nu_i cp_i, lnK = dS/R - dH/(RT),
so the heat of reaction and K are consistent with the heat capacities in the energy balance.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

R = 8.314462618
# NIST Shomate (A..E, F, G, H) for C1 species
SH = {'CH4': (-0.703029, 108.4773, -42.52157, 5.862788, 0.678565, -76.84376, 158.7163, -74.8731),
      'H2O': (30.092, 6.832514, 6.793435, -2.53448, 0.082139, -250.881, 223.3967, -241.8264),
      'H2': (33.066178, -11.363417, 11.432816, -2.772874, -0.158558, -9.980797, 172.707974, 0.0),
      'CO': (25.56759, 6.09613, 4.054656, -2.671301, 0.131021, -118.0089, 227.3665, -110.5271),
      'CO2': (24.99735, 55.18696, -33.69137, 7.948387, -0.136638, -403.6075, 228.2431, -393.5224)}


def nist_cp(s, T):
    A, B, C, D, E = SH[s][:5]; t = T / 1000.
    return A + B * t + C * t * t + D * t ** 3 + E / (t * t)


def nist_h(s, T):
    A, B, C, D, E, F, G, H = SH[s]; t = T / 1000.
    return 1000. * (A * t + B * t * t / 2 + C * t ** 3 / 3 + D * t ** 4 / 4 - E / t + F - H + H)


def nist_g(s, T):
    A, B, C, D, E, F, G, H = SH[s]; t = T / 1000.
    S = A * np.log(t) + B * t + C * t * t / 2 + D * t ** 3 / 3 - E / (2 * t * t) + G
    return nist_h(s, T) - T * S


class Reaction:
    def __init__(self, name, species, nu, feed, p_bar, Tin, E, key=0, thermo='nist', dH=None, dS=None, cp=None, T0=None):
        self.name, self.sp, self.nu, self.F0 = name, species, np.asarray(nu, float), np.asarray(feed, float)
        self.p, self.Tin, self.E, self.key, self.thermo = p_bar, Tin, E, key, thermo
        self.dH0, self.dS0, self.cpc = dH, dS, (np.asarray(cp, float) if cp is not None else None)
        self.dn = self.nu.sum()
        if thermo == 'const':
            if T0 is None:
                raise ValueError("thermo='const' needs T0, the temperature at which dH and dS are given")
            self.T0, self.dcp = float(T0), float(np.dot(self.nu, self.cpc))

    def cp(self, T):
        if self.thermo == 'nist':
            return np.array([nist_cp(s, T) for s in self.sp])
        return self.cpc

    def dH(self, T):
        if self.thermo == 'nist':
            return float(sum(n * nist_h(s, T) for n, s in zip(self.nu, self.sp)))
        return self.dH0 + self.dcp * (T - self.T0)

    def dS(self, T):
        return self.dS0 + self.dcp * np.log(T / self.T0)          # 'const' only

    def lnK(self, T):
        if self.thermo == 'nist':
            return -float(sum(n * nist_g(s, T) for n, s in zip(self.nu, self.sp))) / (R * T)
        return self.dS(T) / R - self.dH(T) / (R * T)

    def lnQ(self, F):
        y = np.maximum(F, 1e-30) / F.sum()
        return float(np.sum(self.nu * np.log(y)) + self.dn * np.log(self.p))

    def Teq(self, F):
        lq = self.lnQ(F); f = lambda T: self.lnK(T) - lq
        if f(150.) * f(3000.) > 0:          # far from equilibrium (Q ~ 0): report the lower bracket
            return 150.
        return brentq(f, 150., 3000.)


class Reactor:
    def __init__(self, rx, L=1., nz=200):
        self.rx, self.L = rx, L
        self.zg = np.linspace(0, L, nz + 1); self.zc = .5 * (self.zg[1:] + self.zg[:-1])

    def solve(self, P, w, a, k0, Tr, n_out=201):
        """w(z): normalized power density (1/m, integral 1); a: activity on self.zc (array).  Returns profiles."""
        rx = self.rx; zc = self.zc

        def rhs(z, u):
            F = u[:-1]; T = u[-1]
            ai = np.interp(z, zc, a)
            y = np.maximum(F, 1e-30) / F.sum(); pk = y[rx.key] * rx.p
            beta = np.exp(np.clip(rx.lnQ(F) - rx.lnK(T), -50, 50))
            r = ai * k0 * np.exp(-rx.E / R * (1. / T - 1. / Tr)) * pk * (1. - beta)
            dF = rx.nu * r
            dT = (P * w(z) - rx.dH(T) * r) / float(np.dot(F, rx.cp(T)))
            return np.append(dF, dT)
        u0 = np.append(rx.F0, rx.Tin)
        zs = np.linspace(0, self.L, n_out)
        sol = solve_ivp(rhs, (0, self.L), u0, method='LSODA', t_eval=zs, rtol=1e-8, atol=1e-11)
        if not sol.success:
            raise RuntimeError(sol.message)
        F = sol.y[:-1].T; T = sol.y[-1]
        X = (rx.F0[rx.key] - F[:, rx.key]) / rx.F0[rx.key]
        return dict(z=zs, F=F, T=T, X=X, Xout=float(X[-1]), Tout=float(T[-1]), Tmax=float(T.max()))

    def ate(self, prof, i=-1):
        return float(prof['T'][i] - self.rx.Teq(prof['F'][i]))


def w_uniform(L=1.):
    return lambda z: 1. / L


def w_early(tail, L=1., shape='flat'):
    Lh = L - tail
    if shape == 'flat':
        return lambda z: (1. / Lh) if z <= Lh else 0.
    return lambda z: (2. / Lh) * (1. - z / Lh) if z <= Lh else 0.     # 'triangle': front-loaded, zero at the tail start
