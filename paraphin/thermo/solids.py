r"""Твердые фазы на уравнении состояния: парафин (multi-solid и идеальный твердый раствор), асфальтены (Nghiem).

Плавление чистого компонента (Prausnitz et al., Molecular Thermodynamics of Fluid-Phase Equilibria, 1999):
    ln(f^{0,L}/f^{0,S}) = dG_fus/(RT) = (dH_fus/(RT))(1 - T/T_fus),
по умолчанию без твердо-твердого перехода и скачка теплоемкости; оба слагаемых - необязательные аргументы
(`fusion_ln_ratio`, свойства н-алканов по Coutinho - `characterization.coutinho_nalkane`). При T = T_fus dG = 0,
поэтому WAT чистого компонента в точности равна его T_fus - проверка, не зависящая от уравнения состояния и давления.

Multi-solid (Lira-Galeana, Firoozabadi & Prausnitz, AIChE J 1996, 42:239): каждый выпадающий компонент -
своя чистая твердая фаза, f_i^L(T, p, x) = f_i^{0,S}(T, p). Опора - чистая жидкость из того же уравнения
состояния, а давление в твердой фазе - поправкой Пойнтинга на скачок объема (Pan, Firoozabadi & Fotland,
SPE PF 1997, 12:250):
    ln f_i^{0,S} = ln[phi_i^{pure,L}(T, p) p] - dG_fus,i/(RT) - dv_i (p - p_ref)/(RT),   dv_i = v_i^L - v_i^S.
Без последнего слагаемого твердое молча получает объем жидкости, и наклон dWAT/dP дегазированной нефти
выходит на порядок меньше измеренного (`experiments/уравнение_состояния`): Клапейрон-Клаузиус держится на dv.
Идеальный твердый раствор (упрощенный Won, FPE 1986, 30:265) - одна смешанная фаза с K_i = exp(dG_fus/RT);
переоценивает выпадение легких групп и служит только для сравнения. Неидеальный твердый раствор - предсказательный
UNIQUAC Coutinho (`UniquacSolidWax`).

Асфальтены (Nghiem et al., SPE 26642, 1993; Qin et al., IECR 2000, 39:2644): осаждающийся псевдокомпонент
образует чистую твердую фазу, ln f_s(T, p) = ln f_s* + v_s (p - p*)/(RT); опорная фугитивность f_s*
калибруется по давлению начала осаждения p*: ln f_s* = ln f_a^L(T*, p*, x_feed).
"""
import numpy as np
from scipy.optimize import brentq

from .eos import R_GAS


def fusion_ln_ratio(T, t_fus, dh_fus, t_tr=None, dh_tr=None, dcp=None):
    """dG/(RT) = ln(f^{0,L}/f^{0,S}): больше нуля ниже T_fus, где твердое устойчивее жидкости.

    Полное условие (Prausnitz et al., 1999; обзор, разд. 1.4): ниже температуры перехода T_tr ротаторной фазы в
    орторомбическую добавляется (dH_tr/(RT))(1 - T/T_tr), а скачок теплоемкости при плавлении dCp дает
    -(dCp/R)[(T_fus/T - 1) - ln(T_fus/T)]."""
    r = (dh_fus / (R_GAS * T)) * (1.0 - T / t_fus)
    if t_tr is not None:
        r = r + np.where(T < t_tr, (dh_tr / (R_GAS * T)) * (1.0 - T / t_tr), 0.0)
    if dcp is not None:
        ratio = t_fus / T
        r = r - (dcp / R_GAS) * ((ratio - 1.0) - np.log(ratio))
    return r


class MultiSolidWax:
    """Multi-solid: осадки - чистые компоненты `formers` с температурами и теплотами плавления t_fus, dh_fus и
    скачками мольного объема dv [м^3/моль] при давлении p_ref [Па], к которому отнесены t_fus, dh_fus."""

    def __init__(self, eos, formers, t_fus, dh_fus, dv=0.0, p_ref=101325.0, t_tr=None, dh_tr=None, mw=None):
        self.eos = eos
        self.formers = np.asarray(formers, int)
        self.t_fus = np.asarray(t_fus, float)
        self.dh_fus = np.asarray(dh_fus, float)
        self.dv = np.broadcast_to(np.asarray(dv, float), self.formers.shape)
        self.p_ref = p_ref
        # необязательные: твердо-твердый переход (T_tr, dH_tr) и скачок теплоемкости по Pedersen (молярные массы)
        shape = self.formers.shape
        self.t_tr = None if t_tr is None else np.broadcast_to(np.asarray(t_tr, float), shape)
        self.dh_tr = None if dh_tr is None else np.broadcast_to(np.asarray(dh_tr, float), shape)
        self.mw = None if mw is None else np.broadcast_to(np.asarray(mw, float), shape)
        self._pure = [eos.pure(int(i)) for i in self.formers]

    def ln_ratio(self, T):
        """ln(f^{0,L}/f^{0,S}) выпадающих компонентов при давлении отсчета, [-]."""
        from .characterization import pedersen_dcp
        dcp = None if self.mw is None else pedersen_dcp(self.mw, T)
        return fusion_ln_ratio(T, self.t_fus, self.dh_fus, self.t_tr, self.dh_tr, dcp)

    def solid_ln_fugacity(self, T, p):
        """ln f_i^{0,S}(T, p) выпадающих компонентов; +inf у остальных."""
        lnfs = np.full(self.eos.nc, np.inf)
        dg = self.ln_ratio(T)
        for k, (i, pure, dv) in enumerate(zip(self.formers, self._pure, self.dv)):
            lnfs[i] = (pure.ln_phi(T, p, np.ones(1), 'liquid')[0] + np.log(p) - dg[k]
                       - dv * (p - self.p_ref) / (R_GAS * T))
        return lnfs

    def liquid_ln_fugacity(self, T, p, x):
        return np.log(np.clip(x, 1e-300, None)) + self.eos.ln_phi(T, p, x, 'liquid') + np.log(p)

    def _solve_set(self, T, p, z, S, lnfs, tol=1e-12, maxiter=300):
        """Равновесие при фиксированном наборе осадков S: x_i = f_i^S/(phi_i p) для i из S, остальные -
        z_i/L. Возвращает (x, s, L) или None, если набор несовместим (выпало бы все)."""
        nc = z.size
        non = np.ones(nc, bool)
        non[S] = False
        z_non = z[non].sum()
        x, L = z.copy(), 1.0
        for _ in range(maxiter):
            x_s = np.exp(lnfs[S] - self.eos.ln_phi(T, p, x, 'liquid')[S] - np.log(p))
            denom = 1.0 - x_s.sum()
            if denom <= 1e-12:
                return None
            L = z_non / denom if z_non > 0 else 0.0
            x_new = np.empty(nc)
            if L > 0:
                x_new[non] = z[non] / L
                x_new[S] = x_s
            else:  # вся смесь из выпадающих компонентов
                x_new[non] = 0.0
                x_new[S] = x_s / x_s.sum()
            done = np.max(np.abs(x_new - x)) < tol
            x = x_new
            if done:
                break
        s = np.zeros(nc)
        s[S] = z[S] - L * x[S]
        return x, s, L

    def precipitate(self, T, p, z, tol=1e-9):
        """Multi-solid flash: набор осадков наращивается по самому пересыщенному компоненту, компоненты
        с неположительным количеством убираются.

        Returns
        -------
        s, L, x: numpy.ndarray, float, numpy.ndarray
            Моли компонентов в твердых фазах на моль смеси, мольная доля жидкости, состав жидкости
        """
        z = np.asarray(z, float)
        lnfs = self.solid_ln_fugacity(T, p)
        S = []
        x, s, L = z.copy(), np.zeros(z.size), 1.0
        for _ in range(self.formers.size + 5):
            if S:
                out = self._solve_set(T, p, z, S, lnfs)
                if out is None:
                    break
                x, s, L = out
                neg = [i for i in S if s[i] <= 1e-12]
                if neg:
                    S = [i for i in S if i not in neg]
                    continue
            sup = self.liquid_ln_fugacity(T, p, x) - lnfs
            cand = [int(j) for j in self.formers if j not in S and sup[j] > tol]
            if not cand:
                break
            S.append(max(cand, key=lambda k: sup[k]))
        return s, L, x

    def wat(self, p, z, t_lo=150.0, t_hi=450.0):
        """Температура начала кристаллизации состава z при давлении p, [K]."""
        z = np.asarray(z, float)

        def f(T):
            return float(np.max(self.liquid_ln_fugacity(T, p, z) - self.solid_ln_fugacity(T, p)))

        return float(brentq(f, t_lo, t_hi, xtol=1e-6))


class IdealSolidSolutionWax:
    """Идеальный твердый раствор: одна смешанная твердая фаза из компонентов с маской `is_former`."""

    def __init__(self, is_former, t_fus, dh_fus):
        self.is_former = np.asarray(is_former, bool)
        self.t_fus = np.asarray(t_fus, float)    # по всем компонентам; у не выпадающих не читается
        self.dh_fus = np.asarray(dh_fus, float)

    def k_values(self, T):
        """K_i = x_i^S/x_i^L = exp(dG_fus/RT) у выпадающих, 0 у остальных."""
        K = np.zeros(self.is_former.size)
        f = self.is_former
        K[f] = np.exp(fusion_ln_ratio(T, self.t_fus[f], self.dh_fus[f]))
        return K

    def precipitate(self, T, z):
        """Доля твердого psi из sum z_i (K_i - 1)/[1 + psi(K_i - 1)] = 0. Возвращает (s, L, x), как multi-solid."""
        z = np.asarray(z, float)
        K = self.k_values(T)
        if np.sum(z * (K - 1.0)) <= 0.0:
            return np.zeros(z.size), 1.0, z.copy()
        psi = float(brentq(lambda q: np.sum(z * (K - 1.0) / (1.0 + q * (K - 1.0))), 0.0, 1.0 - 1e-12,
                           xtol=1e-14))
        x = z / (1.0 + psi * (K - 1.0))
        return psi * K * x, 1.0 - psi, x

    def wat(self, z, t_lo=150.0, t_hi=450.0):
        """WAT идеального твердого раствора: sum z_i (K_i - 1) = 0, [K]."""
        z = np.asarray(z, float)
        return float(brentq(lambda T: np.sum(z * (self.k_values(T) - 1.0)), t_lo, t_hi, xtol=1e-6))


class UniquacSolidWax:
    """Твердый раствор н-алканов по предсказательному UNIQUAC (Coutinho, IECR 1998, 37:4870; SPE 78324, 2002).

    Равновесие x_i^L phi_i^L P = x_i^S gamma_i^S f_i^{0,S}: опора - чистое твердое тех же групп (`MultiSolidWax`,
    с переходом и dCp), неидеальность твердого раствора - UNIQUAC с параметрами из свойств н-алканов:
    r_i = 0.1 n + 0.0672, q_i = 0.1 n + 0.1141, Z = 6 (орторомбическая решетка),
    lambda_ii = -(2/Z)(dH_sub,i - RT), dH_sub = dH_vap + dH_m + dH_tr, а энергия разнородной пары - как у пары
    более короткой молекулы. Жидкость - уравнение состояния или идеальный раствор (`liquid='ideal'`, так
    Coutinho & Daridon описывали нефть). Твердый раствор один.

    ponytail: одна твердая фаза - расслоение на несколько твердых растворов (UNIQUAC его предсказывает у
    бимодальных смесей) - когда понадобится.
    """
    Z = 6.0

    def __init__(self, ms, n_carbon, liquid='eos'):
        from .characterization import nalkane_dh_vap
        self.ms = ms
        self.formers = ms.formers
        n = np.asarray(n_carbon, float)
        self.r, self.q = 0.1 * n + 0.0672, 0.1 * n + 0.1141
        dh_tr = 0.0 if ms.dh_tr is None else ms.dh_tr
        self.dh_sub = nalkane_dh_vap(n) + ms.dh_fus + dh_tr
        self.liquid = liquid

    def ln_gamma(self, T, xs):
        """ln gamma_i твердого раствора состава xs (по выпадающим компонентам)."""
        xs = np.maximum(np.asarray(xs, float), 1e-300)
        lam = -(2.0 / self.Z) * (self.dh_sub - R_GAS * T)
        lam_ij = np.maximum(lam[:, None], lam[None, :])  # пара - как у более короткой (меньшее |lambda|)
        tau = np.exp(-(lam_ij - lam[None, :]) / (self.q[None, :] * R_GAS * T))  # tau[j, i]
        phi = xs * self.r / np.sum(xs * self.r)
        th = xs * self.q / np.sum(xs * self.q)
        s = th @ tau  # s_i = sum_j theta_j tau_ji
        z2 = self.Z / 2.0
        return (np.log(phi / xs) + 1.0 - phi / xs - z2 * self.q * (np.log(phi / th) + 1.0 - phi / th)
                + self.q * (1.0 - np.log(s) - tau @ (th / s)))

    def _ln_k0(self, T, p, x):
        """ln(x^S gamma^S / x^L) выпадающих компонентов: K-значение без неидеальности твердого."""
        ms = self.ms
        dv = ms.dv * (p - ms.p_ref) / (R_GAS * T)
        if self.liquid == 'ideal':
            return ms.ln_ratio(T) + dv
        lnphi = ms.eos.ln_phi(T, p, x, 'liquid')[self.formers]
        return lnphi + np.log(p) - ms.solid_ln_fugacity(T, p)[self.formers]

    def incipient(self, T, p, z, tol=1e-12, maxiter=500):
        """Пробный состав твердого раствора и сумма S = sum z_i K_i: S > 1 - твердое выпадает."""
        z = np.asarray(z, float)
        lk0 = self._ln_k0(T, p, z)
        zf = z[self.formers]
        w = zf * np.exp(lk0)
        w = w / w.sum()
        for _ in range(maxiter):
            W = zf * np.exp(lk0 - self.ln_gamma(T, w))
            w_new = W / W.sum()
            done = np.max(np.abs(w_new - w)) < tol
            w = w_new
            if done:
                break
        return float(W.sum()), w

    def precipitate(self, T, p, z, tol=1e-11, maxiter=500):
        """Flash жидкость - твердый раствор. Возвращает (s, L, x, xs), как multi-solid, плюс состав твердого."""
        z = np.asarray(z, float)
        S, xs = self.incipient(T, p, z)
        if S <= 1.0:
            return np.zeros(z.size), 1.0, z.copy(), xs
        f = self.formers
        x = z.copy()
        for _ in range(maxiter):
            K = np.zeros(z.size)
            K[f] = np.exp(self._ln_k0(T, p, x) - self.ln_gamma(T, xs))
            rr = lambda psi: np.sum(z * (K - 1.0) / (1.0 + psi * (K - 1.0)))
            psi = 0.0 if rr(0.0) <= 0.0 else float(brentq(rr, 0.0, 1.0 - 1e-12, xtol=1e-15))
            x_new = z / (1.0 + psi * (K - 1.0))
            xs_new = (K * x_new)[f]
            xs_new = xs_new / xs_new.sum()
            done = np.max(np.abs(x_new - x)) < tol and np.max(np.abs(xs_new - xs)) < tol
            x, xs = x_new, xs_new
            if done:
                break
        s = np.zeros(z.size)
        s[f] = psi * xs
        return s, 1.0 - psi, x, xs

    def wat(self, p, z, t_lo=150.0, t_hi=450.0):
        """Температура появления твердого раствора: S(T) = 1, [K]."""
        return float(brentq(lambda T: np.log(self.incipient(T, p, z)[0]), t_lo, t_hi, xtol=1e-6))


class NghiemAsphaltene:
    """Модель твердой фазы асфальтенов: компонент `idx` уравнения `eos`, мольный объем твердого v_solid [м^3/моль]."""

    def __init__(self, eos, idx, v_solid):
        self.eos = eos
        self.idx = int(idx)
        self.v_solid = float(v_solid)
        self._ref = None  # (p*, ln f_s*)

    def liquid_ln_fugacity(self, T, p, x):
        return float(np.log(x[self.idx]) + self.eos.ln_phi(T, p, x, 'liquid')[self.idx] + np.log(p))

    def calibrate_onset(self, T, p_onset, x):
        """f_s* - по давлению начала осаждения: при p_onset жидкость состава x ровно насыщена."""
        self._ref = (p_onset, self.liquid_ln_fugacity(T, p_onset, x))

    def solid_ln_fugacity(self, T, p):
        # ponytail: f_s* не зависит от T - температура входит только через phi_a^L уравнения состояния.
        # Член с теплотой (Nghiem & Coombe) - когда появятся давления начала осаждения при двух температурах
        p0, lnfs0 = self._ref
        return lnfs0 + self.v_solid * (p - p0) / (R_GAS * T)

    def solubility(self, T, p, x, tol=1e-12, maxiter=200):
        """Предельная мольная доля растворенных асфальтенов в жидкости, остальной состав которой
        пропорционален x: x_a = f_s/(phi_a p). Считается и при недосыщении - тогда она больше x_a.

        Returns
        -------
        x_a, x_sat: float, numpy.ndarray
            Предельная доля и состав насыщенной жидкости
        """
        i = self.idx
        rest = np.asarray(x, float).copy()
        rest[i] = 0.0
        rest /= rest.sum()
        lnfs = self.solid_ln_fugacity(T, p)
        xa = float(x[i])
        for _ in range(maxiter):
            xs = rest * (1.0 - xa)
            xs[i] = xa
            xa_new = min(float(np.exp(lnfs - self.eos.ln_phi(T, p, xs, 'liquid')[i] - np.log(p))), 1.0)
            done = abs(xa_new - xa) <= tol * xa_new
            xa = xa_new
            if done:
                break
        xs = rest * (1.0 - xa)
        xs[i] = xa
        return xa, xs

    def precipitate(self, T, p, x):
        """Выпавшие асфальтены из жидкости состава x, [моль на моль жидкости]: остальные компоненты остаются в
        жидкости, L (1 - x_a^sat) = 1 - x_a, s = x_a - L x_a^sat; при недосыщении 0."""
        xa_sat = self.solubility(T, p, x)[0]
        xa = float(x[self.idx])
        if xa_sat >= 1.0:  # растворимость не ограничена
            return 0.0
        return max(0.0, xa - (1.0 - xa) / (1.0 - xa_sat) * xa_sat)
