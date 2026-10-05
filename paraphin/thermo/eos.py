r"""Уравнение состояния Пенга-Робинсона, тест устойчивости фаз и flash пар-жидкость.

Уравнение состояния (Peng & Robinson, Ind. Eng. Chem. Fundam. 1976, 15:59):
    p = RT/(v - b) - a(T)/[v(v + b) + b(v - b)],
в факторе сжимаемости Z = pv/(RT):
    Z^3 - (1 - B)Z^2 + (A - 2B - 3B^2)Z - (AB - B^2 - B^3) = 0,   A = a*p/(RT)^2, B = b*p/(RT).
Параметры компонента: a_i = 0.45724 R^2 Tc^2/Pc * [1 + m_i(1 - sqrt(T/Tc))]^2, b_i = 0.07780 R Tc/Pc,
m_i = 0.37464 + 1.54226w - 0.26992w^2 при w <= 0.491, иначе поправка для тяжелых
m_i = 0.379642 + 1.48503w - 0.164423w^2 + 0.016666w^3 (Robinson & Peng, GPA RR-28, 1978). Правила смешения
Ван-дер-Ваальса a = sum x_i x_j (1 - k_ij) sqrt(a_i a_j), b = sum x_i b_i. Коэффициент фугитивности
(Michelsen & Mollerup, Thermodynamic Models, 2007):
    ln phi_i = (b_i/b)(Z - 1) - ln(Z - B) - A/(2 sqrt2 B) (2 psi_i/a - b_i/b) ln[(Z + (1 + sqrt2)B)/(Z + (1 - sqrt2)B)],
    psi_i = sqrt(a_i) sum_j x_j (1 - k_ij) sqrt(a_j).

Устойчивость - касательная плоскость Michelsen (Fluid Phase Equilib. 1982, 9:1); flash - последовательные
подстановки K = phi_L/phi_V с уравнением Рачфорда-Райса (JPT 1952, 4:19; Michelsen, FPE 1982, 9:21).
Все в СИ: T в K, p в Па.
"""
import numpy as np
from scipy.optimize import brentq

R_GAS = 8.314462618
SQRT2 = np.sqrt(2.0)
_OMEGA_A = 0.45723552894
_OMEGA_B = 0.07779607390
_K_MIN = 1e-200  # K асфальтенов и тяжелых групп в холодном газе уходит в машинный ноль: деление на 0 в Рачфорде-Райсе


class PengRobinson:
    """PR EOS для смеси с критическими свойствами tc [K], pc [Па], ацентрическими факторами omega и kij."""

    def __init__(self, tc, pc, omega, kij=None):
        self.tc = np.asarray(tc, float)
        self.pc = np.asarray(pc, float)
        self.omega = np.asarray(omega, float)
        self.nc = self.tc.size
        self.kij = np.zeros((self.nc, self.nc)) if kij is None else np.asarray(kij, float)
        w = self.omega
        self.m_i = np.where(w <= 0.491, 0.37464 + 1.54226 * w - 0.26992 * w**2,
                            0.379642 + 1.48503 * w - 0.164423 * w**2 + 0.016666 * w**3)
        self.b_i = _OMEGA_B * R_GAS * self.tc / self.pc
        self.sqrt_ac = np.sqrt(_OMEGA_A * (R_GAS * self.tc) ** 2 / self.pc)
        self._one_minus_k = 1.0 - self.kij

    def pure(self, i):
        """Однокомпонентное уравнение для компонента i - опора чистой жидкости в равновесии с твердым."""
        return PengRobinson(self.tc[i:i + 1], self.pc[i:i + 1], self.omega[i:i + 1])

    def _sqrt_a(self, T):
        return self.sqrt_ac * (1.0 + self.m_i * (1.0 - np.sqrt(T / self.tc)))

    def _AB(self, T, p, x):
        s = self._sqrt_a(T)
        a = float(x @ (self._one_minus_k * np.outer(s, s)) @ x)
        b = float(x @ self.b_i)
        return a * p / (R_GAS * T) ** 2, b * p / (R_GAS * T), a, b, s

    def z_roots(self, T, p, x):
        A, B = self._AB(T, p, x)[:2]
        roots = np.roots([1.0, -(1.0 - B), A - 2.0 * B - 3.0 * B**2, -(A * B - B**2 - B**3)])
        real = roots[np.abs(roots.imag) < 1e-9].real
        return np.sort(real[real > B + 1e-12])

    def z_factor(self, T, p, x, phase='auto'):
        """Корень Z: 'liquid' - наименьший, 'vapor' - наибольший, 'auto' - с наименьшей энергией Гиббса."""
        roots = self.z_roots(T, p, x)
        if roots.size == 0:
            raise RuntimeError(f'Нет физического корня Z при T={T}, p={p}')
        if phase == 'liquid' or roots.size == 1:
            return float(roots[0])
        if phase == 'vapor':
            return float(roots[-1])
        A, B = self._AB(T, p, x)[:2]
        g = [(Z - 1.0) - np.log(Z - B) - A / (2.0 * SQRT2 * B)
             * np.log((Z + (1.0 + SQRT2) * B) / (Z + (1.0 - SQRT2) * B)) for Z in roots]
        return float(roots[int(np.argmin(g))])

    def ln_phi(self, T, p, x, phase='auto'):
        x = np.asarray(x, float)
        A, B, a, b, s = self._AB(T, p, x)
        Z = self.z_factor(T, p, x, phase)
        psi = s * (self._one_minus_k @ (x * s))
        L = np.log((Z + (1.0 + SQRT2) * B) / (Z + (1.0 - SQRT2) * B))
        bi_b = self.b_i / b
        return bi_b * (Z - 1.0) - np.log(Z - B) - A / (2.0 * SQRT2 * B) * (2.0 * psi / a - bi_b) * L


def wilson_k(eos, T, p):
    """K-значения по корреляции Вильсона - начальное приближение устойчивости и flash."""
    return eos.pc / p * np.exp(5.373 * (1.0 + eos.omega) * (1.0 - eos.tc / T))


def stability_test(eos, T, p, z, tol=1e-12, maxiter=1000, trivial_atol=1e-5, vapor_only=False):
    """Тест устойчивости по касательной плоскости: стационарные точки ln W_i = d_i - ln phi_i(w),
    d_i = ln z_i + ln phi_i(z); неустойчиво, если для нетривиального w сумма W > 1. Старты Вильсона
    «паровой» W = z*K и «жидкостный» W = z/K; vapor_only - только паровой, и пробная фаза должна быть газом по
    правилу Кея (псевдокритическая температура sum w_i T_ci ниже T): у нефти ищется лишь выделение газа, а
    расслоение жидкость-жидкость (при большом kij асфальтены-легкие PR делит живую нефть выше давления
    насыщения на асфальтеновую и обедненную жидкости, и к последней сходится паровой старт) в модели описывает
    твердая фаза Нгхайема.

    ponytail: правило Кея отсекает и газ с T_pc > T (жирный газ при низкой температуре); у нефти модели газ -
    метан, T_pc = 191 K. Строже - трехфазный flash L1-L2-V.

    Returns
    -------
    trial_w: numpy.ndarray или None
        Пробный состав самой неустойчивой стационарной точки; None - состав устойчив
    """
    z = np.asarray(z, float)
    d = np.log(z) + eos.ln_phi(T, p, z)
    K = wilson_k(eos, T, p)
    worst_s, worst_w = 1.0 + 1e-7, None
    for W in ((z * K,) if vapor_only else (z * K, z / K)):
        for _ in range(maxiter):
            lnW = d - eos.ln_phi(T, p, W / W.sum())
            done = np.max(np.abs(lnW - np.log(W))) < tol
            W = np.exp(lnW)
            if done:
                break
        S = float(W.sum())
        w = W / S
        if vapor_only and np.dot(w, eos.tc) >= T:
            continue
        if not np.allclose(w, z, atol=trivial_atol) and S > worst_s:
            worst_s, worst_w = S, w
    return worst_w


def rachford_rice(z, K):
    """Доля пара beta из sum z_i (K_i - 1)/[1 + beta(K_i - 1)] = 0; вне [0, 1] - граница."""
    if np.sum(z * (K - 1.0)) <= 0.0:
        return 0.0
    if np.sum(z * (K - 1.0) / K) >= 0.0:
        return 1.0
    return float(brentq(lambda b: np.sum(z * (K - 1.0) / (1.0 + b * (K - 1.0))), 0.0, 1.0,
                        xtol=1e-15, rtol=8.9e-16))


def flash(eos, T, p, z, tol=1e-12, maxiter=500, vapor_only=False):
    """Flash пар-жидкость: тест устойчивости (vapor_only - см. `stability_test`), затем последовательные
    подстановки K = phi_L/phi_V.

    Returns
    -------
    beta, x, y: float, numpy.ndarray, numpy.ndarray
        Мольная доля пара и составы жидкости и пара. Однофазное состояние - beta = 0 (жидкость) или 1 (пар),
        x = y = z
    """
    z = np.asarray(z, float)
    trial = stability_test(eos, T, p, z, vapor_only=vapor_only)
    if trial is None:
        roots = eos.z_roots(T, p, z)
        Z = eos.z_factor(T, p, z)
        vapor = (roots.size == 1 and Z > 0.5) or (roots.size > 1 and Z == roots[-1])
        return (1.0 if vapor else 0.0), z.copy(), z.copy()
    K = np.maximum(trial / z, _K_MIN)
    for _ in range(maxiter):
        beta = rachford_rice(z, K)
        x = z / (1.0 + beta * (K - 1.0))
        K_new = np.maximum(np.exp(eos.ln_phi(T, p, x, 'liquid') - eos.ln_phi(T, p, K * x, 'vapor')), _K_MIN)
        done = np.max(np.abs(np.log(K_new / K))) < tol
        K = K_new
        if done:
            break
    beta = rachford_rice(z, K)
    if np.max(np.abs(K - 1.0)) < 1e-4 or beta <= 1e-9 or beta >= 1.0 - 1e-9:  # вырождение в одну фазу
        return beta, z.copy(), z.copy()
    x = z / (1.0 + beta * (K - 1.0))
    return beta, x, K * x


def bubble_pressure(eos, T, z, p_lo=1e5, p_hi=1e8, rtol=1e-6):
    """Давление насыщения жидкости состава z - бисекция по ln p на признаке «flash дает две фазы», [Па].

    Признак, а не уравнение sum z_i K_i = 1: выше давления насыщения подстановки сходятся к тривиальному
    решению, и невязка обращается в ноль, не будучи корнем. Пробная фаза - только паровая: насыщение жидкости
    есть зарождение пара.
    """
    def two_phase(p):
        return 0.0 < flash(eos, T, p, z, vapor_only=True)[0] < 1.0

    if not two_phase(p_lo) or two_phase(p_hi):
        raise ValueError(f'Давление насыщения при T={T:.2f} K вне [{p_lo:.3g}, {p_hi:.3g}] Па')
    lo, hi = np.log(p_lo), np.log(p_hi)
    while hi - lo > rtol:
        mid = 0.5 * (lo + hi)
        if two_phase(np.exp(mid)):
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))
