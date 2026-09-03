"""Вспомогательные процедуры для реализации решения методом конечных объемов."""
from numba import njit, prange

from paraphin.constants import data_type, ro_o, ro_p, K_o, K_f, K_w, K_p, Nx, Ny
from .phase_f import pf_o, pf_w


@njit(parallel=True, cache=True)
def calc_mobility(k, S, mu_o, mu_w, lam_o, lam_w) -> None:
    """Подвижности фаз k*pf/mu во всех ячейках.

    Считаются один раз за шаг, до сборки матрицы давления. Раньше `mid_Ko_Kw`, `up_ko` и `up_kw`
    пересчитывали их для обеих ячеек на каждой из четырех граней - по 8 вычислений `pf_o`/`pf_w`
    на ячейку вместо одного.
    """
    for i in prange(Nx):
        for j in range(Ny):
            lam_o[i, j] = _K_o(k[i, j], S[i, j], mu_o[i, j])
            lam_w[i, j] = _K_w(k[i, j], S[i, j], mu_w[i, j])


@njit(cache=True)
def up_fraction(lam_up: data_type, lam_sum: data_type) -> data_type:
    """Доля фазы в суммарной подвижности ячейки вверх по потоку. Ноль при нулевой подвижности."""
    return lam_up / lam_sum if lam_sum > 0.0 else 0.0


@njit(cache=True)
def mid(x: data_type, y: data_type) -> data_type:
    """Среднее гармоническое. Ноль при нулевой сумме: иначе 0/0 даст NaN и испортит всю матрицу."""
    s = x + y
    return 2.0 * x * y / s if s > 0.0 else 0.0


@njit(cache=True)
def mid_Ko_Kw(k_i: data_type, s_i: data_type, mu_o_i: data_type, mu_w_i: data_type,
              k_j: data_type, s_j: data_type, mu_o_j: data_type, mu_w_j: data_type) -> data_type:
    """mid(Ko + Kw)_ij"""
    x = _K_o(k_i, s_i, mu_o_i) + _K_w(k_i, s_i, mu_w_i)
    y = _K_o(k_j, s_j, mu_o_j) + _K_w(k_j, s_j, mu_w_j)
    s = x + y

    return 2.0 * x * y / s if s > 0.0 else 0.0


@njit(cache=True)
def up_kw(k_i: data_type, s_i: data_type, p_i: data_type, mu_o_i: data_type, mu_w_i: data_type,
          k_j: data_type, s_j: data_type, p_j: data_type, mu_o_j: data_type, mu_w_j: data_type) -> data_type:
    """Значение берется вверх по потоку: up(kw / (ko + kw)"""
    if p_i >= p_j:
        kw, ko = _K_w(k_i, s_i, mu_w_i), _K_o(k_i, s_i, mu_o_i)
    else:
        kw, ko = _K_w(k_j, s_j, mu_w_j), _K_o(k_j, s_j, mu_o_j)

    s = kw + ko

    return kw / s if s > 0.0 else 0.0


@njit(cache=True)
def up_ko(k_i: data_type, s_i: data_type, p_i: data_type, mu_o_i: data_type, mu_w_i: data_type,
          k_j: data_type, s_j: data_type, p_j: data_type, mu_o_j: data_type, mu_w_j: data_type) -> data_type:
    """Значение берется вверх по потоку: up(ko / (ko + kw)"""
    if p_i >= p_j:
        kw, ko = _K_w(k_i, s_i, mu_w_i), _K_o(k_i, s_i, mu_o_i)
    else:
        kw, ko = _K_w(k_j, s_j, mu_w_j), _K_o(k_j, s_j, mu_o_j)

    s = kw + ko

    return ko / s if s > 0.0 else 0.0


@njit(cache=True)
def up_T(p_i: data_type, T_i: data_type, p_j: data_type, T_j: data_type) -> data_type:
    """Значение берется вверх по потоку: up(kw / (ko + kw)"""
    ret = 0.0

    if p_i >= p_j:
        ret = T_i
    else:
        ret = T_j

    return ret


@njit(cache=True)
def up_wp(p_i: data_type, Wp_i: data_type, Wps_i: data_type,
          p_j: data_type, Wp_j: data_type, Wps_j: data_type) -> data_type:
    """Вычисление взвешенного и растворенного парафина вверх по потку. """
    ret = 0.0

    if p_i >= p_j:
        ret = ro_o * Wp_i + ro_p * Wps_i
    else:
        ret = ro_o * Wp_j + ro_p * Wps_j

    return ret


@njit(cache=True)
def mid_lam(S_i: data_type, m_i: data_type, Wps_i: data_type,
            S_j: data_type, m_j: data_type, Wps_j: data_type) -> data_type:
    """Вычисление осредненного коэффициента теплопроводности."""
    # if p_i >= p_j:
    #     ret = m_i * (S_i * K_w + (1.0 - S_i) * ((1.0 - Wps_i) * K_o + Wps_i * K_p)) + (1.0 - m_i) * K_f
    # else:
    #     ret = m_j * (S_j * K_w + (1.0 - S_j) * ((1.0 - Wps_j) * K_o + Wps_j * K_p)) + (1.0 - m_j) * K_f

    ret_i = m_i * (S_i * K_w + (1.0 - S_i) * ((1.0 - Wps_i) * K_o + Wps_i * K_p)) + (1.0 - m_i) * K_f
    ret_j =  m_j * (S_j * K_w + (1.0 - S_j) * ((1.0 - Wps_j) * K_o + Wps_j * K_p)) + (1.0 - m_j) * K_f

    return mid(ret_i, ret_j)


@njit(cache=True)
def _K_o(k: data_type, s: data_type, mu_o: data_type) -> data_type:
    """Фазовая проницаемость нефти."""
    return k * pf_o(s) / mu_o


@njit(cache=True)
def _K_w(k: data_type, s: data_type, mu_w: data_type) -> data_type:
    """Фазовая проницаемость воды."""
    return k * pf_w(s) / mu_w
