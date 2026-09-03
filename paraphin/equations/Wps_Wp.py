"""Вычисление концентрации взвешенных частиц (Wps) и растворенного парафина (Wp) парафина по явной схеме."""
import numpy as  np
from numba import njit

from paraphin.constants import ro_p, ro_o, volume, Tm, R, alpha, data_type, init_Wp, init_T

min_Wp_bound = 1e-6

reverse_Tm = 1.0 / Tm
alpha_R = alpha / R


@njit(cache=True)
def wps_wp_equation(i, j, qp, m, m_0, S, S_0, Wp, Wps, T, T_0, cells_Wp_eq, new_Wp, new_Wps, dt) -> None:
    """Концентрации растворенного (Wp) и взвешенного (Wps) парафина по явной схеме.

    Считается только ниже начальной температуры. Это нужно, чтобы избежать лишних вычислений в области,
    где температурный фронт еще не прошел. Читает `cells_Wp_eq` до того, как `flows_in_cells` перезапишет буфер,
     то есть работает с перетоками предыдущего шага.

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    if T[i, j] < init_T * 0.95:
        if Wp[i, j] > min_Wp_bound:
            Wps_i  = _get_Wps(Wp[i, j], Wps[i, j], T[i, j])
            Wps_0_i  = _get_Wps(Wp[i, j], Wps[i, j], T_0[i, j])

            _new_Wp = Wp[i, j] + dt / (m[i, j] * (1.0 - S[i, j]) * ro_o) * (
                    - Wp[i, j] * ro_o * (m[i, j] * (1.0 - S[i, j]) - m_0[i, j] * (1.0 - S_0[i, j])) / dt
                    - ro_p * (m[i, j] * (1.0 - S[i, j]) * Wps_i - m_0[i, j] * (1.0 - S_0[i, j]) * Wps_0_i) / dt
                    + cells_Wp_eq[i, j] / volume + ro_p * qp[i, j])

            colmatation = qp[i, j] * dt * ro_p / ((1.0-Wps[i,j]) * ro_o + Wps[i,j] * ro_p)
            new_Wp[i, j] += max(_new_Wp, 0.0)
            new_Wps[i, j] = max(init_Wp - new_Wp[i, j] + colmatation , 0.0)

        else:
            colmatation = qp[i, j] * dt * ro_p / ((1.0-Wps[i,j]) * ro_o + Wps[i,j] * ro_p)
            new_Wps[i, j] = max(Wps[i, j] + colmatation, 0)
    else:
        new_Wp[i, j] = Wp[i, j]


@njit(cache=True)
def _get_Wps(Wp: data_type, Wps: data_type, T: data_type) -> data_type:
    """Моделирование процесса кристаллизации парафина."""
    new_Wps = Wps

    # exact_solution = alpha_R * Tm / (alpha_R + Tm * ti.log(border))
    if Wp > min_Wp_bound and T > 0.9 * Tm:
        new_Wps = Wp * np.exp(alpha_R * (1.0 / T - reverse_Tm))

    return new_Wps


@njit(cache=True)
def wps_wp_wells(well, m, S, T, Wp, Wps, new_Wp, dt) -> None:
    """Вынос парафина добывающей скважиной из уравнения для Wp.

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    i, j = well.i, well.j
    if T[i, j] < init_T * 0.95 and Wp[i, j] > min_Wp_bound:
        new_Wp[i, j] -= well.q[0] * (Wp[i, j] * ro_o + Wps[i, j] * ro_p) * dt / (m[i, j] * (1.0 - S[i, j]) * ro_o * volume)
