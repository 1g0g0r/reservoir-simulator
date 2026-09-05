"""Решение уравнения концентрации взвешенных частиц парафина по явной схеме."""
from numba import njit

from paraphin import r1, r2, r3, r4, r5, r6, n_pass
from paraphin.constants import Nr, init_m, init_k, min_Wps_bound


@njit(cache=True)
def calc_qp_m_k_fi(i, j, Wps, qp, m, k, fi, Ur, Ub, integr_r2_fi0, integr_r4_fi0, a_tdma, b_tdma,
                   new_qp, new_fi, new_k, new_m, dt) -> None:
    """Скорость отложения парафина, пористость, проницаемость и функция пор по размерам.

    Пористость и проницаемость восстанавливаются по интегралам r^2*fi и r^4*fi, отнесенным к тем же
    интегралам от начального распределения (`integr_r2_fi0`, `integr_r4_fi0`). Сама `fi`
    обновляется прогонкой по сетке радиусов.

    a_tdma, b_tdma: numpy.ndarray(Nr)
        Прогоночные коэффициенты. Своя строка на каждый i: один общий буфер на все ячейки давал
        гонку в prange - потоки затирали друг другу коэффициенты, и fi считалась по мусору.

    Описание остальных аргументов - в докстринге пакета `paraphin.equations`.
    """
    if Wps[i, j] > min_Wps_bound:
        # Вычисление изменения пористости и проницаемости пласта
        qp1, qp2, r2fi, r4fi = _calculate_integrals(fi, Ur, Ub, i, j)
        new_qp[i, j] = m[i, j] * (2.0 * qp1 + Wps[i, j] * qp2) / r2fi
        new_m[i, j] = init_m * r2fi / integr_r2_fi0
        new_k[i, j] = init_k * r4fi / integr_r4_fi0

        # Обновление функции пор по размерам
        _update_fi(new_fi, fi, Ur, Ub, i, j, a_tdma, b_tdma, dt)
    else:
        # Ниже порога кольматации поля не меняются: переносим текущие значения, чтобы new_*
        # оставались согласованы с текущим слоем (new_m читает `saturation_equation`).
        new_qp[i, j] = qp[i, j]
        new_m[i, j] = m[i, j]
        new_k[i, j] = k[i, j]


@njit(cache=True)
def _calculate_integrals(fi, Ur, Ub, i: int, j: int):
    """Интегралы функции пор по размерам для ячейки (i, j).

    Подынтегральные функции восстанавливаются кусочно-линейно по узлам сетки радиусов, поэтому
    интегралы берутся точно по каждому отрезку.

    Returns
    -------
    qp1, qp2, r2fi, r4fi: float
        Интегралы r*ur*fi, ub*r^2, r^2*fi и r^4*fi
    """
    qp1, qp2, r2fi, r4fi = 0.0, 0.0, 0.0, 0.0

    for ij in range(1, Nr):
        dr = r1[ij] - r1[ij - 1]
        A_fi = (fi[i, j, ij - 1] * r1[ij] - fi[i, j, ij] * r1[ij - 1]) / dr
        B_fi = (fi[i, j, ij] - fi[i, j, ij - 1]) / dr
        A_ur = (Ur[i, j, ij - 1] * r1[ij] - Ur[i, j, ij] * r1[ij - 1]) / dr
        B_ur = (Ur[i, j, ij] - Ur[i, j, ij - 1]) / dr

        qp1 += ((r2[ij] - r2[ij - 1]) * A_fi * A_ur / 2 + (r3[ij] - r3[ij - 1]) * (A_fi * B_ur + B_fi * A_ur) / 3 +
                (r4[ij] - r4[ij - 1]) * B_fi * B_ur / 4)  # r * ur * fi
        r2fi += (r3[ij] - r3[ij - 1]) * A_fi / 3 + (r4[ij] - r4[ij - 1]) * B_fi / 4  # r^2 * fi
        r4fi += (r5[ij] - r5[ij - 1]) * A_fi / 5 + (r6[ij] - r6[ij - 1]) * B_fi / 6  # r^4 * fi

    # Ub отлична от нуля только при r < r_pass (частица не проходит горло), поэтому интеграл
    # ub*r^2 идет отдельным коротким циклом, а не проверкой радиуса на каждом узле общего
    for ij in range(1, n_pass):
        dr = r1[ij] - r1[ij - 1]
        A_ub = (Ub[i, j, ij - 1] * r1[ij] - Ub[i, j, ij] * r1[ij - 1]) / dr
        B_ub = (Ub[i, j, ij] - Ub[i, j, ij - 1]) / dr
        qp2 += (r3[ij] - r3[ij - 1]) * A_ub / 3 + (r4[ij] - r4[ij - 1]) * B_ub / 4  # ub * r^2

    return qp1, qp2, r2fi, r4fi


@njit(cache=True)
def _update_fi(new_fi, fi, Ur, Ub, i: int, j: int, a_tdma, b_tdma, dt):
    """Обновление функции пор по размерам по неявной схеме методом прогонки.

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    c, d, e, f = 0.0, 0.0, 0.0, 0.0
    # Вычисление прогоночных коэффициентов
    if Ur[i, j, 0] > 0:
        d = 1.0 / dt + Ur[i, j, 0] / (r1[1] - r1[0])
        e = 0.0
    else:
        d = 1.0 / dt - Ur[i, j, 0] / (r1[1] - r1[0])
        e = Ur[i, j, 1] / (r1[1] - r1[0])
    a_tdma[0] = -e / d
    b_tdma[0] = (fi[i, j, 0] / dt - Ub[i, j, 0]) / d

    # TODO Надо сделать проверку решения в вольфраме
    for ij in range(1, Nr):
        dr = r1[ij] - r1[ij-1]
        f = fi[i, j, ij] / dt - Ub[i, j, ij]
        if Ur[i, j, ij] >= 0:
            c = - Ur[i, j, ij-1] / dr
            # c = - Ur[i, j, ij] / dr
            d = 1.0 / dt + Ur[i, j, ij] / dr
            # d = 1.0 / dt + (2.0 * Ur[i, j, ij] - Ur[i, j, ij-1]) / dr
            e = 0.0
        else:
            c = 0.0
            d = 1.0 / dt - Ur[i, j, ij] / dr
            # d = 1.0 / dt - Ur[i, j, ij-1] / dr
            e = Ur[i, j, ij + 1] / dr if ij + 1 < Nr else 0.0  # за Nr-1 соседа нет
            # e = Ur[i, j, ij] / dr
        denominator = c * a_tdma[ij - 1] + d
        a_tdma[ij] = -e / denominator
        b_tdma[ij] = (f - c * b_tdma[ij - 1]) / denominator
    a_tdma[Nr - 1] = 0.0

    # Вычисление функции пор размерам
    new_fi[i, j, Nr - 1] = max(b_tdma[Nr - 1], 0.0)
    for _ij in range(1, Nr):  # с 1: последний элемент уже посчитан, иначе чтение за границей fi
        ij = Nr - 1 - _ij  # тк обратный ход
        new_fi[i, j, ij] = max(new_fi[i, j, ij + 1] * a_tdma[ij] + b_tdma[ij], 0.0)
