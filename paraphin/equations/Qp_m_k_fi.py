"""Решение уравнения концентрации взвешенных частиц парафина по явной схеме."""
import taichi as ti

from paraphin import r1, r2, r3, r4, r5, r6
from paraphin.constants import data_type, Nr, dt, D, gamma

D_2_gamma = D * 0.5 / gamma
_a = ti.field(dtype=data_type, shape=Nr)
_b = ti.field(dtype=data_type, shape=Nr)


@ti.func
def calc_qp_m_k_fi(i, j, Wps, m, fi, Ur, Ub, integr_r2_fi0, integr_r4_fi0, new_qp, new_fi, k_mult, m_mult) -> None:
    """Вычисление концентрации взвешенных частиц парафина по явной схеме.

    Parameters
    ----------
    i, j : int
        Индексы текущей ячейки, [-]
    Wps: taichi.field(Nx, Ny)
        Концентрации взвешенных частиц парафина, [-]
    m: taichi.field(Nx, Ny)
        Пористость, [-]
    fi: taichi.field(Nx, Ny, Nr)
        Функция распределения пор по размеру, [-]
    Ub: taichi.field(Nx, Ny, Nr)
        Скорость блокирования капилляра, [м/с]
    Ur: taichi.field(Nx, Ny, Nr)
        Скорость изменения радиуса капилляра, [м/с]
    integr_r2_fi0: float
        Интеграл r^2 * fi_o(r), [m^3]
    integr_r4_fi0: float
        Интеграл r^4 * fi_o(r), [m^5]
    new_qp: taichi.field(Nx, Ny)
         Скорость отложения парафиновых отложений в общем объеме пористой породы
    new_fi: taichi.field(Nx, Ny, Nr)
        Обновленная функция распределения пор по размеру, [-]
    m_mult: taichi.field(Nx, Ny)
        Изменение пористости из-за влияния частиц парафина, [-]
    k_mult: taichi.field(Nx, Ny)
        Изменение проницаемости из-за влияния частиц парафина, [-]
    """
    if Wps[i, j] > 1e-6:
        # Вычисление изменения пористости и проницаемости пласта
        qp1, qp2, r2fi, r4fi = _calculate_integrals(fi, Ur, Ub, i, j)
        new_qp[i, j] = m[i, j] * (2.0 * qp1 + Wps[i, j] * qp2) / r2fi
        m_mult[i, j] = r2fi / integr_r2_fi0
        k_mult[i, j] = r4fi / integr_r4_fi0

        # Обновление функции пор по размерам
        _update_fi(new_fi, fi, Ur, Ub, i, j)


@ti.func
def _calculate_integrals(fi: ti.template(), Ur: ti.template(), Ub: ti.template(), i: int, j: int):
    """Вычисление интегралов функции пор по размерам.

    Parameters
    ----------
    fi: taichi.field(Nx, Ny, Nr)
        Функции распределения пор по размерам
    Ur: taichi.field(Nx, Ny, Nr)
        Скорость изменения радиуса капилляра
    Ub: taichi.field(Nx, Ny, Nr)
        Скорость блокирования капилляров, [м/с]
    i, j: int
        Индексы текущей ячейки, [-]

    Returns
    -------
    qp1: float
        Интеграл функции r * ur * fi
    qp2: float
        Интеграл функции ub * r^2
    r2fi: float
        Интеграл функции r^2 * fi
    r4fi: float
        Интеграл функции r^4 * fi
    """
    qp1, qp2, r2fi, r4fi = 0.0, 0.0, 0.0, 0.0

    for ij in ti.ndrange((1, Nr)):
        dr = r1[ij] - r1[ij - 1]
        A_fi = (fi[i, j, ij - 1] * r1[ij] - fi[i, j, ij] * r1[ij - 1]) / dr
        B_fi = (fi[i, j, ij] - fi[i, j, ij - 1]) / dr
        A_ur = (Ur[i, j, ij - 1] * r1[ij] - Ur[i, j, ij] * r1[ij - 1]) / dr
        B_ur = (Ur[i, j, ij] - Ur[i, j, ij - 1]) / dr

        qp1 += ((r2[ij] - r2[ij - 1]) * A_fi * A_ur / 2 + (r3[ij] - r3[ij - 1]) * (A_fi * B_ur + B_fi * A_ur) / 3 +
                (r4[ij] - r4[ij - 1]) * B_fi * B_ur / 4)  # r * ur * fi
        r2fi += (r3[ij] - r3[ij - 1]) * A_fi / 3 + (r4[ij] - r4[ij - 1]) * B_fi / 4  # r^2 * fi
        r4fi += (r5[ij] - r5[ij - 1]) * A_fi / 5 + (r6[ij] - r6[ij - 1]) * B_fi / 6  # r^4 * fi

        if r1[ij] <= D_2_gamma:  # D * 0.5 / gamma
            A_ub = (Ub[i, j, ij - 1] * r1[ij] - Ub[i, j, ij] * r1[ij - 1]) / dr
            B_ub = (Ub[i, j, ij] - Ub[i, j, ij - 1]) / dr
            qp2 += (r3[ij] - r3[ij - 1]) * A_ub / 3 + (r4[ij] - r4[ij - 1]) * B_ub / 4  # ub * r^2

    return qp1, qp2, r2fi, r4fi


@ti.func
def _update_fi(new_fi: ti.template(), fi: ti.template(), Ur: ti.template(), Ub: ti.template(), i: int, j: int):
    """Обновление функции пор по размерам по неявной схеме с использованием метода прогонки.

    Parameters
    ----------
    new_fi: taichi.field(Nx, Ny, Nr)
        Обновленная функция распределения пор по размерам
    fi: taichi.field(Nx, Ny, Nr)
        Функции распределения пор по размерам
    Ur: taichi.field(Nx, Ny, Nr)
        Скорость изменения радиуса капилляра
    Ub: taichi.field(Nx, Ny, Nr)
        Скорость блокирования капилляров, [м/с]
    i, j: int
        Индексы текущей ячейки, [-]
    """
    c, d, e, f = 0.0, 0.0, 0.0, 0.0
    # TODO переписать аппроксимацию производной по пространству и сравнить
    # Вычисление прогоночных коэффициентов
    if Ur[i, j, 0] > 0:
        d = 1.0 / dt + Ur[i, j, 0] / (r1[1] - r1[0])
        e = 0.0
    else:
        d = 1.0 / dt - Ur[i, j, 0] / (r1[1] - r1[0])
        e = Ur[i, j, 1] / (r1[1] - r1[0])
    _a[0] = -e / d
    _b[0] = (fi[i, j, 0] / dt - Ub[i, j, 0]) / d

    for ij in ti.ndrange((1, Nr)):
        dr = r1[ij] - r1[ij-1]
        f = fi[i, j, ij] / dt - Ub[i, j, ij]
        if Ur[i, j, ij] >= 0:
            c = - Ur[i, j, ij-1] / dr
            d = 1.0 / dt + Ur[i, j, ij] / dr
            e = 0.0
        else:
            c = 0.0
            d = 1.0 / dt - Ur[i, j, ij] / dr
            e = Ur[i, j, ij+1] / dr
        denominator = c * _a[ij - 1] + d
        _a[ij] = -e / denominator
        _b[ij] = (f - c * _b[ij - 1]) / denominator
    _a[Nr - 1] = 0.0

    # Вычисление функции пор размерам
    new_fi[i, j, Nr - 1] = _b[Nr - 1]
    for _ij in ti.ndrange(Nr):
        ij = Nr - 1 - _ij  # тк обратный ход
        new_fi[i, j, ij] = ti.max(new_fi[i, j, ij + 1] * _a[ij] + _b[ij], 0.0)


@ti.func
def _update_fi_deprecated(fi: ti.template(), Ur: ti.template(), Ub: ti.template(), i: int, j: int, ij: int):
    """Обновление функции пор по размерам.

    Parameters
    ----------
    fi: taichi.field(Nx, Ny, Nr)
        Функции распределения пор по размерам
    Ur: taichi.field(Nx, Ny, Nr)
        Скорость изменения радиуса капилляра
    Ub: taichi.field(Nx, Ny, Nr)
        Скорость блокирования капилляров, [м/с]
    i, j, ij: int
        Индексы текущей ячейки, [-]
    """
    ij1, ij2 = ij, ij - 1
    if ij != Nr-1:
        if Ur[i, j, ij]<=0.0:
            ij1, ij2 = ij + 1, ij
        else:
            ij1, ij2 = ij, ij - 1
        dr = r1[ij1] - r1[ij2]
        fi[i, j, ij] -= dt * ((Ur[i, j, ij1] * fi[i, j, ij1] - Ur[i, j, ij2] * fi[i, j, ij2]) / dr + Ub[i, j, ij])

        if fi[j, i, ij] < 1e-9:
            fi[j, i, ij] = 1e-9
