import taichi as ti

from paraphin import r1, r2, r3, r4, r5, r6
from paraphin.constants import data_type, Nr, dt, D, gamma

D_2_g = D * 0.5 / gamma


@ti.func
def calc_qp_m_k_fi(i, j, Wps, m, fi, Ur, Ub, integr_r2_fi0, integr_r4_fi0, new_qp, k_mult, m_mult) -> None:
    """
    Вычисление концентрации взвешенных частиц парафина по явной схеме

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
    m_mult: taichi.field(Nx, Ny)
        Изменение пористрости из-за влияния частиц парафина, [-]
    k_mult: taichi.field(Nx, Ny)
        Изменение проницаемости из-за влияния частиц парафина, [-]
    """
    qp1 = 0.0
    qp2 = 0.0
    r2fi = 0.0
    r4fi = 0.0

    for ij in ti.ndrange((1, Nr)):
        dr = r1[ij] - r1[ij-1]
        A_fi = (fi[i,j,ij-1] * r1[ij] - fi[i,j,ij] * r1[ij-1]) / dr
        B_fi = (fi[i,j,ij] - fi[i,j,ij-1]) / dr
        A_ur = (Ur[i,j,ij-1] * r1[ij] - Ur[i,j,ij] * r1[ij - 1]) / dr
        B_ur = (Ur[i,j,ij] - Ur[i,j,ij-1]) / dr

        qp1 += ((r2[ij] - r2[ij-1]) * B_fi * B_ur / 2 + (r4[ij] - r4[ij-1]) * A_fi * A_ur / 4 +
                (r3[ij] - r3[ij-1]) * (A_fi * B_ur + B_fi * A_ur) / 3)  # r * ur * fi
        r2fi += (r3[ij] - r3[ij-1]) * A_fi / 3 + (r4[ij] - r4[ij-1]) * B_fi / 4  # r^2 * fi
        r4fi += (r5[ij] - r5[ij-1]) * A_fi / 5 + (r6[ij] - r6[ij-1]) * B_fi / 6  # r^4 * fi
        if r1[ij] <= D_2_g:  # D * 0.5 / gamma
            A_ub = (Ub[i,j,ij-1] * r1[ij] - Ub[i,j,ij] * r1[ij - 1]) / dr
            B_ub = (Ub[i,j,ij] - Ub[i,j,ij-1]) / dr
            qp2 +=  (r3[ij] - r3[ij-1]) * A_ub / 3 + (r4[ij] - r4[ij-1]) * B_ub / 4  # ub * r^2

        # Обновление функции пор по размерам
        fi[i, j, ij] = upd_fi(fi[i, j, ij], Ur[i, j, ij], fi[i, j, ij - 1], Ur[i, j, ij-1], dr, Ub[i, j, ij])

    new_qp[i, j] = m[i, j] * (2.0 * qp1 + Wps[i, j] * qp2) / r2fi
    m_mult[i, j] = r2fi / integr_r2_fi0
    k_mult[i, j] = r4fi / integr_r4_fi0

    # if i == 0 and j == 0:
    #     print()
    #     print(r2fi/integr_r2_fi0, r4fi/integr_r4_fi0)
    #     print(qp1/ r2fi, qp2/ r2fi)


@ti.func
def upd_fi(fi: data_type, Ur: data_type, fi1: data_type, Ur1: data_type, dr: data_type, Ub: data_type) -> data_type:
    """
    Обновление функции пор по размерам.

    Parameters
    ----------
    fi: float
        fi[i] - функции распределения пор по размерам
    Ur: float
        ur[i] - Скорость изменения радиуса капилляра
    fi1: float
        fi[i-1] - функции распределения пор по размерам
    Ur1: float
        ur[i-1] - Скорость изменения радиуса капилляра
    dr: float
        r[i]-r[i-1] - шаг дискретизации
    Ub: float
        ub[i] - Скорость блокирования капилляров, [м/с]

    Returns
    -------
    fi: float
        Обновленная функции пор по размерам
    """

    fi -= dt * ((fi * Ur - fi1 * Ur1) / dr + Ub)
    return fi
