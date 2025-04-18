import taichi as ti

from paraphin.constants import data_type, r, Nr, dt, D, gamma

D_2_g = D * 0.5 / gamma
rr = ti.field(dtype=data_type, shape=Nr)
r2 = ti.field(dtype=data_type, shape=Nr)
r3 = ti.field(dtype=data_type, shape=Nr)
r4 = ti.field(dtype=data_type, shape=Nr)
r5 = ti.field(dtype=data_type, shape=Nr)
r6 = ti.field(dtype=data_type, shape=Nr)
r2_np = r * r
r3_np = r2_np * r
r4_np = r3_np * r
r5_np = r4_np * r
r6_np = r5_np * r
rr.from_numpy(r)
r2.from_numpy(r2_np)
r3.from_numpy(r3_np)
r4.from_numpy(r4_np)
r5.from_numpy(r5_np)
r6.from_numpy(r6_np)


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
        dr = rr[ij] - rr[ij-1]
        A_fi = (fi[i,j,ij-1] * rr[ij] - fi[i,j,ij] * rr[ij-1]) / dr
        B_fi = (fi[i,j,ij] - fi[i,j,ij-1]) / dr
        A_ur = (Ur[i,j,ij-1] * rr[ij] - Ur[i,j,ij] * rr[ij - 1]) / dr
        B_ur = (Ur[i,j,ij] - Ur[i,j,ij-1]) / dr

        qp1 += ((r2[ij] - r2[ij-1]) * B_fi * B_ur / 2 + (r4[ij] - r4[ij-1]) * A_fi * A_ur / 4 +
                (r3[ij] - r3[ij-1]) * (A_fi * B_ur + B_fi * A_ur) / 3)  # r * ur * fi
        r2fi += (r3[ij] - r3[ij-1]) * A_fi / 3 + (r4[ij] - r4[ij-1]) * B_fi / 4  # r^2 * fi
        r4fi += (r5[ij] - r5[ij-1]) * A_fi / 5 + (r6[ij] - r6[ij-1]) * B_fi / 6  # r^4 * fi
        if rr[ij] <= D_2_g:  # D * 0.5 / gamma
            A_ub = (Ub[i,j,ij-1] * rr[ij] - Ub[i,j,ij] * rr[ij - 1]) / dr
            B_ub = (Ub[i,j,ij] - Ub[i,j,ij-1]) / dr
            qp2 +=  (r3[ij] - r3[ij-1]) * A_ub / 3 + (r4[ij] - r4[ij-1]) * B_ub / 4  # ub * r^2

        # Обновление функции пор по размерам
        fi[i, j, ij] = upd_fi(fi[i, j, ij], Ur[i, j, ij], fi[i, j, ij - 1],
                              Ur[i, j, ij-1], dr, Ub[i, j, ij])

    new_qp[i, j] = m[i, j] * (2.0 * qp1 + Wps[i, j] * qp2) / r2fi
    m_mult[i, j] = r2fi / integr_r2_fi0
    k_mult[i, j] = r4fi / integr_r4_fi0


@ti.func
def upd_fi(fi: float, Ur: float, fi1: float, Ur1: float, dr: float, Ub: float) -> float:
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
