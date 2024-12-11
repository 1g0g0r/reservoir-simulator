import taichi as ti

from paraphin.constants import data_type, r, Nx, Ny, Nr, dt, D, gamma, DEBUGGING
from paraphin.utils.vizualization import show_plot

# Операции с константными величинами (вычисляются один раз только при импорте модуля)
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


def calc_qp(Wps, m, fi, Ur, Ub, integr_r2_fi0, integr_r4_fi0) -> (ti.field(dtype=data_type, shape=(Nx, Ny)),
                                                                           ti.field(dtype=data_type, shape=(Nx, Ny)),
                                                                           ti.field(dtype=data_type, shape=(Nx, Ny))):
    """
    Вычисление концентрации взвешенных частиц парафина по явной схеме

    Parameters
    ----------
    Wps: taichi.field(Nx, Ny)
        Концентрации взвешенных частиц парафина, [-]
    m: taichi.field(Nx, Ny)
        Пористость, [-]
    qp: taichi.field(Nx, Ny)
        Скорость отложения парафина в общем объеме, [-]
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

    Returns
    -------
    qp: taichi.field(Nx, Ny)
         Скорость отложения парафиновых отложений в общем объеме пористой породы
    m_mult: taichi.field(Nx, Ny)
        Изменение пористрости из-за влияния частиц парафина, [-]
    k_mult: taichi.field(Nx, Ny)
        Изменение проницаемости из-за влияния частиц парафина, [-]
    """
    new_qp = ti.field(dtype=data_type, shape=(Nx, Ny))
    k_mult = ti.field(dtype=data_type, shape=(Nx, Ny))
    m_mult = ti.field(dtype=data_type, shape=(Nx, Ny))

    @ti.kernel
    def calc_qp_loop():
        for i in ti.ndrange(Nx):
            for j in ti.ndrange(Ny):
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
                            (r3[ij] - r3[ij-1]) * (A_fi * B_ur + B_fi * A_ur) / 3)
                    r2fi += (r3[ij] - r3[ij-1]) * A_fi / 3 + (r4[ij] - r4[ij-1]) * B_fi / 4
                    r4fi += (r5[ij] - r5[ij-1]) * A_fi / 5 + (r6[ij] - r6[ij-1]) * B_fi / 6
                    if rr[ij] <= D_2_g:  # D * 0.5 / gamma
                        A_ub = (Ub[i,j,ij-1] * rr[ij] - Ub[i,j,ij] * rr[ij - 1]) / dr
                        B_ub = (Ub[i,j,ij] - Ub[i,j,ij-1]) / dr
                        qp2 +=  (r3[ij] - r3[ij-1]) * A_ub / 3 + (r4[ij] - r4[ij-1]) * B_ub / 4

                    # Обновление функции пор по размерам
                    fi[i, j, ij] = upd_fi(fi[i, j, ij], Ur[i, j, ij], fi[i, j, ij - 1], Ur[i, j, ij-1],
                                          dr, Ub[i, j, ij])

                new_qp[i, j] = m[i, j] * (2.0 * qp1 + Wps[i, j] * qp2) / r2fi
                m_mult[i, j] = r2fi / integr_r2_fi0
                k_mult[i, j] = r4fi / integr_r4_fi0

    calc_qp_loop()
    if DEBUGGING:
        show_plot(new_qp.to_numpy(), 'Qp')
    return new_qp, m_mult, k_mult


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


"""
def calc_qp_deprecated(Wps, mu_o, m, qp, fi, h_sloy, r, integr_r2_fi0, integr_r4_fi0, Um_r2) -> (field(dtype=default_type, shape=(Nx, Ny)),
                                                                                      field(dtype=default_type, shape=(Nx, Ny)),
                                                                                      field(dtype=default_type, shape=(Nx, Ny))):
    '''Вычисление концентрации взвешенных частиц парафина по явной схеме

    Parameters
    ----------
    Wps: taichi.field(Nx, Ny)
        Концентрации взвешенных частиц парафина
    m: taichi.field(Nx, Ny)
        Пористость
    qp: taichi.field(Nx, Ny)
        Скорость отложения парафина в общем объеме
    fi: taichi.field(Nx, Ny, Nr)
        Функция распределения пор по размеру
    h_sloy: taichi.field(Nx, Ny, Nr)
        Толщина осадочного слоя
    r: taichi.field(Nr)
        Радиусы пор
    integr_r2_fi0: float
        Интеграл r^2 * fi_o(r)
    integr_r4_fi0: float
        Интеграл r^4 * fi_o(r)
    Um_r2: numpy.ndarray
        Средняя скорость в канале разделенная на r^2

    Returns
    -------
    qp: taichi.field(Nx, Ny)
         Скорость отложения парафиновых отложений в общем объеме пористой породы
    fi: taichi.field(Nx, Ny, Nr)
        Обновленная функция распределения пор по размеру
    '''
    k_mult = field(dtype=default_type, shape=(Nx, Ny))
    m_mult = field(dtype=default_type, shape=(Nx, Ny))

    r2 = r * r
    r3 = r2 * r
    r4 = r3 * r
    r5 = r4 * r
    r6 = r5 * r

    @kernel
    def calc_qp_loop():
        for i in ndrange(Nx):
            for j in ndrange(Ny):
                qp1 = 0.0
                qp2 = 0.0
                r2fi = 0.0
                r4fi = 0.0

                # Расчет скоростей Ur, Ub, Uc
                um = Um_r2[i, j] * r2[0]
                ub = u_b(um, Wps[i,j], fi[i,j,0], r[0])
                uc = u_c(r[0], mu_o[i,j], ro_o[i, j])
                ur = u_r(Wps[i,j], um, uc, r[0], h_sloy[i, j, 0])
                h_sloy[i, j, 0] = sed_h(h_sloy[i, j, 0], ur, r[0])

                for ij in ndrange(1, Nr):
                    # Проверить вычисление скоростей.
                    # Вынести расчет средней скорости до обновления давления.
                    um_new = Um_r2[i, j] * r2[ij]
                    ub_new = u_b(um_new, Wps[i, j], fi[i, j, ij], r[ij])
                    uc_new = u_c(r[ij], mu_o[i, j], ro_o[i, j])
                    ur_new = u_r(Wps[i, j], um_new, uc_new, r[ij], h_sloy[i, j, ij])
                    h_sloy[i, j, ij] = sed_h(h_sloy[i, j, ij], ur_new, r[ij])

                    dr = r[ij] - r[ij-1]
                    A_fi = (fi[i,j,ij-1] * r[ij] - fi[i,j,ij] * r[ij-1]) / dr
                    B_fi = (fi[i,j,ij] - fi[i,j,ij-1]) / dr
                    A_ur = (ur * r[ij] - ur_new * r[ij - 1]) / dr
                    B_ur = (ur_new - ur) / dr

                    qp1 += (r3[ij] - r3[ij-1]) * A_ur / 3 + (r4[ij] - r4[ij-1]) * B_ur / 4
                    r2fi += (r3[ij] - r3[ij-1]) * A_fi / 3 + (r4[ij] - r4[ij-1]) * B_fi / 4
                    r4fi += (r5[ij] - r5[ij-1]) * A_fi / 5 + (r6[ij] - r6[ij-1]) * B_fi / 6
                    if r[ij] <= D * 0.5 / gamma:
                        A_ub = (ub * r[ij] - ub_new * r[ij - 1]) / dr
                        B_ub = (ub_new - ub) / dr
                        qp2 +=  ((r2[ij] - r2[ij-1]) * B_fi * B_ur / 2 + (r4[ij] - r4[ij-1]) * A_fi * A_ur / 4 +
                                 (r3[ij] - r3[ij-1]) * (A_fi * B_ur + B_fi * A_ur) / 3)

                    # Обновление скоростей
                    um = um_new
                    ur = ur_new
                    ub = ub_new

                    fi[i, j, ij] = upd_fi(fi[i, j, ij], ur_new, fi[i, j, ij - 1], ur, r[ij] - r[ij - 1], ub)

                qp[i, j] = m[i, j] * (2.0 * qp1 + Wps[i, j] * qp2) / r2fi
                m_mult[i, j] = r2fi / integr_r2_fi0
                k_mult[i, j] = r4fi / integr_r4_fi0

    calc_qp_loop()

    return qp, m_mult, k_mult
"""