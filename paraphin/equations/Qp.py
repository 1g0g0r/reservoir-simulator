from taichi import field, ndrange, func, kernel

from paraphin.constants import default_type, Nx, Ny, Nr, dt, ro_o, D, gamma
from .Velocitys import u_c, u_b, u_r


def calc_qp(Wps, mu_o, m, qp, fi, h_sloy, r, integr_r2_fi0, integr_r4_fi0, Um_r2) -> (field(dtype=default_type, shape=(Nx, Ny)),
                                                                                         field(dtype=default_type, shape=(Nx, Ny)),
                                                                                         field(dtype=default_type, shape=(Nx, Ny))):
    """
    Вычисление концентрации взвешенных частиц парафина по явной схеме

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
    """
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
                # TODO спросить про осадочный слой - что первее вычисляется ?
                ur = u_r(Wps[i,j], um, r[0], h_sloy[i, j, 0])
                ub = u_b(um, Wps[i,j], fi[i,j,0], r[0])
                uc = u_c(r[0], mu_o[i,j], ro_o[i, j])
                h_sloy[i, j, 0] = sed_h(h_sloy[i, j, 0], ur, r[0])

                for ij in ndrange(1, Nr):
                    # Проверить вычисление скоростей.
                    # Вынести расчет средней скорости до обновления давления.
                    um_new = Um_r2[i, j] * r2[ij]
                    ur_new = u_r(Wps[i, j], um_new, r[ij], h_sloy[i, j, ij])
                    ub_new = u_b(um_new, Wps[i, j], fi[i, j, ij], r[ij])
                    uc_new = u_c(r[ij], mu_o[i, j], ro_o[i, j])  # где используется крит скорость ??
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

                    # TODO уточнить когда обновляется
                    fi[i, j, ij] = upd_fi(fi[i, j, ij], ur_new, fi[i, j, ij - 1], ur, r[ij] - r[ij - 1], ub)

                qp[i, j] = m[i, j] * (2.0 * qp1 + Wps[i, j] * qp2) / r2fi
                m_mult[i, j] = r2fi / integr_r2_fi0
                k_mult[i, j] = r4fi / integr_r4_fi0

    calc_qp_loop()

    return qp, m_mult, k_mult


@func
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


@func
def sed_h(h0, ur, r):
    """
    Вычисление толщины осадочного слоя.

    Parameters
    ----------
    h0: float
        Толщина осадочного слоя, [m].
    ur: float
        Скорость изменения радиуса капилляра, [m/c]
    r: float
        Радиус капилляра, [m].

    Returns
    -------
    hr: float
        Толщина осадочного слоя, [m].
    """
    hr = h0 - dt * ur
    hr = max(0.0, min(hr, r - 1e-7))
    return hr
