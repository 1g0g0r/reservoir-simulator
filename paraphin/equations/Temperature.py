import taichi as ti

from paraphin.constants import Nx, Ny, dt, volume, ro_w, ro_f, ro_o, ro_p


@ti.func
def temperature_equation(i, j, T, m, m_0, S, S_0, C_o, C_w, C_f, C_p, Wp, Wp_0, Wps, Wps_0,
                         up_kw_val, up_ko_val, dt_val, new_T) -> None:
    """Вычисление температуры по явной схеме.

    Parameters
    ----------
    i, j : int
        Индексы текущей ячейки, [-]
    T: taichi.field(Nx, Ny)
        Температура, [С]
    m: taichi.field(Nx, Ny)
        Пористость, [-]
    m_0: taichi.field(Nx, Ny)
        Пористость на старом временном слое, [-]
    S: taichi.field(Nx, Ny)
        Водонасыщенность, [-]
    S_0: taichi.field(Nx, Ny)
        Водонасыщенность на старом временном слое, [-]
    C_o: taichi.field(Nx, Ny)
        Теплоемкость нефти, [Дж/C]
    C_w: taichi.field(Nx, Ny)
        Теплоемкость воды, [Дж/C]
    C_f: taichi.field(Nx, Ny)
        Теплоемкость пласта, [Дж/C]
    C_p: taichi.field(Nx, Ny)
        Теплоемкость парафина, [Дж/C]
    Wp: taichi.field(Nx, Ny)
        Концентрация растворенного парафина, [-]
    Wp_0: taichi.field(Nx, Ny)
        Концентрация растворенного парафина на старом временном слое, [-]
    Wps: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина, [-]
    Wps_0: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина на старом временном слое, [-]
    up_kw_val: taichi.field(Nx, Ny)
        Перетоки воды в ячейках, [Па*м]
    up_ko_val: taichi.field(Nx, Ny)
        Перетоки нефти в ячейках, [Па*м]
    dt_val: taichi.field(Nx, Ny)
        Величина (T_i - T_j) * area / h_ij, [C*м]
    new_T: taichi.field(Nx, Ny)
        Температура на новом временном слое, [С]
    """
    derivative_add = T[i, j] * (ro_w * C_w[i,j] * (m[i,j]*S[i,j] - m_0[i,j]*S_0[i,j])/dt + ro_o * C_o[i,j] *
        (m[i,j]*(1.0-S[i,j]) - m_0[i,j]*(1.0-S_0[i,j])) / dt + ro_p * C_p[i,j] * ((Wp[i,j]-Wp_0[i,j]) / dt +
        (m[i,j]*(1.0-S[i,j])*Wps[i,j] - m_0[i,j]*(1.0-S_0[i,j])*Wps_0[i,j]) / dt)) * volume

    # if (i == 0 and j == 0) or (i == Nx-1 and j == Ny-1):
    #     print((i, j), derivative_add)
    multiplier = dt / volume / (m[i, j] * S[i, j] * ro_w * C_w[i, j] + m[i, j] * (1.0 - S[i, j]) * ro_o * C_o[i, j] +
                       (m[i, j] * (1.0 - S[i, j]) * Wps[i, j] + Wp[i, j]) * ro_p * C_p[i, j] +
                                (1.0 - m[i, j] - Wp[i, j]) * ro_f * C_f[i, j])

    t1 = dt_val[i, j]

    t2 = T[i, j] * ro_w * C_w[i, j] * up_kw_val[i, j]

    t3 = T[i, j] * (ro_o * C_o[i, j] * (1.0 - Wps[i, j]) + ro_p * Wps[i, j] * C_p[i, j]) * up_ko_val[i, j]

    new_T[i, j] = T[i, j] + multiplier * (t1 + t2 + t3 - derivative_add)


def temperature_well(well, T, m, S, C_o, C_w, C_f, C_p, Wp, Wps, new_T) -> None:
    """Учет скважины в уравнении энергии.

    Parameters
    ----------
    well: Well
        Объект класса скважина
    T: taichi.field(Nx, Ny)
        Температура, [С]
    m: taichi.field(Nx, Ny)
        Пористость, [-]
    S: taichi.field(Nx, Ny)
        Водонасыщенность, [-]
    C_o: taichi.field(Nx, Ny)
        Теплоемкость нефти, [Дж/C]
    C_w: taichi.field(Nx, Ny)
        Теплоемкость воды, [Дж/C]
    C_f: taichi.field(Nx, Ny)
        Теплоемкость пласта, [Дж/C]
    C_p: taichi.field(Nx, Ny)
        Теплоемкость парафина, [Дж/C]
    Wp: taichi.field(Nx, Ny)
        Концентрация растворенного парафина, [-]
    Wps: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина, [-]
    new_T: taichi.field(Nx, Ny)
        Температура на новом временном слое, [С]
    """
    i, j = well.i, well.j
    Twell = T[i, j] if well.T is None else well.T

    mult =  dt / volume / (m[i, j] * S[i, j] * ro_w * C_w[i, j] + m[i, j] * (1.0 - S[i, j]) * ro_o * C_o[i, j] +
                           (m[i, j] * (1.0 - S[i, j]) * Wps[i, j] + Wp[i, j]) * ro_p * C_p[i, j] +
                           (1.0 - m[i, j] - Wp[i, j]) * ro_f * C_f[i, j])

    new_T[i, j] -= (C_o[i, j] * ro_o * well.q[0] + C_w[Nx - 1, Ny - 1] * ro_w * well.q[1]) * mult * Twell
