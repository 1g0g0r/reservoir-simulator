import taichi as ti

from paraphin.constants import dt, ro_p, ro_o, volume, Tm, R, alpha, data_type, init_Wp


@ti.func
def wps_wp_equation(i, j, qp, m, m_0, S, S_0, Wp, Wp_0, Wps, Wps_0, T, T_0, cells_Wp_eq, new_Wp, new_Wps) -> None:
    """Вычисление концентрации взвешенных частиц (Wps) и растворенного парафина (Wp) парафина по явной схеме.

    Parameters
    ----------
    i, j : int
        Индексы текущей ячейки, [-]
    qp: taichi.field(Nx, Ny)
         Скорость отложения парафиновых отложений в общем объеме пористой породы
    m: taichi.field(Nx, Ny)
        Пористость, [-]
    m_0: taichi.field(Nx, Ny)
        Пористость на прошлом временном слое, [-]
    S: taichi.field(Nx, Ny)
        Водонасыщенность, [-]
    S_0: taichi.field(Nx, Ny)
        Водонасыщенность на прошлом временном слое, [-]
    Wp: taichi.field(Nx, Ny)
        Концентрация растворенного парафина, [-]
    T: taichi.field(Nx, Ny)
        Температура, [С]
    cells_Wp_eq: taichi.field(Nx, Ny)
        Перетоки нефти в ячейках, [Па*м]
    new_Wp: taichi.field(Nx, Ny)
        Концентрация растворенного парафина на новом временном слое, [-]
    new_Wps: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина на новом временном слое, [-]
    """
    Wps_i   = _get_Wps(Wp[i, j], Wps[i, j], T[i, j])
    Wps_0_i = _get_Wps(Wp[i, j], Wps[i, j], T_0[i, j])  # _get_Wps(Wp_0[i, j], Wps_0[i, j], T_0[i, j])

    _new_Wp = Wp[i, j] + dt / (m[i, j] * (1.0 - S[i, j]) * ro_o) * (
            - Wp[i, j] * ro_o * (m[i, j] * (1.0 - S[i, j]) - m_0[i, j] * (1.0 - S_0[i, j])) / dt
            - ro_p * (m[i, j] * (1.0 - S[i, j]) * Wps_i - m_0[i, j] * (1.0 - S_0[i, j]) * Wps_0_i) / dt
            + cells_Wp_eq[i, j] / volume + ro_p * qp[i, j])

    new_Wp[i, j] = ti.max(_new_Wp, 0.0)
    new_Wps[i, j] = ti.max(_get_Wps(new_Wp[i, j], Wps[i, j], T[i, j]), 0.0)

    # if i == j == 0:
    #     print(qp[i, j], dt / (m[i, j] * (1.0 - S[i, j]) * ro_o) * ro_p * qp[i, j] / volume)


@ti.func
def wps_wp_wells(well, m, S, Wp, Wps, new_Wp) -> None:
    """Вычисление концентрации взвешенных частиц (Wps) и растворенного парафина (Wp) парафина по явной схеме.

    Parameters
    ----------
    well: Well
        Объект класса скважина
    m: taichi.field(Nx, Ny)
        Пористость, [-]
    S: taichi.field(Nx, Ny)
        Водонасыщенность, [-]
    Wp: taichi.field(Nx, Ny)
        Концентрация растворенного парафина, [-]
    Wps: taichi.field(Nx, Ny)
        Концентрация взвешенного парафина, [-]
    new_Wp: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина на новом временном слое, [-]
    """
    i, j = well.i, well.j
    new_Wp[i, j] -= well.q[0] * (Wp[i, j] * ro_o + Wps[i, j] * ro_p) * dt / (m[i, j] * (1.0 - S[i, j]) * ro_o * volume)


@ti.func
def _get_Wps(Wp: data_type, Wps: data_type, T: data_type) -> data_type:
    """Моделирование процесса кристаллизации парафина."""
    new_Wps = Wps

    if Wp > 1e-6:
        new_Wps = Wp * ti.exp(alpha / R * (1.0 / T - 1.0 / Tm))

    return new_Wps
