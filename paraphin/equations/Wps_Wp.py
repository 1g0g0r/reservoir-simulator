"""Вычисление концентрации взвешенных частиц (Wps) и растворенного парафина (Wp) парафина по явной схеме."""
import taichi as ti

from paraphin.constants import dt, ro_p, ro_o, volume, Tm, R, alpha, data_type, init_Wp, init_T

reverse_Tm = 1.0 / Tm
alpha_R = alpha / R


@ti.func
def wps_wp_equation(i, j, qp, m, m_0, S, S_0, Wo, Wp, Wp_0, Wps, Wps_0, T, T_0, cells_Wp_eq, new_Wp, new_Wps) -> None:
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
    Wo: taichi.field(Nx, Ny)
        Концентрация нефтяного компонента в нефти, [-]
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
    if T[i, j] < init_T * 0.95:
        if Wp[i, j] > 1e-6:
            Wps_i  = _get_Wps(Wp[i, j], Wps[i, j], T[i, j])
            Wps_0_i  = _get_Wps(Wp[i, j], Wps[i, j], T_0[i, j])

            _new_Wp = Wp[i, j] + dt / (m[i, j] * (1.0 - S[i, j]) * ro_o) * (
                    - Wp[i, j] * ro_o * (m[i, j] * (1.0 - S[i, j]) - m_0[i, j] * (1.0 - S_0[i, j])) / dt
                    - ro_p * (m[i, j] * (1.0 - S[i, j]) * Wps_i - m_0[i, j] * (1.0 - S_0[i, j]) * Wps_0_i) / dt
                    + cells_Wp_eq[i, j] / volume + ro_p * qp[i, j])

            colmatation = qp[i, j] * dt * ro_p / ((1.0-Wps[i,j]) * ro_o + Wps[i,j] * ro_p)
            new_Wp[i, j] += ti.max(_new_Wp, 0.0)
            new_Wps[i, j] = ti.max(init_Wp - new_Wp[i, j] + colmatation , 0.0)

        else:
            colmatation = qp[i, j] * dt * ro_p / ((1.0-Wps[i,j]) * ro_o + Wps[i,j] * ro_p)
            new_Wps[i, j] = ti.max(Wps[i, j] + colmatation, 0)
    else:
        new_Wp[i, j] = Wp[i, j]


@ti.func
def _get_Wps(Wp: data_type, Wps: data_type, T: data_type) -> data_type:
    """Моделирование процесса кристаллизации парафина."""
    new_Wps = Wps

    # exact_solution = alpha_R * Tm / (alpha_R + Tm * ti.log(border))
    if Wp > 1e-6 and T > 0.9 * Tm:
        new_Wps = Wp * ti.exp(alpha_R * (1.0 / T - reverse_Tm))

    return new_Wps


@ti.func
def wps_wp_wells(well, m, S, T, Wp, Wps, new_Wp) -> None:
    """Вычисление массовой доли взвешенных частиц (Wps) и растворенного парафина (Wp) парафина в нефти по явной схеме.

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
    if T[i, j] < init_T * 0.95 and Wp[i, j] > 1e-6:
        new_Wp[i, j] -= well.q[0] * (Wp[i, j] * ro_o + Wps[i, j] * ro_p) * dt / (m[i, j] * (1.0 - S[i, j]) * ro_o * volume)
