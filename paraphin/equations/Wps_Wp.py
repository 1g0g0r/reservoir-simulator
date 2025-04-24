import taichi as ti

from paraphin.constants import dt, ro_p, ro_o, volume, Tm, R

# Операции с константными величинами (вычисляются один раз только при импорте модуля)
temp = 1.0 / (1.8 * Tm + 32.0)


@ti.func
def wps_wp_equation(i, j, qp, m, m_0, S, S_0, Wp, Wp_0, Wps, T, T_0, C_p, up_ko_val, new_Wp, new_Wps) -> None:
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
    Wp_0: taichi.field(Nx, Ny)
        Концентрация растворенного парафина на прошлом временном слое, [-]
    Wps: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина, [-]
    T: taichi.field(Nx, Ny)
        Температура, [С]
    T_0: taichi.field(Nx, Ny)
        Температура на прошлом временном слое, [С]
    C_p: taichi.field(Nx, Ny)
        Теплоемкость парафина, [Дж/C]
    up_ko_val: taichi.field(Nx, Ny)
        Перетоки нефти в ячейках, [Па*м]
    new_Wp: taichi.field(Nx, Ny)
        Концентрация растворенного парафина на новом временном слое, [-]
    new_Wps: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина на новом временном слое, [-]
    """
    new_Wps[i, j] = Wps[i, j] + dt / (m[i, j] * (1.0 - S[i, j]) * ro_p) * ((Wps[i, j] * ro_p + ro_o * Wp[i, j]) *
                (-(m[i, j] * (1.0 - S[i, j]) - m_0[i, j] * (1.0 - S_0[i, j])) / dt + up_ko_val[i, j] * volume) -
                ro_o * Wp[i, j] * (1.0 - S[i, j]) * (Wp[i, j] - Wp_0[i, j]) / dt - ro_p * qp[i, j])

    # delta_Hp = Cp * delta (T) - молярные доли парафина, растворенные в нефти
    delta_Hp = C_p[i, j] * (T[i, j] - T_0[i, j]) * 1.8   # * 1.8 - перевод в фаренгейты
    new_Wp[i, j] = Wps[i, j] * ti.exp(delta_Hp / R * (temp - 1.0 / (1.8 * T[i, j] + 32.0)))


@ti.func
def wps_wp_wells(well, m, S, Wp, Wps, new_Wps) -> None:
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
        Концентрация взвешенных частиц парафина, [-]
        Концентрация растворенного парафина на новом временном слое, [-]
    new_Wps: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина на новом временном слое, [-]
    """
    i, j = well.i, well.j
    new_Wps[i, j] -= (dt * well.q[0] * (Wp[i, j] * ro_o + Wps[i, j] * ro_p) / (m[i, j] * (1.0 - S[i, j]) * ro_p))
