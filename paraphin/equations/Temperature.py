import taichi as ti

from paraphin.constants import Nx, Ny, dt, volume, ro_w, ro_f, ro_o, ro_p


@ti.func
def temperature_equation(i, j, T, m, m_0, S, S_0, C_o, C_w, C_f, C_p, Wps, Wps_0, temp_eq_val, new_T) -> None:
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
    Wps: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина, [-]
    Wps_0: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина на старом временном слое, [-]
    temp_eq_val: taichi.field(Nx, Ny)
		Сумма величин перетоков тепла в уравнении энергии
    new_T: taichi.field(Nx, Ny)
        Температура на новом временном слое, [С]
    """
    derivative_add = T[i, j] * volume * (
            ro_w * C_w[i,j] * (m[i,j]*S[i,j] - m_0[i,j]*S_0[i,j]) / dt +
            ro_o * C_o[i,j] * (m[i,j]*(1.0-S[i,j])*(1.0-Wps[i,j]) - m_0[i,j]*(1.0-S_0[i,j])*(1.0-Wps_0[i,j])) / dt +
            ro_p * C_p[i,j] * (m[i,j]*Wps[i,j]*(1.0-S[i,j]) - m_0[i,j]*Wps_0[i,j]*(1.0-S_0[i,j])) / dt  -
            ro_f * C_f[i,j] * (m[i,j] - m_0[i,j]) / dt)

    multiplier = (m[i, j] * (S[i, j] * ro_w * C_w[i, j] + (1.0 - S[i, j]) * (ro_o * C_o[i,j] * (1.0 - Wps[i,j]) +
                                     ro_p * C_p[i,j] * Wps[i,j])) + (1.0 - m[i,j]) * ro_f * C_f[i,j]) * volume / dt

    new_T[i, j] = T[i, j] + (temp_eq_val[i, j] - derivative_add) / multiplier


@ti.func
def temperature_well(well, T, m, S, C_o, C_w, C_f, C_p, Wps, new_T) -> None:
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
    Wps: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина, [-]
    new_T: taichi.field(Nx, Ny)
        Температура на новом временном слое, [С]
    """
    i, j = well.i, well.j
    Twell = T[i, j] if well.T == -9999 else well.T

    multiplier = (m[i, j] * (S[i, j] * ro_w * C_w[i, j] + (1.0 - S[i, j]) * (ro_o * C_o[i, j] * (1.0 - Wps[i, j]) +
                                 ro_p * C_p[i, j] * Wps[i, j])) + (1.0 - m[i, j]) * ro_f * C_f[i, j]) * volume / dt

    new_T[i, j] -= (C_o[i, j] * ro_o * well.q[0] + C_w[Nx - 1, Ny - 1] * ro_w * well.q[1]) / multiplier * Twell
