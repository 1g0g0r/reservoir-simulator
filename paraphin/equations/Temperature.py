"""Решение уравнения температуры по явной схеме."""
import taichi as ti

from paraphin.constants import Nx, Ny, dt, volume, ro_w, ro_f, ro_o, ro_p


@ti.func
def temperature_equation(i, j, T, m, m_0, S, S_0, C_o, C_w, C_f, C_p, Wps, Wps_0, qp, cells_T_eq, new_T) -> None:
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
        Теплоемкость нефти, [Дж/(кг*C)]
    C_w: taichi.field(Nx, Ny)
        Теплоемкость воды, [Дж/(кг*C)]
    C_f: taichi.field(Nx, Ny)
        Теплоемкость пласта, [Дж/(кг*C)]
    C_p: taichi.field(Nx, Ny)
        Теплоемкость парафина, [Дж/(кг*C)]
    Wps: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина, [-]
    Wps_0: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина на старом временном слое, [-]
    qp: taichi.field(Nx, Ny)
         Скорость отложения парафиновых отложений в общем объеме пористой породы
    cells_T_eq: taichi.field(Nx, Ny)
		Сумма величин перетоков тепла в уравнении энергии
    new_T: taichi.field(Nx, Ny)
        Температура на новом временном слое, [С]
    """
    derivative_add = T[i, j] * volume * (
            ro_w * C_w[i,j] * (m[i,j] * S[i,j] - m_0[i,j] * S_0[i,j]) / dt +
            ro_o * C_o[i,j] * (m[i,j] * (1.0-S[i,j]) * (1.0-Wps[i,j]) - m_0[i,j] * (1.0-S_0[i,j]) * (1.0-Wps_0[i,j])) / dt +
            ro_o * C_o[i,j] * (m[i,j] * Wps[i,j] * (1.0-S[i,j]) - m_0[i,j] * Wps_0[i,j] * (1.0-S_0[i,j])) / dt  -
            ro_f * C_f[i,j] * (m[i,j] - m_0[i,j]) / dt)

    multiplier = (m[i, j] * (S[i, j] * ro_w * C_w[i, j] + (1.0 - S[i, j]) * (ro_o * C_o[i,j] * (1.0 - Wps[i,j]) +
                                     ro_o * C_o[i,j] * Wps[i,j])) + (1.0 - m[i,j]) * ro_f * C_f[i,j]) * volume / dt

    new_T[i, j] = T[i, j] + (cells_T_eq[i, j] - derivative_add + qp[i, j] * ro_p * C_p[i, j] * volume) / multiplier
    # TODO убрать нафиг множитель qp[i, j] * ro_p * C_p[i, j] * volume 😊😊😊


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
        Теплоемкость нефти, [Дж/(кг*C)]
    C_w: taichi.field(Nx, Ny)
        Теплоемкость воды, [Дж/(кг*C)]
    C_f: taichi.field(Nx, Ny)
        Теплоемкость пласта, [Дж/(кг*C)]
    C_p: taichi.field(Nx, Ny)
        Теплоемкость парафина, [Дж/(кг*C)]
    Wps: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина, [-]
    new_T: taichi.field(Nx, Ny)
        Температура на новом временном слое, [С]
    """
    i, j = well.i, well.j
    Twell = 0.0

    # Если скважина нагнетательная, то учитывается ее температура
    if well.is_injector == 1:
        Twell = well.T
    else:
        Twell = T[i, j]

    multiplier = (m[i, j] * (S[i, j] * ro_w * C_w[i, j] + (1.0 - S[i, j]) * (ro_o * C_o[i, j] * (1.0 - Wps[i, j]) +
                                    ro_o * C_o[i, j] * Wps[i, j])) + (1.0 - m[i, j]) * ro_f * C_f[i, j]) * volume / dt

    new_T[i, j] -= (C_o[i, j] * ro_o * well.q[0] + C_w[Nx - 1, Ny - 1] * ro_w * well.q[1]) / multiplier * Twell

    # Если скважина добывающая, то температура определяется температурой в соседних ячейках
    # else:
    #     t_aver = 0.0
    #     num = 0
    #
    #     arr = [[i + 1, j], [i - 1, j], [i, j + 1], [i, j - 1]]
    #     for qq in ti.static(ti.ndrange(4)):
    #         i1, j1 = arr[qq]
    #         if (0 <= i1 < Nx) and (0 <= j1 < Ny):
    #             t_aver += T[i1, j1]
    #             num += 1
    #
    #     new_T[i, j] = t_aver / num
