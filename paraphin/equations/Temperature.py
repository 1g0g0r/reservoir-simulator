"""Решение уравнения температуры по явной схеме."""
import taichi as ti

from paraphin.constants import dt, volume, ro_w, ro_f, ro_o, ro_p, Twater


@ti.func
def temperature_equation(i, j, T, m, m_0, S, S_0, C_o, C_w, C_f, C_p, Wps, Wps_0, qp, cells_T_eq, new_T, new_m, new_S) -> None:
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
    new_m: taichi.field(Nx, Ny)
        Пористость на новом временном слое, [-]
    new_S: taichi.field(Nx, Ny)
        Водонасыщенность на новом временном слое, [-]
    """
    psi = _psi(i, j, m, S, Wps, C_w, C_o, C_p, C_f)
    psi_next = _psi(i, j, new_m, new_S, Wps, C_w, C_o, C_p, C_f)
    derivative_add = T[i, j] * volume * (psi_next - psi) / dt

    new_T[i, j] += T[i, j] + dt / psi_next / volume * (cells_T_eq[i, j] - derivative_add  + qp[i, j] * ro_p * C_p[i, j] * volume)


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
    if well.is_injector == 1:  # Если скважина нагнетательная, то учитывается ее температура
        Twell = well.T
    else:
        Twell = T[i, j]

    multiplier = dt / _psi(i, j, m, S, Wps, C_w, C_o, C_p, C_f) / volume

    new_T[i, j] -= (C_o[i, j] * ro_o * well.q[0] + C_w[i, j] * ro_w * well.q[1]) * multiplier * Twell


@ti.func
def _psi(i, j, m, S, Wps, C_w, C_o, C_p, C_f):
    # TODO уточнить энергию осевшего на порах парафина
    return (m[i, j] * (S[i, j] * ro_w * C_w[i, j] + (1.0 - S[i, j]) * (ro_o * C_o[i, j] * (1.0 - Wps[i, j]) +
                                     ro_p * C_p[i, j] * Wps[i, j])) + (1.0 - m[i, j]) * ro_f * C_f[i, j])


@ti.func
def _get_top_bottom_heat_losses():
    """Вычисление потерь тепла через кровлю и подошву пласта по методу Ловерье."""
    t_loss = 0.0

    ksi = 4.0 * lam / (V_o * C_o * rho_o + V_o * C_o * rho_o) / h
    teta = 4.0 * lam * t / (C_f * rho_f) / h / h

    if teta > ksi:
        t_loss = (Twater - T_init) * ti.erfc(ksi / ti.sqrt((C_f * rho_f)/(C * rho) * (teta-ksi)) * 0.5)

    return t_loss
