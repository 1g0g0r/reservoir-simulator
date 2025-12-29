"""Решение уравнения температуры по явной схеме."""
import numpy as np
from numba import njit

from paraphin.constants import dt, volume, h, ro_w, ro_f, ro_ff, ro_o, ro_p, init_T, K_ff
from paraphin.utils.math_utils import ti_erfc
from paraphin.utils.math_utils.FVM_utils import _K_w, _K_o


@njit
def temperature_equation(i, j, T, m, S, C_o, C_w, C_f, C_ff, C_p, Wps, qp, cells_T_eq, t, k,
                         mu_o, mu_w, grad_p, new_T, new_m, new_S) -> None:
    """Вычисление температуры по явной схеме.

    Parameters
    ----------
    i, j : int
        Индексы текущей ячейки, [-]
    T: numpy.ndarray(Nx, Ny)
        Температура, [С]
    m: numpy.ndarray(Nx, Ny)
        Пористость, [-]
    S: numpy.ndarray(Nx, Ny)
        Водонасыщенность, [-]
    C_o: numpy.ndarray(Nx, Ny)
        Теплоемкость нефти, [Дж/(кг*C)]
    C_w: numpy.ndarray(Nx, Ny)
        Теплоемкость воды, [Дж/(кг*C)]
    C_f: numpy.ndarray(Nx, Ny)
        Теплоемкость пласта, [Дж/(кг*C)]
    C_ff: numpy.ndarray(Nx, Ny)
        Теплоемкость окружающих пород пласта, [Дж/(кг*C)]
    C_p: numpy.ndarray(Nx, Ny)
        Теплоемкость парафина, [Дж/(кг*C)]
    Wps: numpy.ndarray(Nx, Ny)
        Концентрация взвешенных частиц парафина, [-]
    qp: numpy.ndarray(Nx, Ny)
         Скорость отложения парафиновых отложений в общем объеме пористой породы
    cells_T_eq: numpy.ndarray(Nx, Ny)
		Сумма величин перетоков тепла в уравнении энергии
    new_T: numpy.ndarray(Nx, Ny)
        Температура на новом временном слое, [С]
    new_m: numpy.ndarray(Nx, Ny)
        Пористость на новом временном слое, [-]
    new_S: numpy.ndarray(Nx, Ny)
        Водонасыщенность на новом временном слое, [-]
    t: float
        Физическое время, прошедшее с начала моделирования задачи, [сек]
    k: numpy.ndarray(Nx, Ny)
        Проницаемость, [м^2]
    mu_o: numpy.ndarray(Nx, Ny)
		Вязкость нефти, [Па*с]
	mu_w: numpy.ndarray(Nx, Ny)
		Вязкость воды, [Па*с]
    grad_p: numpy.ndarray(Nx, Ny)
        Поле перепада давления, [Па/м]
    """
    psi = _psi(i, j, m, S, Wps, C_w, C_o, C_p, C_f)
    psi_next = _psi(i, j, new_m, new_S, Wps, C_w, C_o, C_p, C_f)
    derivative_add = T[i, j] * volume * (psi_next - psi) / dt
    T_losses = 0.0  #_top_bottom_heat_losses(i, j, t, k, S, T, mu_o, mu_w, C_o, C_w, C_f, C_ff, grad_p)

    new_T[i, j] += T[i, j] + dt / psi_next / volume * (cells_T_eq[i, j] - derivative_add - T_losses * volume +
                                                       qp[i, j] * ro_p * C_p[i, j] * volume)


@njit
def temperature_well(well, T, m, S, C_o, C_w, C_f, C_p, Wps, new_T) -> None:
    """Учет скважины в уравнении энергии.

    Parameters
    ----------
    well: Well
        Объект класса скважина
    T: numpy.ndarray(Nx, Ny)
        Температура, [С]
    m: numpy.ndarray(Nx, Ny)
        Пористость, [-]
    S: numpy.ndarray(Nx, Ny)
        Водонасыщенность, [-]
    C_o: numpy.ndarray(Nx, Ny)
        Теплоемкость нефти, [Дж/(кг*C)]
    C_w: numpy.ndarray(Nx, Ny)
        Теплоемкость воды, [Дж/(кг*C)]
    C_f: numpy.ndarray(Nx, Ny)
        Теплоемкость пласта, [Дж/(кг*C)]
    C_p: numpy.ndarray(Nx, Ny)
        Теплоемкость парафина, [Дж/(кг*C)]
    Wps: numpy.ndarray(Nx, Ny)
        Концентрация взвешенных частиц парафина, [-]
    new_T: numpy.ndarray(Nx, Ny)
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


@njit
def _psi(i, j, m, S, Wps, C_w, C_o, C_p, C_f):
    # TODO уточнить энергию осевшего на порах парафина
    return (m[i, j] * (S[i, j] * ro_w * C_w[i, j] + (1.0 - S[i, j]) * (ro_o * C_o[i, j] * (1.0 - Wps[i, j]) +
                                     ro_p * C_p[i, j] * Wps[i, j])) + (1.0 - m[i, j]) * ro_f * C_f[i, j])


@njit
def _top_bottom_heat_losses(i, j, t, k, S, T, mu_o, mu_w, C_o, C_w, C_f, C_ff, grad_p):
    """Вычисление потерь тепла через кровлю и подошву пласта по методу Ловерье.

    Parameters
    ----------
    i, j : int
        Индексы текущей ячейки, [-]
    t: float
        Текущее физическое время расчета, [с]
    S: numpy.ndarray(Nx, Ny)
        Водонасыщенность, [-]
    T: numpy.ndarray(Nx, Ny)
        Температура, [С]
    k: numpy.ndarray(Nx, Ny)
        Проницаемость пористой среды, [-]
    C_o: numpy.ndarray(Nx, Ny)
        Теплоемкость нефти, [Дж/(кг*C)]
    C_w: numpy.ndarray(Nx, Ny)
        Теплоемкость воды, [Дж/(кг*C)]
    C_f: numpy.ndarray(Nx, Ny)
        Теплоемкость пласта, [Дж/(кг*C)]
    C_ff: numpy.ndarray(Nx, Ny)
        Теплоемкость окружающих пород пласта, [Дж/(кг*C)]
    grad_p: numpy.ndarray(Nx, Ny)
        Градиент давления в центрах ячеек, [Па/м]
    """
    t_loss = 0.0
    V_o = grad_p[i, j] * _K_o(k[i, j], S[i, j], mu_o[i, j])
    V_w = grad_p[i, j] * _K_w(k[i, j], S[i, j], mu_w[i, j])

    teta = 4.0 * K_ff * t / (C_f[i, j] * ro_f) / h / h
    ksi = 4.0 * K_ff / (V_o * C_o[i, j] * ro_o + V_w * C_w[i, j] * ro_w) / h

    if teta > ksi:
        erfs_argument = ksi / np.sqrt((C_f[i, j] * ro_f) / (C_ff[i, j] * ro_ff) * (teta - ksi)) * 0.5
        t_loss = (T[i, j] - init_T) * ti_erfc(erfs_argument)

    return t_loss
