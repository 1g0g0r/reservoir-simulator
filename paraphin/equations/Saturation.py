"""Решение уравнения водонасыщенности по явной схеме."""
from numba import njit

from paraphin.constants import volume, S_min, S_max


@njit(cache=True)
def saturation_equation(i, j, S, m, m_0, cells_S_eq, new_m, new_S, dt, clip_field) -> None:
    """Вычисление водонасыщенности по явной схеме.

    Parameters
    ----------
    i, j : int
        Индексы текущей ячейки, [-]
    S: numpy.ndarray(Nx, Ny)
        Водонасыщенность, [-]
    m: numpy.ndarray(Nx, Ny)
        Пористость, [-]
    m_0: numpy.ndarray(Nx, Ny)
        Пористость на прошлом временном слое, [-]
    cells_S_eq: numpy.ndarray(Nx, Ny)
        Перетоки воды в ячейках, [Па*м]
    new_m: numpy.ndarray(Nx, Ny)
        Пористость на новом временном слое, [-]
    new_S: numpy.ndarray(Nx, Ny)
        Водонасыщенность на новом временном слое, [-]
    dt: float
        Текущий шаг по времени, [с]
    clip_field: numpy.ndarray(Nx, Ny)
        Величина выхода за физические границы по ячейкам; ноль, если обрезания не было.
        Поячеечно, а не общим счетчиком, чтобы цикл по ячейкам можно было распараллелить.
    """
    new_S[i, j] += S[i, j] + (-S[i, j] * (new_m[i, j] - m[i, j]) + dt * cells_S_eq[i, j] / volume) / new_m[i, j]

    # Выход за физические границы означает нарушенный баланс (как правило, превышен предел
    # устойчивости явной схемы). Обрезаем, но считаем нарушения: молчаливое обрезание скрыло бы причину.
    s_new = new_S[i, j]
    if s_new < S_min:
        clip_field[i, j] = S_min - s_new
        new_S[i, j] = S_min
    elif s_new > S_max:
        clip_field[i, j] = s_new - S_max
        new_S[i, j] = S_max
    else:
        clip_field[i, j] = 0.0


@njit(cache=True)
def saturation_well(well, m, new_m, new_S, dt) -> None:
    """Учет скважины в уравнении водонасыщенности.

    Parameters
    ----------
    well: Well
        Объект класса скважина
    m: numpy.ndarray(Nx, Ny)
        Пористость, [-]
    new_m: numpy.ndarray(Nx, Ny)
        Пористость на новом временном слое, [-]
    new_S: numpy.ndarray(Nx, Ny)
        Водонасыщенность на новом временном слое, [-]
    dt: float
        Текущий шаг по времени, [с]
    """
    i, j = well.i, well.j
    new_S[i, j] -= dt * well.q[1] / m[i, j] / volume
