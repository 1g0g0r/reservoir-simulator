"""Решение уравнения водонасыщенности по явной схеме."""
from numba import njit

from paraphin.constants import dt, volume, init_m


@njit
def saturation_equation(i, j, S, m, m_0, cells_S_eq, new_m, new_S) -> None:
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
    """
    new_S[i, j] += S[i, j] + (-S[i, j] * (new_m[i, j] - m[i, j]) + dt * cells_S_eq[i, j] / volume) / new_m[i, j]


@njit
def saturation_well(well, m, new_m, new_S) -> None:
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
    """
    i, j = well.i, well.j
    new_S[i, j] -= dt * well.q[1] / m[i, j] / volume
