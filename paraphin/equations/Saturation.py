"""Решение уравнения водонасыщенности по явной схеме."""
from numba import njit

from paraphin.constants import volume


@njit(cache=True)
def saturation_equation(i, j, S, m, cells_S_eq, new_m, new_S, dt) -> None:
    """Вычисление водонасыщенности по явной схеме.
    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    new_S[i, j] += S[i, j] + (-S[i, j] * (new_m[i, j] - m[i, j]) + dt * cells_S_eq[i, j] / volume) / new_m[i, j]


@njit(cache=True)
def saturation_well(well, m, new_S, dt) -> None:
    """Учет отбора/закачки воды скважиной в уравнении водонасыщенности.

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    i, j = well.i, well.j
    new_S[i, j] -= dt * well.q[1] / m[i, j] / volume
