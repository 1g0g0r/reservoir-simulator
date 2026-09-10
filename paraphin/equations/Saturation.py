"""Решение уравнения водонасыщенности по явной схеме."""
from numba import njit

from paraphin.constants import volume


@njit(cache=True)
def saturation_equation(i, j, S, m, cells_S_eq, new_m, new_S, dt) -> None:
    """Вычисление водонасыщенности по явной схеме.

        S^new = S + [-S*(m^new - m) + dt*cells_S_eq/|V|] / m^new

    Дебит скважины входит в `cells_S_eq` наравне с перетоками через грани (см. `_wells_loop`),
    поэтому делится на ту же пористость нового слоя.

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    new_S[i, j] = S[i, j] + (-S[i, j] * (new_m[i, j] - m[i, j]) + dt * cells_S_eq[i, j] / volume) / new_m[i, j]
