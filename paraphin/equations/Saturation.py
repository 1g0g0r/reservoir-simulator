"""Решение уравнения водонасыщенности по явной схеме."""
from numba import njit

from paraphin.constants import volume, S_min, S_max


@njit(cache=True)
def saturation_equation(i, j, S, m, cells_S_eq, new_m, new_S, dt, clip_field) -> None:
    """Вычисление водонасыщенности по явной схеме.

    Выход за физические границы [S_min, S_max] означает нарушенный баланс (как правило, превышен
    предел устойчивости явной схемы). Обрезаем, но записываем величину выхода в `clip_field` -
    поячеечно, а не общим счетчиком, чтобы цикл по ячейкам можно было распараллелить; свертку
    делает `_equations_loop`. Молчаливое обрезание скрыло бы причину.

    Описание остальных аргументов - в докстринге пакета `paraphin.equations`.
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
def saturation_well(well, m, new_S, dt) -> None:
    """Учет отбора/закачки воды скважиной в уравнении водонасыщенности.

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    i, j = well.i, well.j
    new_S[i, j] -= dt * well.q[1] / m[i, j] / volume
