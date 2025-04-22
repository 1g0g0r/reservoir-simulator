import taichi as ti

from paraphin.constants import dt, volume


@ti.func
def saturation_equation(i, j, S, m, m_0, up_kw_val, new_S) -> None:
    """Вычисление водонасыщенности по явной схеме.

    Parameters
    ----------
    i, j : int
        Индексы текущей ячейки, [-]
    S: taichi.field(Nx, Ny)
        Водонасыщенность, [-]
    m: taichi.field(Nx, Ny)
        Пористость, [-]
    m_0: taichi.field(Nx, Ny)
        Пористость на прошлом временном слое, [-]
    up_kw_val: taichi.field(Nx, Ny)
        Перетоки воды в ячейках, [Па*м]
    new_S: taichi.field(Nx, Ny)
        Водонасыщенность на новом временном слое, [-]
    """
    new_S[i, j] = S[i, j] + (-S[i, j] * (m[i, j] - m_0[i, j]) + dt * up_kw_val[i, j] / volume) / m[i, j]


def saturation_well(well, m, new_S) -> None:
    """Учет скважины в уравнении водонасыщенности.

    Parameters
    ----------
    well: Well
        Объект класса скважина
    m: taichi.field(Nx, Ny)
        Пористость, [-]
    new_S: taichi.field(Nx, Ny)
        Водонасыщенность на новом временном слое, [-]
    """
    i, j = well.i, well.j
    new_S[i, j] -= dt * well.q[1] / m[i, j] / volume
