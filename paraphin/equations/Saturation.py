import taichi as ti

from paraphin.constants import Nx, Ny, dt, volume, DEBUGGING, S_max, S_min
from paraphin.utils import show_plot


def calc_saturation(S, m, m_0, inj, prod, up_kw_val, new_S) -> None:
    """Вычисление водонасыщенности по явной схеме.

    Parameters
    ----------
    S: taichi.field(Nx, Ny)
        Водонасыщенность, [-]
    m: taichi.field(Nx, Ny)
        Пористость, [-]
    m_0: taichi.field(Nx, Ny)
        Пористость на прошлом временном слое, [-]
    inj: taichi.field(3)
        Дебит нагнетательной скважины [oil, water, total], [м^3/c]
    prod: taichi.field(3)
        Дебит добывающей скважины [oil, water, total], [м^3/c]
    up_kw_val: taichi.field(Nx, Ny)
        Перетоки воды в ячейках, [Па*м]
    new_S: taichi.field(Nx, Ny)
        Водонасыщенность на новом временном слое, [-]
    """

    @ti.kernel
    def calc_saturation_loop():
        for i, j in ti.ndrange(Nx, Ny):
            new_S[i, j] = S[i, j] + (-S[i, j] * (m[i, j] - m_0[i, j]) + dt * up_kw_val[i, j] / volume) / m[i, j]

        # учет скважины
        new_S[0, 0] = S_max  # += dt * inj[1] / m[0, 0]
        new_S[Nx - 1, Ny - 1] = S_min  # += dt * prod[1] / m[Nx - 1, Ny - 1]

    calc_saturation_loop()

    # TODO величины разных порядков
    print(dt * inj[1] / m[0, 0])
    print(dt * up_kw_val[0, 0] / volume / m[0, 0])
    show_plot(dt * up_kw_val.to_numpy(), 'Flows')

    if DEBUGGING:
        show_plot(new_S.to_numpy(), 'Saturation')
