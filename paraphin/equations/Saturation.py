import taichi as ti

from paraphin.constants import Nx, Ny, dt, volume, DEBUGGING, S_max
from paraphin.utils import show_plot


def calc_saturation(S, m, m_0, inj, prod, up_kw_val, mid_val, dp_val, new_S) -> None:
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
        Вычисленный параметр Kw по схеме против потока, [-]
    mid_val: taichi.field(Nx, Ny)
        Осредненное значение mid(Ko + Kw)_ij, [-]
    dp_val: taichi.field(Nx, Ny)
        Величина (p_i - p_j) * area / h_ij, [Па*м]
    new_S: taichi.field(Nx, Ny)
        Водонасыщенность на новом временном слое, [-]
    """

    @ti.kernel
    def calc_saturation_loop():
        for i, j in ti.ndrange(Nx, Ny):
            new_S[i, j] = (S[i, j] - S[i, j] * (m[i, j] - m_0[i, j]) +
                           dt / volume * up_kw_val[i, j] * mid_val[i, j] * dp_val[i, j]) / m[i, j]
        # учет скважины
        new_S[0, 0] += dt * inj[1] / m[0, 0]
        new_S[Nx - 1, Ny - 1] -= dt * prod[1] / m[Nx - 1, Ny - 1]

    calc_saturation_loop()

    if DEBUGGING:
        show_plot(new_S.to_numpy(), 'Saturation')
