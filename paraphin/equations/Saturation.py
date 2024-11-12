import numpy as np
import taichi as ti

from paraphin.constants import default_type, Nx, Ny, hx, hy, dt, Pw, Po, rw, volume, area
from paraphin.utils import up_kw, mid, show_plot
from paraphin.utils.phase_f import pf_w

well_mult = 2.0 * np.pi / np.log(rw / (0.14 * np.sqrt(hx*hx + hy*hy)))


def calc_saturation(S, p, k, m, m_0, mu_o, mu_w) -> ti.field(dtype=default_type, shape=(Nx, Ny)):
    """Вычисление водонасыщенности по явной схеме.

    Parameters
    ----------
    S: taichi.field(Nx, Ny)
        Водоносыщенность, [-]
    p: taichi.field(Nx, Ny)
        Давление, [Па]
    k: taichi.field(Nx, Ny)
        Проницаемость, [м^2]
    m: taichi.field(Nx, Ny)
        Пористость, [-]
    m_0: taichi.field(Nx, Ny)
        Пористость на прошлом временном слое, [-]
    mu_o: taichi.field(Nx, Ny)
        Вязкость нефти, [Па*с]
    mu_w: taichi.field(Nx, Ny)
        Вязкость воды, [Па*с]

    Returns
    -------
    S: taichi.field(Nx, Ny)
        Водоносыщенность на новом временном слое, [-]
    """

    @ti.kernel
    def calc_saturation_loop():
        # учет скважины
        qw = (Pw - p[0, 0]) * well_mult * k[0, 0] / mu_w[0, 0]
        S[0, 0] += dt * qw / m[0, 0]

        qo = (Po - p[Nx - 1, Ny - 1]) * well_mult * k[Nx - 1, Ny - 1] / mu_w[Nx - 1, Ny - 1]
        S[Nx - 1, Ny - 1] -= dt * qo / m[Nx - 1, Ny - 1] * pf_w(S[Nx - 1, Ny - 1])

        for i in ti.ndrange(Nx):
            for j in ti.ndrange(Ny):
                temp_val = 0.0
                arr = [[i+1, j, hx], [i-1, j, hx], [i, j+1, hy], [i, j-1, hy]]

                for idx in ti.static(ti.ndrange(4)):
                    i1, j1, hij = arr[idx]
                    if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                        temp_val += up_kw(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
                                          k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * \
                                    mid(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                        k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1])*(p[i, j] - p[i1, j1])/hij

                S[i, j] += -S[i, j] * (m[i, j] - m_0[i, j]) / m[i, j] - dt * temp_val * area / m[i, j] / volume

    calc_saturation_loop()

    # show_plot(S.to_numpy(), 'plotly')
    return S
