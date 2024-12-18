import taichi as ti

from paraphin.constants import data_type, Nx, Ny, hx, hy, dt, volume, area, DEBUGGING, S_max
from paraphin.utils import up_kw, mid, show_plot


def calc_saturation(S, p, k, m, m_0, mu_o, mu_w, inj, prod) -> ti.field(dtype=data_type, shape=(Nx, Ny)):
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
    inj: taichi.field(3)
        Дебит нагнетательной скважины [oil, water, total], [м^3/c]
    prod: taichi.field(3)
        Дебит добывающей скважины [oil, water, total], [м^3/c]

    Returns
    -------
    S: taichi.field(Nx, Ny)
        Водоносыщенность на новом временном слое, [-]
    """

    new_S = ti.field(dtype=data_type, shape=(Nx, Ny))  # водонасыщенность на новом временном слое

    @ti.kernel
    def calc_saturation_loop():
        for i in ti.ndrange(Nx):
            for j in ti.ndrange(Ny):
                temp_val = 0.0
                arr = [[i+1, j, hx], [i-1, j, hx], [i, j+1, hy], [i, j-1, hy]]

                for idx in ti.static(ti.ndrange(4)):
                    i1, j1, hij = arr[idx]
                    if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                        temp_val += up_kw(k[i, j],   S[i, j],   p[i, j],   mu_o[i, j],   mu_w[i, j],
                                          k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * \
                                    mid(k[i, j],   S[i, j],   mu_o[i, j],   mu_w[i, j],
                                        k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1])*(p[i, j] - p[i1, j1])/hij
                if (i == Nx - 1 and j == Ny - 1) or (i == 0 and j == 0):
                    temp_val = 0.0
                new_S[i, j] = S[i, j] - S[i, j] * (m[i, j] - m_0[i, j]) / m[i, j] - dt * temp_val * area / m[i, j] / volume

        # учет скважины
        # new_S[0, 0] += dt * inj[1] / m[0, 0]
        new_S[0, 0] = S_max
        new_S[Nx - 1, Ny - 1] -= dt * prod[1] / m[Nx - 1, Ny - 1]

    calc_saturation_loop()
    if DEBUGGING:
        show_plot(new_S.to_numpy(), 'Saturation')
    return new_S
