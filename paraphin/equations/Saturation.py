from numpy import log, sqrt, pi
from taichi import field, ndrange, kernel, static

from paraphin.utils import up_kw, mid
from paraphin.constants import default_type, Nx, Ny, hx, hy, dt, Pw, rw, volume, area

well_mult = 2.0 * pi / log(rw / (0.14 * sqrt(hx*hx + hy*hy)))


def calc_saturation(S, p, k, m, m_0, mu_o, mu_w) -> field(dtype=default_type, shape=(Nx, Ny)):
    """
    Вычисление водонасыщенности по явной схеме.
    """

    @kernel
    def calc_saturation_loop():
        for i in ndrange((1, Nx - 1)):
            for j in ndrange((1, Ny - 1)):
                temp_val = 0.0
                arr = [[i+1, j, hx], [i-1, j, hx], [i, j+1, hy], [i, j-1, hy]]

                for idx in static(ndrange(4)):
                    i1, j1, hij = arr[idx]
                    temp_val += up_kw(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
                                      k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * \
                                mid(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                    k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1])*area*(p[i, j] - p[i1, j1])/hij

                S[i, j] += -S[i, j] * (m[i, j] - m_0[i, j]) / m[i, j] - temp_val / m[i, j] / volume

    calc_saturation_loop()

    # учет скважины
    qw = (Pw - p[0, 0]) * well_mult * k[0, 0] / mu_w[0, 0]
    S[0, 0] += dt * qw / m[0, 0]

    return S
