import numpy as np
import taichi as ti

from paraphin.constants import (data_type, Nx, Ny, hx, hy, dt, volume, area, ro_w, ro_f,
                                ro_o, ro_p, K_o, K_f, K_w, K_p, Pw, Po, rw, Twater)
from paraphin.utils import up_kw, up_ko, mid
from paraphin.utils.phase_f import pf_w, pf_o

well_mult = 2.0 * np.pi / np.log(rw / (0.14 * np.sqrt(hx*hx + hy*hy)))


def calc_temperature(T, m, S, C_o, C_w, C_f, C_p, Wp, Wps, p, k, mu_o, mu_w) -> ti.field(dtype=data_type, shape=(Nx, Ny)):
    """Вычисление температуры по явной схеме.

    Parameters
    ----------
    T: taichi.field(Nx, Ny)
        Температура, [С]
    m: taichi.field(Nx, Ny)
        Пористость, [-]
    S: taichi.field(Nx, Ny)
        Водоносыщенность, [-]
    C_o: taichi.field(Nx, Ny)
        Теплоемкость нефти, [Дж/C]
    C_w: taichi.field(Nx, Ny)
        Теплоемкость воды, [Дж/C]
    C_f: taichi.field(Nx, Ny)
        Теплоемкость пласта, [Дж/C]
    C_p: taichi.field(Nx, Ny)
        Теплоемкость парафина, [Дж/C]
    Wp: taichi.field(Nx, Ny)
        Концентрация растворенного парафина, [-]
    Wps: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина, [-]
    p: taichi.field(Nx, Ny)
        Давление, [Па]
    k: taichi.field(Nx, Ny)
        Проницаемость, [м^2]
    mu_o: taichi.field(Nx, Ny)
        Вязкость нефти, [Па*с]
    mu_w: taichi.field(Nx, Ny)
        Вязкость воды, [Па*с]

    Returns
    -------
    T: taichi.field(Nx, Ny)
        Температура на новом временном слое, [С]
    """

    @ti.kernel
    def calc_temperature_loop():
        mult00 =  dt / (m[0, 0] * S[0, 0] * ro_w * C_w[0, 0] + m[0, 0] * (1.0 - S[0, 0]) * ro_o * C_o[0, 0] +
                                   (m[0, 0] * (1.0 - S[0, 0]) * Wps[0, 0] + Wp[0, 0]) * ro_p * C_p[0, 0] + (1.0 - m[0, 0] - Wp[0, 0]) * ro_f * C_f[0, 0])
        multNN =  dt / (m[Nx-1, Ny-1] * S[Nx-1, Ny-1] * ro_w * C_w[Nx-1, Ny-1] + m[Nx-1, Ny-1] * (1.0 - S[Nx-1, Ny-1]) * ro_o * C_o[Nx-1, Ny-1] +
                                   (m[Nx-1, Ny-1] * (1.0 - S[Nx-1, Ny-1]) * Wps[Nx-1, Ny-1] + Wp[Nx-1, Ny-1]) * ro_p * C_p[Nx-1, Ny-1] + (1.0 - m[Nx-1, Ny-1] - Wp[Nx-1, Ny-1]) * ro_f * C_f[Nx-1, Ny-1])
        # учет скважин
        qw = (p[0, 0] - Pw) * well_mult * k[0, 0] / mu_w[0, 0]
        T[0, 0] += qw * Twater * C_w[0, 0] * ro_w * mult00

        qo = (p[Nx - 1, Ny - 1] - Po) * well_mult * k[Nx - 1, Ny - 1] / mu_w[Nx - 1, Ny - 1]
        T[Nx - 1, Ny - 1] -= qo * (C_o[Nx - 1, Ny - 1] * ro_o * pf_o(S[Nx - 1, Ny - 1]) +
                                   C_w[Nx - 1, Ny - 1] * ro_w * pf_w(S[Nx - 1, Ny - 1])) * multNN
        for i in ti.ndrange(Nx):
            for j in ti.ndrange(Ny):
                multiplier = dt / volume / (m[i, j] * S[i, j] * ro_w * C_w[i, j] + m[i, j] * (1.0 - S[i, j]) * ro_o * C_o[i, j] +
                                   (m[i, j] * (1.0 - S[i, j]) * Wps[i, j] + Wp[i, j]) * ro_p * C_p[i, j] + (1.0 - m[i, j] - Wp[i, j]) * ro_f * C_f[i, j])
                t1, t2, t3 = 0.0, 0.0, 0.0

                # цикл по граням
                arr = [[i + 1, j, hx], [i - 1, j, hx], [i, j + 1, hy], [i, j - 1, hy]]
                for idx in ti.static(ti.ndrange(4)):
                    i1, j1, hij = arr[idx]
                    if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                        temp_val = mid(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                       k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * (p[i, j] - p[i1, j1]) / hij * area

                        t1 += (T[i, j] - T[i1, j1]) / hij
                        t2 += up_kw(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
                                    k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * temp_val

                        t3 += up_ko(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
                                    k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * temp_val

                t1 *= area * (m[i, j] * (S[i, j] * K_w + (1.0 - S[i, j]) * K_o) +
                             (m[i, j] * (1.0 - S[i, j]) * Wps[i, j] + Wp[i, j]) * K_p + (1.0 - m[i, j] - Wp[i, j]) * K_f)

                t2 *= T[i, j] * ro_w * C_w[i, j]

                t3 *= T[i, j] * (ro_o * C_o[i, j] * (1.0 - Wps[i, j]) + ro_p * Wps[i, j] * C_p[i, j])

                T[i, j] += multiplier * (t1 + t2 + t3)

    calc_temperature_loop()

    return T
