import taichi as ti

from paraphin.constants import Nx, Ny, dt, volume, ro_w, ro_f, ro_o, ro_p, K_o, K_f, K_w, K_p, Twater, DEBUGGING
from paraphin.utils import show_plot


def calc_temperature(T, m, S, C_o, C_w, C_f, C_p, Wp, Wps, inj, prod,
                     up_kw_val, up_ko_val, dt_val, new_T) -> None:
    """Вычисление температуры по явной схеме.

    Parameters
    ----------
    T: taichi.field(Nx, Ny)
        Температура, [С]
    m: taichi.field(Nx, Ny)
        Пористость, [-]
    S: taichi.field(Nx, Ny)
        Водонасыщенность, [-]
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
    inj: taichi.field(3)
         Дебит нагнетательной скважины [oil, water, total], [м^3/c]
    prod: taichi.field(3)
         Дебит добывающей скважины [oil, water, total], [м^3/c]
    up_kw_val: taichi.field(Nx, Ny)
        Перетоки воды в ячейках, [Па*м]
    up_ko_val: taichi.field(Nx, Ny)
        Перетоки нефти в ячейках, [Па*м]
    dt_val: taichi.field(Nx, Ny)
        Величина (T_i - T_j) * area / h_ij, [C*м]
    new_T: taichi.field(Nx, Ny)
        Температура на новом временном слое, [С]
    """

    @ti.kernel
    def calc_temperature_loop():
        for i, j in ti.ndrange(Nx, Ny):
            multiplier = dt / volume / (m[i, j] * S[i, j] * ro_w * C_w[i, j] + m[i, j] * (1.0 - S[i, j]) * ro_o * C_o[i, j] +
                               (m[i, j] * (1.0 - S[i, j]) * Wps[i, j] + Wp[i, j]) * ro_p * C_p[i, j] + (1.0 - m[i, j] - Wp[i, j]) * ro_f * C_f[i, j])

            t1 = dt_val[i, j] * (m[i, j] * (S[i, j] * K_w + (1.0 - S[i, j]) * K_o) +
                    (m[i, j] * (1.0 - S[i, j]) * Wps[i, j] + Wp[i, j]) * K_p + (1.0 - m[i, j] - Wp[i, j]) * K_f)

            t2 = T[i, j] * ro_w * C_w[i, j] * up_kw_val[i, j]

            t3 = T[i, j] * (ro_o * C_o[i, j] * (1.0 - Wps[i, j] - Wp[i, j]) + ro_p * Wps[i, j] * C_p[i, j]) * up_ko_val[i, j]

            new_T[i, j] = T[i, j] + multiplier * (t1 + t2 + t3)

        # Учет скважин
        mult00 =  dt / (m[0, 0] * S[0, 0] * ro_w * C_w[0, 0] + m[0, 0] * (1.0 - S[0, 0]) * ro_o * C_o[0, 0] +
                                   (m[0, 0] * (1.0 - S[0, 0]) * Wps[0, 0] + Wp[0, 0]) * ro_p * C_p[0, 0] + (1.0 - m[0, 0] - Wp[0, 0]) * ro_f * C_f[0, 0])
        multNN =  dt / (m[Nx-1, Ny-1] * S[Nx-1, Ny-1] * ro_w * C_w[Nx-1, Ny-1] + m[Nx-1, Ny-1] * (1.0 - S[Nx-1, Ny-1]) * ro_o * C_o[Nx-1, Ny-1] +
                                   (m[Nx-1, Ny-1] * (1.0 - S[Nx-1, Ny-1]) * Wps[Nx-1, Ny-1] + Wp[Nx-1, Ny-1]) * ro_p * C_p[Nx-1, Ny-1] + (1.0 - m[Nx-1, Ny-1] - Wp[Nx-1, Ny-1]) * ro_f * C_f[Nx-1, Ny-1])


        # new_T[0, 0] -= inj[1] * Twater * C_w[0, 0] * ro_w * mult00
        new_T[0, 0] = Twater

        new_T[Nx - 1, Ny - 1] += (C_o[Nx - 1, Ny - 1] * ro_o * prod[0] +
                                  C_w[Nx - 1, Ny - 1] * ro_w * prod[1]) * multNN

    calc_temperature_loop()
    if DEBUGGING:
        show_plot(new_T.to_numpy(), 'Temperature')
