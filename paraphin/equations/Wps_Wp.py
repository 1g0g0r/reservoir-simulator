from taichi import field, ndrange, kernel, static, exp

from paraphin.utils import up_ko, mid
from paraphin.constants import default_type, Nx, Ny, hx, hy, dt, ro_p, ro_o, volume, area, Tm, R

# Операции с константными величинами (вычисляются один раз только при импорте модуля)
temp = 1.0 / (1.8 * Tm + 32.0)


def calc_wps_wp(qp, m, m_0, S, S_0, Wp, Wp_0, Wps, p, k, mu_o, mu_w, T, T_0, C_p) -> (field(dtype=default_type, shape=(Nx, Ny)),
                                                                                     field(dtype=default_type, shape=(Nx, Ny))):
    """Вычисление концентрации взвешенных частиц (Wps) и растворенного парафина (Wp) парафина по явной схеме.

    Parameters
    ----------
    qp: taichi.field(Nx, Ny)
         Скорость отложения парафиновых отложений в общем объеме пористой породы
    m: taichi.field(Nx, Ny)
        Пористость, [-]
    m_0: taichi.field(Nx, Ny)
        Пористость на прошлом временном слое, [-]
    S: taichi.field(Nx, Ny)
        Водоносыщенность, [-]
    S_0: taichi.field(Nx, Ny)
        Водоносыщенность на прошлом временном слое, [-]
    Wp: taichi.field(Nx, Ny)
        Концентрация растворенного парафина, [-]
    Wp_0: taichi.field(Nx, Ny)
        Концентрация растворенного парафина на прошлом временном слое, [-]
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
    T: taichi.field(Nx, Ny)
        Температура, [С]
    T_0: taichi.field(Nx, Ny)
        Температура на прошлом временном слое, [С]
    C_p: taichi.field(Nx, Ny)
        Теплоемкость парафина, [Дж/C]
    Returns
    -------
    Wp: taichi.field(Nx, Ny)
        Концентрация растворенного парафина на новом временном слое, [-]
    Wps: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина на новом временном слое, [-]
    """
    @kernel
    def calc_wp_wps_loop():
        for i in ndrange((1, Nx - 1)):
            for j in ndrange((1, Ny - 1)):
                temp_val = 0.0
                arr = [[i + 1, j, hx], [i - 1, j, hx], [i, j + 1, hy], [i, j - 1, hy]]
                for idx in static(ndrange(4)):
                    i1, j1, hij = arr[idx]
                    temp_val += up_ko(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
                                      k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * \
                                mid(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                    k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1])*area*(p[i, j] - p[i1, j1])/hij

                Wps[i, j] += dt / (m[i, j] * (1.0 - S[i, j]) * ro_p * volume) * (-(Wps[i, j] * ro_p + ro_o * Wp[i, j]) *
                            ((m[i, j] * (1.0 - S[i, j]) - m_0[i, j] * (1.0 - S_0[i, j])) / dt - temp_val) - ro_o * Wp[i, j] *
                            (1.0 - S[i, j]) * (Wp[i, j] - Wp_0[i, j]) / dt - ro_p * qp[i, j] * volume)

                # delta_Hp = Cp * delta (T) - молярные доли парафина, растворенные в нефти
                delta_Hp = C_p[i, j] * (T[i, j] - T_0[i, j]) * 1.8   # * 1.8 - перевод в фаренгейты
                Wp[i, j] = Wps[i, j] * exp(delta_Hp / R * (temp - 1.0 / (1.8 * T[i, j] + 32.0)))

    calc_wp_wps_loop()

    return Wps, Wp
