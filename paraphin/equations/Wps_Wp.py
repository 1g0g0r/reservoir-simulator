from taichi import field, ndrange, kernel, static, exp

from paraphin.utils import up_ko, mid
from paraphin.constants import default_type, Nx, Ny, hx, hy, dt, ro_p, ro_o, volume, area, Tm


def calc_wps_wp(qp, m, m_0, S, S_0, Wp, Wp_0, Wps, p, k, mu_o, mu_w, T, T_0, Cp) -> (field(dtype=default_type, shape=(Nx, Ny)),
                                                                                     field(dtype=default_type, shape=(Nx, Ny))):
    """
    Вычисление концентрации взвешенных частиц (Wps) и растворенного парафина (Wp) парафина по явной схеме
    """
    # delta_Hp = delta(Hp) = Cp * delta (T) - молярные фракции парафина, растворенные в нефти
    R = 8.31446261815324  # газовая постоянная [J⋅K^−1⋅mol^−1]
    temp = 1.0 / (1.8 * Tm + 32.0)

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

                delta_Hp = Cp[i, j] * (T[i, j] - T_0[i, j]) * 1.8   # перевод в фаренгейты
                Wp[i, j] = Wps[i, j] * exp(delta_Hp / R * (temp - 1.0 / (1.8 * T[i, j] + 32.0)))

    calc_wp_wps_loop()

    return Wps, Wp
