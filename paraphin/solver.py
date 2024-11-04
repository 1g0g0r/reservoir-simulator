from pickle import dump

from numpy import isclose, array, gradient
from numpy.linalg import norm
from taichi import field, ndrange, data_oriented, kernel, types

from paraphin.constants import (default_type, Nx, Ny, Nr, Time_end, output_file_name, init_T, r, fi_0, init_k, eta,
                                init_S, init_m, init_Wp, init_Wps, init_Wo, init_p, init_qp, init_h_sloy)
from paraphin.equations import calc_qp, calc_pressure, calc_saturation, calc_temperature, calc_wps_wp
from paraphin.utils.fluids_correlations import calc_mu_o, calc_mu_w, calc_c_f, calc_c_o, calc_c_w, calc_c_p


@data_oriented
class Solver:
    def __init__(self, d_type = default_type):
        self.d_type = d_type

        # свойства флюидов
        self.mu_o = field(dtype=d_type, shape=(Nx, Ny))  # вязкость нефти
        self.mu_w = field(dtype=d_type, shape=(Nx, Ny))  # вязкость воды
        self.C_w = field(dtype=d_type, shape=(Nx, Ny))  # теплоемкость воды
        self.C_o = field(dtype=d_type, shape=(Nx, Ny))  # теплоемкость нефти
        self.C_f = field(dtype=d_type, shape=(Nx, Ny))  # теплоемкость пласта
        self.C_p = field(dtype=d_type, shape=(Nx, Ny))  # теплоемкость парафина

        # поля данных
        self.p = field(dtype=d_type, shape=(Nx, Ny))  # давление
        self.S = field(dtype=d_type, shape=(Nx, Ny))  # Водонасыщенность
        self.S_0 = field(dtype=d_type, shape=(Nx, Ny))
        self.Wo = field(dtype=d_type, shape=(Nx, Ny))  # Массовая доля маслянного компонента в нефти
        self.Wo_0 = field(dtype=d_type, shape=(Nx, Ny))
        self.Wp = field(dtype=d_type, shape=(Nx, Ny))  # Массовая доля растворенного парафина в нефти
        self.Wp_0 = field(dtype=d_type, shape=(Nx, Ny))
        self.Wps = field(dtype=d_type, shape=(Nx, Ny))  # Массовая доля взвешенного парафина в нефти
        self.k = field(dtype=d_type, shape=(Nx, Ny))  # проницаемость [m^2]
        self.m = field(dtype=d_type, shape=(Nx, Ny))  # пористость
        self.m_0 = field(dtype=d_type, shape=(Nx, Ny))
        self.T = field(dtype=d_type, shape=(Nx, Ny))  # температура [C]
        self.T_0 = field(dtype=d_type, shape=(Nx, Ny))

        # динамика образования парафина (кольматация\суффозия)
        self.integr_r2_fi0 = field(dtype=d_type, shape=())
        self.integr_r4_fi0 = field(dtype=d_type, shape=())
        self.r = field(dtype=d_type, shape=Nr)
        self.fi = field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.h_sloy = field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.qp = field(dtype=d_type, shape=(Nx, Ny))  # скорость отложения парафина в общем объеме

        # Массив результатов
        self.time =  array([])
        self.pres_arr = array([])
        self.sat_arr = array([])
        self.temp_arr = array([])


    def initialize(self):
        def calc_integrals(rr: types.ndarray(), fi_o: types.ndarray()):
            """Вычисление интегралов от функций r^4*fi_o(r) и r^2*fi_o(r)"""
            self.integr_r2_fi0[None] = 0.0
            self.integr_r4_fi0[None] = 0.0
            r3 = r ** 3
            r4 = r3 * r
            r5 = r4 * r
            r6 = r5 * r

            for i in range(1, Nr):
                dr = rr[i] - rr[i - 1]
                a = (fi_o[i - 1] * rr[i] - fi_o[i] * rr[i - 1]) / dr
                b = (fi_o[i] - fi_o[i - 1]) / dr
                self.integr_r2_fi0[None] += (r3[i] - r3[i-1]) * a / 3 + (r4[i] - r4[i-1]) * b / 4  # r^2 * fi
                self.integr_r4_fi0[None] += (r5[i] - r5[i-1]) * a / 5 + (r6[i] - r6[i-1]) * b / 6  # r^4 * fi

        @kernel
        def initialize_params_loop(fi_o: types.ndarray()):
            for i in ndrange(Nx):
                for j in ndrange(Ny):
                    # параметры пласта
                    self.p[i, j] = init_p
                    self.S[i, j] = init_S
                    self.S_0[i, j] = init_S
                    self.Wo[i, j] = init_Wo
                    self.Wo_0[i, j] = init_Wo
                    self.Wp[i, j] = init_Wp
                    self.Wp_0[i, j] = init_Wp
                    self.Wps[i, j] = init_Wps
                    self.k[i, j] = init_k
                    self.m[i, j] = init_m
                    self.m_0[i, j] = init_m
                    self.T[i, j] = init_T
                    self.T_0[i, j] = init_T
                    self.qp[i, j] = init_qp

                    # свойства флюидов
                    self.mu_o[i, j] = calc_mu_o(init_T)
                    self.mu_w[i, j] = calc_mu_w(init_T)
                    self.C_w[i, j] = calc_c_w(init_T)
                    self.C_o[i, j] = calc_c_o(init_T)
                    self.C_f[i, j] = calc_c_f(init_T)
                    self.C_p[i, j] = calc_c_p(init_T)

                    for ij in ndrange(fi_o.shape[0]):
                        self.fi[i, j, ij] = fi_o[ij]
                        self.h_sloy[i, j, ij] = init_h_sloy

        self.r.from_numpy(r)
        calc_integrals(rr=r, fi_o=fi_0)
        initialize_params_loop(fi_o=fi_0)


    @kernel
    def _update_mu_and_c_temp(self):
        for i, j in ndrange(Nx, Ny):
            self.mu_o[i, j] = calc_mu_o(self.T[i, j])
            self.mu_w[i, j] = calc_mu_w(self.T[i, j])
            self.C_w[i, j] = calc_c_w(self.T[i, j])
            self.C_o[i, j] = calc_c_o(self.T[i, j])
            self.C_f[i, j] = calc_c_f(self.T[i, j])
            self.C_p[i, j] = calc_c_p(self.T[i, j])


    def _update_p(self) -> None:
        """Обновление давления."""
        self.p = calc_pressure(self.p, self.Wo, self.Wo_0, self.m, self.m_0,
                               self.k, self.S, self.mu_o, self.mu_w)


    def _update_s(self) -> field(dtype=default_type, shape=(Nx, Ny)):
        """Обновление насыщенности."""
        return calc_saturation(self.S, self.p, self.k, self.m, self.m_0, self.mu_o, self.mu_w)


    def _update_wps_wp(self) -> (field(dtype=default_type, shape=(Nx, Ny)),
                                 field(dtype=default_type, shape=(Nx, Ny))):
        """Обновлнние концентрации взвешенного и растворенного парафина."""
        return calc_wps_wp(self.qp, self.m, self.m_0, self.S, self.S_0, self.Wp, self.Wp_0, self.Wps,
                           self.p, self.k, self.mu_o, self.mu_w, self.T, self.T_0, self.C_p)


    def _update_t(self) -> field(dtype=default_type, shape=(Nx, Ny)):
        """Обновление температуры."""
        return calc_temperature(self.T, self.m, self.S, self.C_o, self.C_w, self.C_f, self.C_p,
                                self.Wp, self.Wps, self.p, self.k, self.mu_o, self.mu_w)


    def _update_qp_m_k(self, Um_r2) -> (field(dtype=default_type, shape=(Nx, Ny)),
                                        field(dtype=default_type, shape=(Nx, Ny)),
                                        field(dtype=default_type, shape=(Nx, Ny))):
        """Обновление объема выделяемого парафина, пористости и проницаемости."""
        return calc_qp(self.Wps, self.mu_o, self.m, self.qp, self.fi, self.h_sloy,
                       self.r, self.integr_r2_fi0, self.integr_r4_fi0, Um_r2)


    def upd_time_step(self) -> None:
        """Метод IMPES: явный по насыщенности неявный по давлению."""
        # Ввиду параллельного выполнения циклов распараллеливание задач снижает производительность
        Um_r2 = norm(gradient(self.p.to_numpy()), axis=0) * 0.125 / eta / self.mu_o.to_numpy()

        self._update_p()
        new_s = self._update_s()                        # Обновление насыщенности
        new_wps, new_wp = self._update_wps_wp()         # Обновление концентрации взвешенного парафина
        new_t = self._update_t()                        # Обновление температуры
        new_qp, m_mult, k_mult = self._update_qp_m_k(Um_r2)  # Обновление объема выделяемого парафина, пористости, проницаемости
        # self._update_mu_and_c_temp()  # Обновление свойств веществ ввиду изменения температуры
        self._swap_time_steps(new_s, new_wps, new_wp, new_t, new_qp, m_mult, k_mult)


    def _swap_time_steps(self, new_s, new_wps, new_wp, new_t, new_qp, m_mult, k_mult):
        """Обновление полей данных на новом временном слое."""
        self.S_0, self.S = self.S, new_s
        self.Wo_0 = self.Wo
        self.Wo.from_numpy(1.0 - self.Wp.to_array() - self.Wps.to_array())
        self.Wp_0, self.Wp = self.Wp, new_wp
        self.Wps = new_wps
        self.k.from_numpy(self.k.to_array() * k_mult)
        self.m_0 = self.m
        self.m.from_numpy(self.m.to_array() * m_mult)
        self.T_0 = self.T
        self.T = new_t
        self.qp = new_qp


    def save_results(self, t) -> None:
        """Сохранение полей данных в файл формата pkl."""
        self.time.append(t)
        self.pres_arr.append(self.p.to_numpy())
        self.sat_arr.append(self.S.to_numpy())
        self.temp_arr.append(self.T.to_numpy())

        if isclose(t, Time_end):
            output_data = {
                'Time': self.time,
                'Pressure': self.pres_arr,
                'Saturation': self.sat_arr,
                'Temperature': self.temp_arr
            }
            with open(output_file_name, 'wb') as f:
                dump(output_data, f)
