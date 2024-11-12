import os
from logging import basicConfig, INFO, getLogger
from pickle import dump, load

import numpy as np
import taichi as ti

from paraphin.constants import (default_type, Nx, Ny, Nr, results_path, logs_path, init_T, r, fi_0, init_k,
                                init_S, init_m, init_Wp, init_Wo, init_p, init_qp, init_h_sloy)
from paraphin.equations import calc_qp, calc_pressure, calc_saturation, calc_temperature, calc_wps_wp, calc_velocitys_h
from paraphin.utils.fluids_correlations import calc_mu_o, calc_mu_w, calc_c_f, calc_c_o, calc_c_w, calc_c_p

basicConfig(
    filename=logs_path,
    filemode='w',  # 'w' для перезаписи, 'a' для добавления
    level=INFO,
    format='%(asctime)s - %(message)s',  # - %(name)s - %(levelname)s
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = getLogger(__name__)


@ti.data_oriented
class Solver:
    def __init__(self, d_type = default_type):
        self.d_type = d_type

        # свойства флюидов
        self.mu_o = ti.field(dtype=d_type, shape=(Nx, Ny))  # вязкость нефти
        self.mu_w = ti.field(dtype=d_type, shape=(Nx, Ny))  # вязкость воды
        self.C_w  = ti.field(dtype=d_type, shape=(Nx, Ny))  # теплоемкость воды
        self.C_o  = ti.field(dtype=d_type, shape=(Nx, Ny))  # теплоемкость нефти
        self.C_f  = ti.field(dtype=d_type, shape=(Nx, Ny))  # теплоемкость пласта
        self.C_p  = ti.field(dtype=d_type, shape=(Nx, Ny))  # теплоемкость парафина

        # поля данных
        self.p    = ti.field(dtype=d_type, shape=(Nx, Ny))  # давление
        self.S    = ti.field(dtype=d_type, shape=(Nx, Ny))  # Водонасыщенность
        self.S_0  = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.Wo   = ti.field(dtype=d_type, shape=(Nx, Ny))  # Массовая доля маслянного компонента в нефти
        self.Wo_0 = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.Wp   = ti.field(dtype=d_type, shape=(Nx, Ny))  # Массовая доля растворенного парафина в нефти
        self.Wp_0 = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.Wps  = ti.field(dtype=d_type, shape=(Nx, Ny))  # Массовая доля взвешенного парафина в нефти
        self.k    = ti.field(dtype=d_type, shape=(Nx, Ny))  # проницаемость [m^2]
        self.m    = ti.field(dtype=d_type, shape=(Nx, Ny))  # пористость
        self.m_0  = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.T    = ti.field(dtype=d_type, shape=(Nx, Ny))  # температура [C]
        self.T_0  = ti.field(dtype=d_type, shape=(Nx, Ny))

        # динамика образования парафина (кольматация\суффозия)
        self.integr_r2_fi0 = ti.field(dtype=d_type, shape=())
        self.integr_r4_fi0 = ti.field(dtype=d_type, shape=())
        self.r      = ti.field(dtype=d_type, shape=Nr)
        self.qp     = ti.field(dtype=d_type, shape=(Nx, Ny))  # скорость отложения парафина в общем объеме
        self.fi     = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.h_sloy = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.Ur     = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.Ub     = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))


    def initialize(self):
        def calc_integrals(rr: ti.types.ndarray(), fi_o: ti.types.ndarray()):
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

        @ti.kernel
        def initialize_params_loop(fi_o: ti.types.ndarray()):
            for i in ti.ndrange(Nx):
                for j in ti.ndrange(Ny):
                    # параметры пласта
                    self.p[i, j] = init_p
                    self.S[i, j] = init_S
                    self.S_0[i, j] = init_S
                    self.Wo[i, j] = init_Wo
                    self.Wo_0[i, j] = init_Wo
                    self.Wp[i, j] = init_Wp
                    self.Wp_0[i, j] = init_Wp
                    self.Wps[i, j] = 1.0 - init_Wo - init_Wp
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

                    for ij in ti.ndrange(fi_o.shape[0]):
                        self.fi[i, j, ij] = fi_o[ij]
                        self.h_sloy[i, j, ij] = init_h_sloy
                        self.Ur[i, j, ij] = 0.0
                        self.Ub[i, j, ij] = 0.0

        self.r.from_numpy(r)
        calc_integrals(rr=r, fi_o=fi_0)
        initialize_params_loop(fi_o=fi_0)


    @ti.kernel
    def _update_mu_and_c_temp(self):
        for i, j in ti.ndrange(Nx, Ny):
            self.mu_o[i, j] = calc_mu_o(self.T[i, j])
            self.mu_w[i, j] = calc_mu_w(self.T[i, j])
            self.C_w[i, j] = calc_c_w(self.T[i, j])
            self.C_o[i, j] = calc_c_o(self.T[i, j])
            self.C_f[i, j] = calc_c_f(self.T[i, j])
            self.C_p[i, j] = calc_c_p(self.T[i, j])


    def _update_p(self) -> None:
        """Обновление давления."""
        p_new, mat_singularity =  calc_pressure(self.p, self.Wo, self.Wo_0, self.m, self.m_0,
                                                self.k, self.S, self.mu_o, self.mu_w)

        if mat_singularity: logger.error('Матрица сингулярна. Решение получено итерационным методом.')
        logger.info(f"Обновлено давление: {np.mean(p_new.to_numpy())}")
        self.p = p_new


    def _update_s(self) -> ti.field(dtype=default_type, shape=(Nx, Ny)):
        """Обновление насыщенности."""
        new_S =  calc_saturation(self.S, self.p, self.k, self.m, self.m_0, self.mu_o, self.mu_w)

        logger.info(f"Обновлена насыщенность: {np.mean(new_S.to_numpy())}")
        return new_S


    def _update_wps_wp(self) -> (ti.field(dtype=default_type, shape=(Nx, Ny)),
                                 ti.field(dtype=default_type, shape=(Nx, Ny))):
        """Обновлнние концентрации взвешенного и растворенного парафина."""
        new_wps, new_wp = calc_wps_wp(self.qp, self.m, self.m_0, self.S, self.S_0, self.Wp, self.Wp_0, self.Wps,
                                      self.p, self.k, self.mu_o, self.mu_w, self.T, self.T_0, self.C_p)

        logger.info(f"Обновлены доли взвешенного парафина: {np.mean(new_wps.to_numpy())}")
        logger.info(f"Обновлены доли растворенного парафина: {np.mean(new_wps.to_numpy())}")
        return new_wps, new_wp


    def _update_t(self) -> ti.field(dtype=default_type, shape=(Nx, Ny)):
        """Обновление температуры."""
        new_t = calc_temperature(self.T, self.m, self.S, self.C_o, self.C_w, self.C_f, self.C_p,
                                 self.Wp, self.Wps, self.p, self.k, self.mu_o, self.mu_w)

        logger.info(f"Обновлена температура: {np.mean(new_t.to_numpy())}")
        return new_t


    def _update_qp_m_k(self) -> (ti.field(dtype=default_type, shape=(Nx, Ny)),
                                 ti.field(dtype=default_type, shape=(Nx, Ny)),
                                 ti.field(dtype=default_type, shape=(Nx, Ny))):
        """Обновление объема выделяемого парафина, пористости и проницаемости."""
        new_qp, m_mult, k_mult = calc_qp(self.Wps, self.m, self.qp, self.fi, self.Ur, self.Ub,
                                         self.r, self.integr_r2_fi0[None], self.integr_r4_fi0[None])

        logger.info(f"Обновлена доля выпадающего парафина {np.mean(new_qp.to_numpy())}")
        logger.info(f"Обновлен множитель пористости {np.mean(m_mult.to_numpy())}")
        logger.info(f"Обновлен множитель проницаемости {np.mean(k_mult.to_numpy())}")
        return new_qp, m_mult, k_mult


    def _update_h_ur_ub(self) -> (ti.field(dtype=default_type, shape=(Nx, Ny)),
                                  ti.field(dtype=default_type, shape=(Nx, Ny)),
                                  ti.field(dtype=default_type, shape=(Nx, Ny))):
        """Обновление толщины осадочного слоя, скорости изменения радиуса капилляра и скорости блокирования капилляров."""
        new_h, new_ur, new_ub = calc_velocitys_h(self.p, self.Wps, self.mu_o, self.fi, self.r,
                                                 self.h_sloy, self.Ur, self.Ub)

        logger.info(f"Обновлена толщина осадочного слоя: {np.mean(new_h.to_numpy())}")
        logger.info(f"Обновлена скорость изменения радиуса капилляра: {np.mean(new_ur.to_numpy())}")
        logger.info(f"Обновлена скорость блокировки капилляров: {np.mean(new_ub.to_numpy())}")
        return new_h, new_ur, new_ub


    def upd_time_step(self, t) -> None:
        """Метод IMPES: явный по насыщенности неявный по давлению."""
        # Ввиду параллельного выполнения циклов taichi распараллеливание задач снижает производительность
        logger.info('')
        logger.info(f"ВРЕМЕННОЙ СЛОЙ t = {t / 86400.0} день")

        # --- решение гидродинамики ---
        self._update_p()                         # Обновление давления
        new_s = self._update_s()                 # Обновление насыщенности
        new_wps, new_wp = self._update_wps_wp()  # Обновление концентраций парафина
        new_t = self._update_t()                 # Обновление температуры

        # --- решение задачи кольматации\суффозии ---
        new_h, new_Ur, new_Ub = self._update_h_ur_ub()  # Обновление толщины осадочного слоя, скорости изменения радиуса капилляра и скорости блокирования капилляров
        new_qp, m_mult, k_mult = self._update_qp_m_k()  # Обновление ФУНКЦИИ ПОР ПО РАЗМЕРАМ, объема выделяемого парафина, пористости, проницаемости

        # self._update_mu_and_c_temp()  # Обновление свойств веществ ввиду изменения температуры
        self._swap_time_steps(new_s, new_wps, new_wp, new_t, new_qp, m_mult, k_mult, new_h, new_Ur, new_Ub)


    def _swap_time_steps(self, new_s, new_wps, new_wp, new_t, new_qp, m_mult, k_mult, new_h, new_Ur, new_Ub):
        """Обновление полей данных на новом временном слое."""
        self.S_0, self.S = self.S, new_s
        self.Wo_0 = self.Wo
        self.Wo.from_numpy(1.0 - self.Wp.to_numpy() - self.Wps.to_numpy())
        self.Wp_0, self.Wp = self.Wp, new_wp
        self.Wps = new_wps
        self.k.from_numpy(self.k.to_numpy() * k_mult.to_numpy())
        self.m_0 = self.m
        self.m.from_numpy(self.m.to_numpy() * m_mult.to_numpy())
        self.T_0 = self.T
        self.T = new_t
        self.qp = new_qp
        self.h_sloy = new_h
        self.Ur = new_Ur
        self.Ub = new_Ub

        logger.info('Поля данных обновлены на текущем временном слое.')


    def save_results(self, t) -> None:
        """Сохранение полей данных в файл формата pkl."""
        if os.path.exists(results_path) and np.isclose(t, 0.0):
            os.remove(results_path)
            logger.info('Файл результатов очищен.')

        # Пробуем открыть существующий файл
        try:
            with open(results_path, 'rb') as f:
                data = load(f)
            # Добавляем новые данные
            data['Time']        = np.concatenate((data['Time'], [t]), axis=0)
            data['Pressure']    = np.concatenate((data['Pressure'], [self.p.to_numpy()]), axis=0)
            data['Saturation']  = np.concatenate((data['Saturation'], [self.S.to_numpy()]), axis=0)
            data['Temperature'] = np.concatenate((data['Temperature'], [self.T.to_numpy()]), axis=0)

        # Если файл не существует или пустой
        except (FileNotFoundError, EOFError):
            data = {
                'Time':        np.array([t]),
                'Pressure':    np.array([self.p.to_numpy()]),
                'Saturation':  np.array([self.S.to_numpy()]),
                'Temperature': np.array([self.T.to_numpy()])
            }

        # Записываем обновленные данные
        with open(results_path, 'wb') as f:
            dump(data, f)
            logger.info("Данные записаны в файл.")
