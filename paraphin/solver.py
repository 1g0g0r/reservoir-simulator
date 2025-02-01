from logging import basicConfig, INFO, getLogger
from pickle import dump
import psutil

import numpy as np
import taichi as ti

from paraphin.constants import (data_type, Nx, Ny, Nr, results_path, logs_path, init_T, r, fi_0, init_k, init_S,
                                init_m, init_Wp, init_Wo, init_p, init_qp, init_h_sloy, init_Wps, bar_to_pa,
                                well_mult, Pw, Po, day_to_sec, eta, mu_o, mu_w, c_o, c_w, c_p, c_f)
from paraphin.equations import calc_qp, calc_pressure, calc_saturation, calc_temperature, calc_wps_wp, calc_velocitys_h
from paraphin.utils import _pf_o, _pf_w
from paraphin.utils.fluids_correlations import calc_mu_o, calc_mu_w, calc_c_f, calc_c_o, calc_c_w, calc_c_p


@ti.data_oriented
class Solver:
    def __init__(self, d_type = data_type):
        self.d_type = d_type

        # Дебиты скважин
        self.inj  = ti.field(dtype=d_type, shape=3)  # расположена в точке (0, 0)
        self.prod = ti.field(dtype=d_type, shape=3)  # расположена в точке (Lx, Ly)

        # свойства флюидов
        self.mu_o = ti.field(dtype=d_type, shape=(Nx, Ny))  # Вязкость нефти
        self.mu_w = ti.field(dtype=d_type, shape=(Nx, Ny))  # Вязкость воды
        self.C_w  = ti.field(dtype=d_type, shape=(Nx, Ny))  # Теплоемкость воды
        self.C_o  = ti.field(dtype=d_type, shape=(Nx, Ny))  # Теплоемкость нефти
        self.C_f  = ti.field(dtype=d_type, shape=(Nx, Ny))  # Теплоемкость пласта
        self.C_p  = ti.field(dtype=d_type, shape=(Nx, Ny))  # Теплоемкость парафина

        # поля данных
        self.p    = ti.field(dtype=d_type, shape=(Nx, Ny))  # Давление
        self.S    = ti.field(dtype=d_type, shape=(Nx, Ny))  # Водонасыщенность
        self.S_0  = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.Wo   = ti.field(dtype=d_type, shape=(Nx, Ny))  # Массовая доля маслянного компонента в нефти
        self.Wo_0 = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.Wp   = ti.field(dtype=d_type, shape=(Nx, Ny))  # Массовая доля растворенного парафина в нефти
        self.Wp_0 = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.Wps  = ti.field(dtype=d_type, shape=(Nx, Ny))  # Массовая доля взвешенного парафина в нефти
        self.k    = ti.field(dtype=d_type, shape=(Nx, Ny))  # Проницаемость [m^2]
        self.m    = ti.field(dtype=d_type, shape=(Nx, Ny))  # Пористость
        self.m_0  = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.T    = ti.field(dtype=d_type, shape=(Nx, Ny))  # Температура [C]
        self.T_0  = ti.field(dtype=d_type, shape=(Nx, Ny))

        # динамика образования парафина (кольматация\суффозия)
        self.integr_r2_fi0 = ti.field(dtype=d_type, shape=())
        self.integr_r4_fi0 = ti.field(dtype=d_type, shape=())
        self.Um_r2  = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.qp     = ti.field(dtype=d_type, shape=(Nx, Ny))  # Скорость отложения парафина в общем объеме
        self.fi     = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.h_sloy = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.Ur     = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.Ub     = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))

        # поля данный с нового временного слоя
        self.new_h   = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.new_Ur  = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.new_Ub  = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.new_s   = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.new_wps = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.new_wp  = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.new_t   = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.new_qp  = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.m_mult  = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.k_mult  = ti.field(dtype=d_type, shape=(Nx, Ny))

        basicConfig(
            filename=logs_path,
            filemode='w',  # 'w'-перезапись, 'a'-добавление
            level=INFO,
            format='%(asctime)s - %(message)s',  # - %(name)s - %(levelname)s
            datefmt='%H:%M:%S'  # '%Y-%m-%d %H:%M:%S'
        )

        self.logger = getLogger(__name__)


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
                    self.p[i, j]    = init_p
                    self.S[i, j]    = init_S
                    self.S_0[i, j]  = init_S
                    self.Wo[i, j]   = init_Wo
                    self.Wo_0[i, j] = init_Wo
                    self.Wp[i, j]   = init_Wp
                    self.Wp_0[i, j] = init_Wp
                    self.Wps[i, j]  = init_Wps
                    self.k[i, j]    = init_k
                    self.m[i, j]    = init_m
                    self.m_0[i, j]  = init_m
                    self.T[i, j]    = init_T
                    self.T_0[i, j]  = init_T
                    self.qp[i, j]   = init_qp
                    self.Um_r2[i, j]= 0.0

                    # свойства флюидов
                    self.mu_o[i, j] = mu_o
                    self.mu_w[i, j] = mu_w
                    self.C_w[i, j] = c_w
                    self.C_o[i, j] = c_o
                    self.C_f[i, j] = c_f
                    self.C_p[i, j] = c_p

                    for ij in ti.ndrange(fi_o.shape[0]):
                        self.fi[i, j, ij]     = fi_o[ij]
                        self.h_sloy[i, j, ij] = init_h_sloy
                        self.Ur[i, j, ij] = 0.0
                        self.Ub[i, j, ij] = 0.0

        calc_integrals(rr=r, fi_o=fi_0)
        initialize_params_loop(fi_o=fi_0)


    @ti.kernel
    def _update_mu_and_c_temp(self):
        for i, j in ti.ndrange(Nx, Ny):
            self.mu_o[i, j] = calc_mu_o(self.T[i, j])
            self.mu_w[i, j] = calc_mu_w(self.T[i, j])
            self.C_w[i, j]  = calc_c_w(self.T[i, j])
            self.C_o[i, j]  = calc_c_o(self.T[i, j])
            self.C_f[i, j]  = calc_c_f(self.T[i, j])
            self.C_p[i, j]  = calc_c_p(self.T[i, j])


    def _update_p(self) -> None:
        """Обновление давления."""
        calc_pressure(self.p, self.Wo, self.Wo_0, self.m, self.m_0, self.k, self.S, self.mu_o, self.mu_w)

        min_p = np.min(self.p.to_numpy()) / bar_to_pa
        max_p = np.max(self.p.to_numpy()) / bar_to_pa
        self.logger.info(f"Обновлено давление (bar):      min={min_p}  max={max_p}")


    def _update_q(self) -> None:
        """Обновление дебитов скважин."""
        inj_mult  = (self.p[0, 0] - Pw) * well_mult * self.k[0, 0]
        self.inj[0] = 0.0 # inj_mult / self.mu_o[0, 0] * _pf_o(1.0)
        self.inj[1] = inj_mult / self.mu_w[0, 0]  # * _pf_w(1.0)
        self.inj[2] = (self.inj[0] + self.inj[1])

        prod_mult = (self.p[Nx - 1, Ny - 1] - Po) * well_mult * self.k[Nx - 1, Ny - 1]
        self.prod[0] = prod_mult * _pf_o(self.S[Nx - 1, Ny - 1]) / self.mu_o[Nx - 1, Ny - 1]
        self.prod[1] = prod_mult * _pf_w(self.S[Nx - 1, Ny - 1]) / self.mu_w[Nx - 1, Ny - 1]
        self.prod[2] = (self.prod[0] + self.prod[1])

        self.logger.info(f"Дебит нагнетательной скважины: q_o={self.inj[0]}  q_w={self.inj[1]}")
        self.logger.info(f'dp = {self.p[0, 0] - Pw}')
        self.logger.info(f"Дебит добывающей скважины:     q_o={self.prod[0]}  q_w={self.prod[1]}")
        self.logger.info(f'dp = {self.p[Nx - 1, Ny - 1] - Po}')


    def _update_s(self) -> None:
        """Обновление насыщенности."""
        calc_saturation(self.new_s, self.S, self.p, self.k, self.m, self.m_0, self.mu_o, self.mu_w, self.inj, self.prod)
        min_s = np.min(self.new_s.to_numpy())
        max_s = np.max(self.new_s.to_numpy())

        self.logger.info(f"Обновлена насыщенность:                 min={min_s}  max={max_s}")


    def _update_wps_wp(self) -> None:
        """Обновлнние концентрации взвешенного и растворенного парафина."""
        new_wps, new_wp = calc_wps_wp(self.qp, self.m, self.m_0, self.S, self.S_0, self.Wp, self.Wp_0, self.Wps,
                                      self.p, self.k, self.mu_o, self.mu_w, self.T, self.T_0, self.C_p, self.prod)
        min_wps = np.min(new_wps.to_numpy())
        max_wps = np.max(new_wps.to_numpy())
        min_wp = np.min(new_wp.to_numpy())
        max_wp = np.max(new_wp.to_numpy())

        self.new_wps = new_wps
        self.logger.info(f"Обновлены доли взвешенного парафина:    min={min_wps}  max={max_wps}")
        self.mew_wp = new_wp
        self.logger.info(f"Обновлены доли растворенного парафина:  min={min_wp}  max={max_wp}")


    def _update_t(self) -> None:
        """Обновление температуры."""
        new_t = calc_temperature(self.T, self.m, self.S, self.C_o, self.C_w, self.C_f, self.C_p, self.Wp,
                                 self.Wps, self.p, self.k, self.mu_o, self.mu_w, self.inj, self.prod)
        min_t = np.min(new_t.to_numpy())
        max_t = np.max(new_t.to_numpy())

        self.new_t = new_t
        self.logger.info(f"Обновлена температура:                  min={min_t}  max={max_t}")


    def _update_qp_m_k(self) -> None:
        """Обновление объема выделяемого парафина, пористости и проницаемости."""
        new_qp, m_mult, k_mult = calc_qp(self.Wps, self.m, self.fi, self.Ur, self.Ub,
                                         self.integr_r2_fi0[None], self.integr_r4_fi0[None])
        min_qp = np.min(new_qp.to_numpy())
        max_qp = np.max(new_qp.to_numpy())
        min_m_mult = np.min(m_mult.to_numpy())
        max_m_mult = np.max(m_mult.to_numpy())
        min_k_mult = np.min(k_mult.to_numpy())
        max_k_mult = np.max(k_mult.to_numpy())

        self.new_qp = new_qp
        self.logger.info(f"Обновлена доля выпадающего парафина:    min={min_qp}  max={max_qp}")
        self.m_mult = m_mult
        self.logger.info(f"Обновлен множитель пористости:          min={min_m_mult}  max={max_m_mult}")
        self.k_mult = k_mult
        self.logger.info(f"Обновлен множитель проницаемости:       min={min_k_mult}  max={max_k_mult}")


    def _update_h_ur_ub(self) -> None:
        """Обновление толщины осадочного слоя, скорости изменения радиуса капилляра и скорости блокирования капилляров."""
        new_h, new_ur, new_ub = calc_velocitys_h(self.Um_r2, self.Wps, self.mu_o, self.fi,
                                                 self.h_sloy, self.Ur)
        min_mew_h  = np.min(new_h.to_numpy())
        max_mew_h  = np.max(new_h.to_numpy())
        min_new_ur = np.min(new_ur.to_numpy())
        max_new_ur = np.max(new_ur.to_numpy())
        min_new_ub = np.min(new_ub.to_numpy())
        max_new_ub = np.max(new_ub.to_numpy())

        self.new_h = self.new_h
        self.logger.info(f"Обновлена толщина осадочного слоя:              min={min_mew_h}  max={max_mew_h}")
        self.new_ur = new_ur
        self.logger.info(f"Обновлена скорость изменения радиуса капилляра: min={min_new_ur}  max={max_new_ur}")
        self.new_ub = new_ub
        self.logger.info(f"Обновлена скорость блокировки капилляров:       min={min_new_ub}  max={max_new_ub}")


    def upd_time_step(self, t: float, iter: float) -> None:
        """Метод IMPES: явный по насыщенности неявный по давлению."""
        # Ввиду параллельного выполнения циклов taichi запуск задач в разных процессах снижает производительность
        self.logger.info('')
        self.logging_resources()
        self.logger.info(f"ВРЕМЕННОЙ СЛОЙ t = {t} день  ({iter} итерация)")

        # --- решение гидродинамики ---
        self._update_p()         # Обновление давления
        self._update_q()         # Обновление дебитов скважин
        self._update_s()         # Обновление насыщенности
        self._update_t()         # Обновление температуры

        # --- решение задачи кольматации\суффозии ---
        if not np.all(np.isclose(self.Wp.to_numpy(), 0)):
            self._update_wps_wp()   # Обновление концентраций парафина
            self._update_h_ur_ub()  # Обновление толщины осадочного слоя, скорости изменения радиуса капилляра и скорости блокирования капилляров
            self._update_qp_m_k()   # Обновление ФУНКЦИИ ПОР ПО РАЗМЕРАМ, объема выделяемого парафина, пористости, проницаемости
            self.Um_r2.from_numpy(np.linalg.norm(np.gradient(self.p.to_numpy()), axis=0) / self.mu_o.to_numpy() * 0.125 / eta)

        self._swap_time_steps()

        # self._update_mu_and_c_temp()  # Обновление свойств веществ ввиду изменения температуры


    def _swap_time_steps(self):
        """Обновление полей данных на новом временном слое."""
        self.S_0 = self.S
        self.S = self.new_s
        self.T_0 = self.T
        self.T = self.new_t

        if not np.all(np.isclose(self.Wp.to_numpy(), 0)):
            self.Wp_0 = self.Wp
            self.Wp = self.new_wp
            self.Wps = self.new_wps
            self.Wo_0 = self.Wo
            self.Wo.from_numpy(1.0 - self.Wp.to_numpy() - self.Wps.to_numpy())
            self.k.from_numpy(self.k.to_numpy() * self.k_mult.to_numpy())
            self.m_0 = self.m
            self.m.from_numpy(self.m.to_numpy() * self.m_mult.to_numpy())
            self.qp = self.new_qp
            self.h_sloy = self.new_h
            self.Ur = self.new_Ur
            self.Ub = self.new_Ub

        self.logger.info('Поля данных обновлены на текущем временном слое.')

    def logging_resources(self) -> None:
        # cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent

        # self.logger.info(f'CPU Usage:    {cpu_usage}%')
        self.logger.info(f'Memory Usage: {memory_usage}%')


    def save_results(self, t) -> None:
        """Сохранение полей данных в файл формата pkl."""
        if np.isclose(t, 0.0):
            for file_path in results_path.glob(f'*.pkl'):  # Перебор всех файлов .pkl
                file_path.unlink()
            self.logger.info('Старые файлы удалены.')

        data = {
            'Time':        t,
            'Pressure':    self.p.to_numpy(),
            'Saturation':  self.S.to_numpy(),
            'Temperature': self.T.to_numpy(),
            'Wps':         self.Wps.to_numpy(),
            'Wells':       {'inj':  self.inj[2],
                            'prod': self.prod[2]}
        }

        with open(results_path / f'data_{round(t / day_to_sec, 3)}.pkl', 'wb') as f:
            dump(data, f)
            self.logger.info("Данные записаны в файл.")
