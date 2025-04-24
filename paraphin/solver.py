from logging import basicConfig, INFO, getLogger
from pickle import dump

import numpy as np
import psutil
import taichi as ti

from paraphin import r1, r3, r4, r5, r6
from paraphin.constants import (data_type, Nx, Ny, Nr, rw, results_path, logs_path, init_T, fi_0, init_k, init_S,
                                init_m, init_Wp, init_Wo, init_p, init_qp, init_h_sloy, init_Wps, bar_to_pa, eta, h,
                                day_to_sec, mu_o, mu_w, c_o, c_w, c_p, c_f, sol_time_step, Time_end, LOGGING, _re)
from paraphin.equations import (calc_qp_m_k_fi, calc_pressure, saturation_equation, saturation_well, temperature_well,
                                temperature_equation, wps_wp_equation, wps_wp_wells, calc_velocitys_h, flows_in_cells,
                                preprocess_matrix_and_wells)
from paraphin.utils import calc_mu_o, calc_mu_w, calc_c_f, calc_c_o, calc_c_w, calc_c_p
from paraphin.well import WellStruct, upd_q_and_eta


@ti.data_oriented
class Solver:
    def __init__(self, d_type = data_type):
        self.d_type = d_type

        # Скважины
        self.n_wells = 0
        self._wells_buffer = []
        self.wells = WellStruct.field(shape=1)

        # Свойства флюидов
        self.mu_o = ti.field(dtype=d_type, shape=(Nx, Ny))  # Вязкость нефти, [Па*с]
        self.mu_w = ti.field(dtype=d_type, shape=(Nx, Ny))  # Вязкость воды, [Па*с]
        self.C_w  = ti.field(dtype=d_type, shape=(Nx, Ny))  # Теплоемкость воды, [Дж*кг/C]
        self.C_o  = ti.field(dtype=d_type, shape=(Nx, Ny))  # Теплоемкость нефти, [Дж*кг/C]
        self.C_f  = ti.field(dtype=d_type, shape=(Nx, Ny))  # Теплоемкость пласта, [Дж*кг/C]
        self.C_p  = ti.field(dtype=d_type, shape=(Nx, Ny))  # Теплоемкость парафина, [Дж*кг/C]

        # Поля данных
        self.p     = ti.field(dtype=d_type, shape=(Nx, Ny))  # Давление, [Па]
        self.S     = ti.field(dtype=d_type, shape=(Nx, Ny))  # Водонасыщенность, [-]
        self.S_0   = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.Wo    = ti.field(dtype=d_type, shape=(Nx, Ny))  # Массовая доля маслянного компонента в нефти, [-]
        self.Wp    = ti.field(dtype=d_type, shape=(Nx, Ny))  # Массовая доля растворенного парафина в нефти, [-]
        self.Wp_0  = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.Wps   = ti.field(dtype=d_type, shape=(Nx, Ny))  # Массовая доля взвешенного парафина в нефти, [-]
        self.Wps_0 = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.k     = ti.field(dtype=d_type, shape=(Nx, Ny))  # Проницаемость, [м^2]
        self.m     = ti.field(dtype=d_type, shape=(Nx, Ny))  # Пористость, [-]
        self.m_0   = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.T     = ti.field(dtype=d_type, shape=(Nx, Ny))  # Температура [С]
        self.T_0   = ti.field(dtype=d_type, shape=(Nx, Ny))

        # Динамика образования парафина (кольматация\суффозия)
        self.integr_r2_fi0 = ti.field(dtype=d_type, shape=())
        self.integr_r4_fi0 = ti.field(dtype=d_type, shape=())
        self.Um_r2  = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.qp     = ti.field(dtype=d_type, shape=(Nx, Ny))  # Скорость отложения парафина в общем объеме
        self.fi     = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.h_sloy = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.Ur     = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.Ub     = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))

        # Поля данный нового временного слоя
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

        # Временные массивы
        self.dt_val    = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.up_ko_val = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.up_kw_val = ti.field(dtype=d_type, shape=(Nx, Ny))

        # Вспомогательные поля класса
        self._i_img = 0
        self._paraphin = False
        self.row_indices_np = np.ndarray
        self.col_indices_np = np.ndarray
        results_path.mkdir(parents=True, exist_ok=True)

        if LOGGING:
            basicConfig(
                filename=logs_path,
                filemode='w',
                level=INFO,
                format='%(asctime)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            self.logger = getLogger(__name__)
        else:
            self.logger = getLogger(__name__)
            self.logger.handlers = []  # Удаляем обработчики
            self.logger.propagate = False


    def initialize(self):
        def _well_processing():
            self.wells = WellStruct.field(shape=self.n_wells)
            self.row_indices_np, self.col_indices_np, self.wells = preprocess_matrix_and_wells(self.wells, self._wells_buffer,
                                                                                               self.p, self.S, self.k, self.mu_o, self.mu_w)

        @ti.kernel
        def _calc_integrals(fi_o: ti.types.ndarray()):
            """Вычисление интегралов от функций r^4*fi_o(r) и r^2*fi_o(r)"""
            self.integr_r2_fi0[None] = 0.0
            self.integr_r4_fi0[None] = 0.0
            for i in range(1, Nr):
                dr = r1[i] - r1[i - 1]
                a = (fi_o[i - 1] * r1[i] - fi_o[i] * r1[i - 1]) / dr
                b = (fi_o[i] - fi_o[i - 1]) / dr
                self.integr_r2_fi0[None] += (r3[i] - r3[i-1]) * a / 3 + (r4[i] - r4[i-1]) * b / 4  # r^2 * fi
                self.integr_r4_fi0[None] += (r5[i] - r5[i-1]) * a / 5 + (r6[i] - r6[i-1]) * b / 6  # r^4 * fi

        @ti.kernel
        def _initialize_params_loop(fi_o: ti.types.ndarray()):
            for j in ti.ndrange(Ny):
                for i in ti.ndrange(Nx):
                    # Параметры пласта
                    self.p[i, j]    = init_p
                    self.S[i, j]    = init_S
                    self.S_0[i, j]  = init_S
                    self.Wo[i, j]   = init_Wo
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

                    for ij in ti.ndrange(Nr):
                        self.fi[i, j, ij]     = fi_o[ij]
                        self.h_sloy[i, j, ij] = init_h_sloy
                        self.Ur[i, j, ij] = 0.0
                        self.Ub[i, j, ij] = 0.0

        _calc_integrals(fi_o=fi_0)
        _initialize_params_loop(fi_o=fi_0)
        _well_processing()


    def add_well(self, name: str, i: int, j: int, p: float, is_injector: bool = False, T: float = -9999, rw: float = rw):
        """Добавление скважин в расчет"""
        conductivity_mult = 2.0 * np.pi * h / np.log(_re / rw) * 0.25
        well = WellStruct(i=i, j=j, p=p, T=T, rw=rw, is_injector=int(is_injector), conductivity_mult=conductivity_mult)
        self._wells_buffer.append({'well': well, 'name': name})
        self.n_wells += 1


    def upd_time_step(self, t: float) -> None:
        """Решение задачи на текущем временном слое."""
        self._process_time_step()
        self._logging_solution(t)
        self._swap_time_steps()
        self.logger.info('Поля данных обновлены на текущем временном слое.')

        # Запись данных в файл
        if t >= self._i_img * sol_time_step or np.isclose(t, Time_end):
            self._save_results(t)
            self._i_img += 1


    def _update_p(self) -> None:
        """Обновление давления."""
        calc_pressure(self.p, self.Wo, self.m, self.m_0, self.k, self.S, self.mu_o, self.mu_w, self.wells,
                      self.row_indices_np, self.col_indices_np)
        if LOGGING:
            self.logger.info(f"Обновлено давление (bar):      min={self.p.to_numpy().min() / bar_to_pa}  max={self.p.to_numpy().max() / bar_to_pa}")


    @ti.kernel
    def _update_wells_data(self):
        """Обновление дебетов и обводненности скважин."""
        ti.loop_config(serialize=True)  # parallelize=1
        for i in ti.ndrange(self.n_wells):
            self.wells[i] = upd_q_and_eta(self.wells[i], self.p, self.S, self.k, self.mu_o, self.mu_w)


    def _process_time_step(self):
        """Метод IMPES: явный по насыщенности неявный по давлению."""
        self._paraphin = not np.all(np.isclose(self.Wp.to_numpy(), 0.0))
        self._update_p()  # Обновление давления
        self._update_wells_data()  # Обновление дебитов скважин
        self._equations_loop()
        self._wells_loop()
        if self._paraphin:
            # Средняя скорость в капилляре * r^2
            self.Um_r2.from_numpy(np.linalg.norm(np.gradient(self.p.to_numpy()), axis=0) / self.mu_o.to_numpy() * 0.125 / eta)


    @ti.kernel
    def _wells_loop(self):
        ti.loop_config(serialize=True)
        for i in ti.ndrange(self.n_wells):
            saturation_well(self.wells[i], self.m, self.new_s)
            wps_wp_wells(self.wells[i], self.m, self.S, self.Wp, self.Wps, self.new_wps)
            temperature_well(self.wells[i], self.T, self.m, self.S, self.C_o, self.C_w, self.C_f, self.C_p, self.Wp, self.Wps, self.new_t)

    @ti.kernel
    def _equations_loop(self):
        for j in ti.ndrange(Ny):
            for i in ti.ndrange(Nx):
                # ---решение гидродинамики---
                flows_in_cells(i, j, self.p, self.S, self.T, self.k, self.mu_o, self.mu_w, self.m, self.Wps,
                               self.dt_val, self.up_kw_val, self.up_ko_val)
                saturation_equation(i, j, self.S, self.m, self.m_0, self.up_kw_val, self.new_s)
                temperature_equation(i, j, self.T, self.m, self.m_0, self.S, self.S_0, self.C_o, self.C_w, self.C_f, self.C_p,
                                     self.Wp, self.Wp_0, self.Wps, self.Wps_0, self.up_kw_val, self.up_ko_val, self.dt_val, self.new_t)
                # ---решение задачи кольматации\суффозии---
                if self._paraphin:
                    # Обновление концентраций парафина
                    wps_wp_equation(i, j, self.qp, self.m, self.m_0, self.S, self.S_0, self.Wp, self.Wp_0, self.Wps,
                                    self.T, self.T_0, self.C_p, self.up_ko_val, self.new_wp, self.new_wps)
                    # Обновление толщины осадочного слоя, скорости изменения радиуса капилляра и скорости блокирования капилляров
                    calc_velocitys_h(i, j, self.Um_r2, self.Wps, self.mu_o, self.fi, self.h_sloy, self.Ur,
                                     self.new_h, self.new_Ur, self.new_Ub)
                    # Обновление функции пор по размерам, объема выделяемого парафина, пористости, проницаемости
                    calc_qp_m_k_fi(i, j, self.Wps, self.m, self.fi, self.Ur, self.Ub, self.integr_r2_fi0[None],
                                   self.integr_r4_fi0[None], self.new_qp, self.k_mult, self.m_mult)
                # ---пересчет свойств флюидов из-за изменения температуры---
                # self._update_mu_and_c_temp(i, j)


    @ti.func
    def _update_mu_and_c_temp(self, i, j) -> None:
        self.mu_o[i, j] = calc_mu_o(self.T[i, j])
        self.mu_w[i, j] = calc_mu_w(self.T[i, j])
        self.C_w[i, j]  = calc_c_w(self.T[i, j])
        self.C_o[i, j]  = calc_c_o(self.T[i, j])
        self.C_f[i, j]  = calc_c_f(self.T[i, j])
        self.C_p[i, j]  = calc_c_p(self.T[i, j])


    @ti.kernel
    def _swap_time_steps(self):
        """Обновление полей данных на новом временном слое."""
        for j in ti.ndrange(Ny):
            for i in ti.ndrange(Nx):
                self.S_0[i, j] = self.S[i, j]
                self.S[i, j] = self.new_s[i, j]
                self.T_0[i, j] = self.T[i, j]
                self.T[i, j] = self.new_t[i, j]

                if self._paraphin:
                    self.Wp_0[i, j] = self.Wp[i, j]
                    self.Wp[i, j] = self.new_wp[i, j]
                    self.Wps_0[i, j] = self.Wps[i, j]
                    self.Wps[i, j] = self.new_wps[i, j]
                    self.k[i, j] = init_k * self.k_mult[i, j]
                    self.m_0[i, j] = self.m[i, j]
                    self.m[i, j] = init_m * self.m_mult[i, j]
                    self.qp[i, j] = self.new_qp[i, j]

                    for ij in ti.ndrange(Nr):
                        self.h_sloy[i, j, ij] = self.new_h[i, j, ij]
                        self.Ur[i, j, ij] = self.new_Ur[i, j, ij]
                        self.Ub[i, j, ij] = self.new_Ub[i, j, ij]


    def _save_results(self, t) -> None:
        """Сохранение полей данных в файл формата pkl."""
        if np.isclose(t, 0.0):
            for file_path in results_path.glob(f'*.pkl'):  # Перебор всех файлов .pkl
                file_path.unlink()
            self.logger.info('Старые файлы удалены.')

        wells_data_o = {f'{self._wells_buffer[i]["name"]}_oil': self.wells[i].q[0] for i in range(self.n_wells)}
        wells_data_w = {f'{self._wells_buffer[i]["name"]}_water': self.wells[i].q[1] for i in range(self.n_wells)}
        wells_data_t = {f'{self._wells_buffer[i]["name"]}_total': self.wells[i].q[2] for i in range(self.n_wells)}
        wells_data_eta = {f'{self._wells_buffer[i]["name"]}_eta': self.wells[i].eta / day_to_sec
                          for i in range(self.n_wells) if not self.wells[i].is_injector}
        data = {
            'Time':        t,
            'Pressure':    self.p.to_numpy(),
            'Saturation':  self.S.to_numpy(),
            'Temperature': self.T.to_numpy(),
            # 'Wps':         self.Wps.to_numpy(),
            'Wells':       wells_data_o | wells_data_w | wells_data_t | wells_data_eta
        }
        with open(results_path / f'data_{round(t / day_to_sec, 3)}.pkl', 'wb') as f:
            dump(data, f)
            self.logger.info("Данные записаны в файл.")


    def _logging_resources(self) -> None:
        # cpu_usage = psutil.cpu_percent(interval=None)  # , percpu=True
        memory_info = psutil.virtual_memory()
        memory_usage = round(memory_info.used / memory_info.total * 100 , 5)  # memory_info.percent
        # self.logger.info(f'Использование CPU: {cpu_usage}%')
        self.logger.info(f'Использование памяти: {memory_usage}%')


    def _logging_solution(self, t):
        """Логирование полей задачи"""
        if not LOGGING:
            return None
        self.logger.info('')
        self._logging_resources()
        self.logger.info(f"ВРЕМЕННОЙ СЛОЙ t = {round(t / day_to_sec, 5)} день")
        self.logger.info(f"Обновлена насыщенность:  min={self.new_s.to_numpy().min()}  max={self.new_s.to_numpy().max()}")
        self.logger.info(f"Обновлена температура:   min={self.new_t.to_numpy().min()}  max={self.new_t.to_numpy().max()}")
        for i in range(self.n_wells):
            self.logger.info(f"Дебит скважины {self._wells_buffer[i]['name']}: q_o={self.wells[i].q[0] * day_to_sec}  q_w={self.wells[i].q[1] * day_to_sec}")

        if not self._paraphin:
            return None
        self.logger.info(f"Обновлены доли взвешенного парафина:   min={self.new_wps.to_numpy().min()}  max={self.new_wps.to_numpy().max()}")
        self.logger.info(f"Обновлены доли растворенного парафина: min={self.new_wp.to_numpy().min()}  max={self.new_wp.to_numpy().max()}")

        self.logger.info(f"Обновлена доля выпадающего парафина:   min={self.new_qp.to_numpy().min()}  max={self.new_qp.to_numpy().max()}")
        self.logger.info(f"Обновлен множитель пористости:         min={self.m_mult.to_numpy().min()}  max={self.m_mult.to_numpy().max()}")
        self.logger.info(f"Обновлен множитель проницаемости:      min={self.k_mult.to_numpy().min()}  max={self.k_mult.to_numpy().max()}")

        self.logger.info(f"Обновлена толщина осадочного слоя:              min={self.new_h.to_numpy().min()}  max={self.new_h.to_numpy().max()}")
        self.logger.info(f"Обновлена скорость изменения радиуса капилляра: min={self.new_Ur.to_numpy().min()}  max={self.new_Ur.to_numpy().max()}")
        self.logger.info(f"Обновлена скорость блокировки капилляров:       min={self.new_Ub.to_numpy().min()}  max={self.new_Ub.to_numpy().max()}")

