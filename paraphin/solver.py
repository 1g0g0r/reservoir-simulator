from logging import basicConfig, INFO, getLogger
from pickle import dump

import numpy as np
import psutil
import taichi as ti

from paraphin import r1, r3, r4, r5, r6, fi_0
from paraphin.constants import (data_type, Nx, Ny, Nr, rw, results_path, logs_path, init_T, init_k, init_S,
                                init_m, init_p, init_qp, init_h_sloy, init_Wp, init_Wps, bar_to_pa, eta, h, dt,
                                day_to_sec, mu_o, mu_w, c_o, c_w, c_p, c_f, sol_time_step, Time_end, LOGGING, _re)
from paraphin.equations import (calc_qp_m_k_fi, calc_pressure, saturation_equation, saturation_well, temperature_well,
                                temperature_equation, wps_wp_equation, wps_wp_wells, calc_velocitys_h, flows_in_cells)
from paraphin.utils import calc_mu_o, calc_mu_w, calc_c_f, calc_c_o, calc_c_w, calc_c_p, preprocess_matrix_and_wells
from paraphin.well import WellStruct, upd_q_and_eta


@ti.data_oriented
class Solver:
    def __init__(self, d_type = data_type):
        self.d_type = d_type

        # Скважины
        self.n_wells = 0
        self._wells_buffer = []
        self._wells_names = []
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
        self.T     = ti.field(dtype=d_type, shape=(Nx, Ny))  # Температура, [С]
        self.T_0   = ti.field(dtype=d_type, shape=(Nx, Ny))

        # Динамика образования парафина (кольматация\суффозия)
        self.integr_r2_fi0 = ti.field(dtype=d_type, shape=())
        self.integr_r4_fi0 = ti.field(dtype=d_type, shape=())
        self._Um_r2  = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.qp      = ti.field(dtype=d_type, shape=(Nx, Ny))  # Скорость отложения парафина в общем объеме
        self.fi      = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.h_sloy  = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.Ur      = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.Ub      = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))

        # Поля данный нового временного слоя
        self.new_h   = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.new_Ur  = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.new_Ub  = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.new_fi  = ti.field(dtype=d_type, shape=(Nx, Ny, Nr))
        self.new_s   = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.new_t   = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.new_wps = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.new_wp  = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.new_qp  = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.m_mult  = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.k_mult  = ti.field(dtype=d_type, shape=(Nx, Ny))

        # Временные массивы
        self.cells_T_eq  = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.cells_Wp_eq = ti.field(dtype=d_type, shape=(Nx, Ny))
        self.cells_S_eq  = ti.field(dtype=d_type, shape=(Nx, Ny))

        # Вспомогательные поля класса
        self._i_img = 0
        self._paraphin = not np.isclose(init_Wp + init_Wps, 0.0)
        self.rows_indices = np.ndarray
        self.cols_indices = np.ndarray
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
            self.logger.handlers = []
            self.logger.propagate = False


    def initialize(self):
        def _well_processing():
            self._wells_names = [i['name'] for i in self._wells_buffer]
            self.wells = WellStruct.field(shape=self.n_wells)
            self.rows_indices, self.cols_indices, self.wells = preprocess_matrix_and_wells(self.wells, self._wells_buffer, self.p, self.S, self.k, self.mu_o, self.mu_w)

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
            for i in ti.ndrange(Nx):
                for j in ti.ndrange(Ny):
                    # Параметры пласта
                    self.p[i, j]    = init_p
                    self.S[i, j]    = init_S
                    self.S_0[i, j]  = init_S
                    self.Wo[i, j]   = 1.0 - init_Wp - init_Wps
                    self.Wp[i, j]   = init_Wp
                    self.Wp_0[i, j] = init_Wp
                    self.Wps[i, j]  = init_Wps
                    self.k[i, j]    = init_k
                    self.m[i, j]    = init_m
                    self.m_0[i, j]  = init_m
                    self.T[i, j]    = init_T
                    self.T_0[i, j]  = init_T
                    self.qp[i, j]   = init_qp
                    self.m_mult[i, j] = 1.0
                    self.k_mult[i, j] = 1.0

                    # свойства флюидов
                    self.mu_o[i, j] = mu_o
                    self.mu_w[i, j] = mu_w
                    self.C_w[i, j] = c_w
                    self.C_o[i, j] = c_o
                    self.C_f[i, j] = c_f
                    self.C_p[i, j] = c_p

                    # Поля данный нового временного слоя
                    self.new_wps[i, j] = init_Wps
                    self.new_wp[i, j] = init_Wp

                    for ij in ti.ndrange(Nr):
                        self.fi[i, j, ij]     = fi_o[ij]
                        self.new_fi[i, j, ij]     = fi_o[ij]
                        self.h_sloy[i, j, ij] = init_h_sloy
                        self.Ur[i, j, ij] = 0.0
                        self.Ub[i, j, ij] = 0.0

        _calc_integrals(fi_o=fi_0)
        _initialize_params_loop(fi_o=fi_0)
        _well_processing()


    def add_well(self, name: str, i: int, j: int, p: float, is_injector: bool = False, T: float = 0.0, rw: float = rw):
        """Добавление скважин в расчет"""
        productivity_mult = 2.0 * np.pi * h / np.log(_re / rw) * 0.25
        well = WellStruct(i=i, j=j, p=p, T=T, rw=rw, is_injector=int(is_injector), productivity_mult=productivity_mult)
        self._wells_buffer.append({'well': well, 'name': name})
        self.n_wells += 1


    def upd_time_step(self, t: float) -> None:
        """Решение задачи на текущем временном слое."""
        self._process_time_step()
        self._logging_solution(t)
        self._swap_time_steps()

        # Запись данных в файл
        if t >= self._i_img * sol_time_step or np.isclose(t, Time_end):
            self._save_results(t)
            self._i_img += 1


    def _update_p(self) -> None:
        """Обновление давления."""
        calc_pressure(self.p, self.Wo, self.m, self.m_0, self.k, self.S, self.mu_o, self.mu_w, self.wells, self.rows_indices, self.cols_indices)

    @ti.kernel
    def _update_wells_data(self):
        """Обновление дебетов и обводненности скважин."""
        ti.loop_config(serialize=True)  # parallelize=1
        for i in ti.ndrange(self.n_wells):
            self.wells[i] = upd_q_and_eta(self.wells[i], self.p, self.S, self.k, self.mu_o, self.mu_w)


    def _process_time_step(self):
        """Метод IMPES: явный по насыщенности неявный по давлению."""
        self._update_p()           # Обновление давления
        self._update_wells_data()  # Обновление дебетов скважин
        self._equations_loop()     # Решение уравнений по явной схеме
        self._wells_loop()         # Учет скважин в уравнениях
        if self._paraphin:
            # Средняя скорость в капилляре * r^2
            self._Um_r2.from_numpy(np.linalg.norm(np.gradient(self.p.to_numpy()), axis=0) / self.mu_o.to_numpy() * 0.125 / eta)


    @ti.kernel
    def _wells_loop(self):
        ti.loop_config(serialize=True)
        for i in ti.ndrange(self.n_wells):
            saturation_well(self.wells[i], self.m, self.new_s)
            wps_wp_wells(self.wells[i], self.m, self.S, self.Wp, self.Wps, self.new_wp)
            temperature_well(self.wells[i], self.T, self.m, self.S, self.C_o, self.C_w, self.C_f, self.C_p, self.Wps, self.new_t)

    @ti.kernel
    def _equations_loop(self):
        for i in ti.ndrange(Nx):
            for j in ti.ndrange(Ny):
                # ---решение гидродинамики---
                flows_in_cells(i, j, self.p, self.S, self.T, self.k, self.mu_o, self.mu_w, self.m, self.Wp, self.Wps, self.C_o, self.C_w, self.C_p, self.cells_T_eq, self.cells_Wp_eq, self.cells_S_eq)
                saturation_equation(i, j, self.S, self.m, self.m_0, self.cells_S_eq, self.new_s)
                temperature_equation(i, j, self.T, self.m, self.m_0, self.S, self.S_0, self.C_o, self.C_w, self.C_f, self.C_p, self.Wps, self.Wps_0, self.cells_T_eq, self.new_t)
                # ---решение задачи кольматации\суффозии---
                if self._paraphin:
                    # Обновление концентраций парафина
                    wps_wp_equation(i, j, self.qp, self.m, self.m_0, self.S, self.S_0, self.Wo, self.Wp, self.Wp_0, self.Wps, self.Wps_0, self.T, self.T_0, self.cells_Wp_eq, self.new_wp, self.new_wps)
                    # Обновление толщины осадочного слоя, скорости изменения радиуса капилляра и скорости блокирования капилляров
                    calc_velocitys_h(i, j, self.S, self._Um_r2, self.Wps, self.mu_o, self.fi, self.h_sloy, self.Ur, self.new_h, self.new_Ur, self.new_Ub)
                    # Обновление функции пор по размерам, объема выделяемого парафина, пористости, проницаемости
                    calc_qp_m_k_fi(i, j, self.Wps, self.m, self.fi, self.Ur, self.Ub, self.integr_r2_fi0[None], self.integr_r4_fi0[None], self.new_qp, self.new_fi, self.k_mult, self.m_mult)
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
        for i in ti.ndrange(Nx):
            for j in ti.ndrange(Ny):
                self.S_0[i, j] = self.S[i, j]
                self.S[i, j]   = self.new_s[i, j]
                self.T_0[i, j] = self.T[i, j]
                self.T[i, j]   = self.new_t[i, j]

                if self._paraphin:
                    self.Wp_0[i, j]  = self.Wp[i, j]
                    self.Wp[i, j]    = self.new_wp[i, j]
                    self.Wps_0[i, j] = self.Wps[i, j]
                    self.Wps[i, j]   = self.new_wps[i, j]
                    self.k[i, j]     = init_k * self.k_mult[i, j]
                    self.m[i, j]     = init_m * self.m_mult[i, j]
                    self.m_0[i, j]   = self.m[i, j]  # FIXME разобраться с производной
                    self.qp[i, j]    = self.new_qp[i, j]

                    for ij in ti.ndrange(Nr):
                        self.fi[i, j, ij]     = self.new_fi[i, j, ij]
                        self.h_sloy[i, j, ij] = self.new_h[i, j, ij]
                        self.Ur[i, j, ij]     = self.new_Ur[i, j, ij]
                        self.Ub[i, j, ij]     = self.new_Ub[i, j, ij]


    def _save_results(self, t) -> None:
        """Сохранение полей данных в файл формата pkl."""
        if np.isclose(t, 0.0):
            for file_path in results_path.glob(f'*.pkl'):  # Перебор всех файлов .pkl
                file_path.unlink()
            self.logger.info('Старые файлы удалены.')

        wells_data = {}
        for i in range(self.n_wells):
            name = self._wells_names[i]
            well = self.wells[i]
            q_value = well.q
            wells_data.update({
                f'{name}_oil': q_value[0], f'{name}_water': q_value[1],
                f'{name}_total': q_value[2], f'{name}_eta': well.eta
            })

        data = {
            'Time':        t,
            'Pressure':    self.p.to_numpy(),
            'Saturation':  self.S.to_numpy(),
            'Temperature': self.T.to_numpy(),
            # 'Wp':          self.Wp.to_numpy(),
            'Wps':         self.Wps.to_numpy(),
            'm mult':      self.m_mult.to_numpy(),
            'k mult':      self.k_mult.to_numpy(),
            'plots':       {'fi_o': fi_0,
                            'fi': self.fi.to_numpy()[0, 0],
                            # 'dfi': fi_0 - self.fi.to_numpy()[0, 0],
                            'Ub': self.Ub.to_numpy()[0, 0],
                            'Ur': self.Ur.to_numpy()[0, 0],
                            },
            'Wells':       wells_data,
            'Other params': {
                'Wps [0,0]':  self.Wps.to_numpy()[0,0],
                'Wp [0,0]': self.Wp.to_numpy()[0,0],
                'Wp [0,0] + Wps [0,0]': self.Wp.to_numpy()[0,0]+self.Wps.to_numpy()[0,0],
                'k_mult [0,0]':  self.k_mult.to_numpy()[0,0],
                'm_mult [0,0]': self.m_mult.to_numpy()[0,0],
                'qp [0,0]': self.qp.to_numpy()[0,0],
            }
        }
        with open(results_path / f'data_{t / day_to_sec}.pkl', 'wb') as f:
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
        self.logger.info(f"ВРЕМЕННОЙ СЛОЙ t = {round(t / day_to_sec, 5)} день ({int(t / dt)} итерация)")
        self.logger.info(f"Обновлено давление (бар): min={self.p.to_numpy().min() / bar_to_pa}  max={self.p.to_numpy().max() / bar_to_pa}")
        self.logger.info(f"Обновлена насыщенность:   min={self.new_s.to_numpy().min()}  max={self.new_s.to_numpy().max()}")
        self.logger.info(f"Обновлена температура:    min={self.new_t.to_numpy().min()}  max={self.new_t.to_numpy().max()}")
        for i in range(self.n_wells):
            self.logger.info(f"Дебет скважины {self._wells_buffer[i]['name']} (м^3/сут): q_o={self.wells[i].q[0] * day_to_sec}  q_w={self.wells[i].q[1] * day_to_sec}")

        if not self._paraphin:
            return None
        self.logger.info(f"Wps:  min={self.new_wps.to_numpy().min()}  max={self.new_wps.to_numpy().max()}")
        self.logger.info(f"Wp:   min={self.new_wp.to_numpy().min()}  max={self.new_wp.to_numpy().max()}")
        oil_components = self.new_wp.to_numpy() + self.new_wps.to_numpy() + self.Wo.to_numpy()
        self.logger.info(f"Wp+Wps+Wo:   min={oil_components.min()}  max={oil_components.max()}")

        self.logger.info(f"qp:     min={self.new_qp.to_numpy().min()}  max={self.new_qp.to_numpy().max()}")
        self.logger.info(f"m_mult: min={self.m_mult.to_numpy().min()}  max={self.m_mult.to_numpy().max()}")
        self.logger.info(f"k_mult: min={self.k_mult.to_numpy().min()}  max={self.k_mult.to_numpy().max()}")

        self.logger.info(f"fi:   min={self.fi.to_numpy().min()}  max={self.fi.to_numpy().max()}")
        self.logger.info(f"Ur:   min={self.new_Ur.to_numpy().min()}  max={self.new_Ur.to_numpy().max()}")
        self.logger.info(f"Ub:   min={self.new_Ub.to_numpy().min()}  max={self.new_Ub.to_numpy().max()}")
        self.logger.info(f"Um:   min={self._Um_r2.to_numpy().min()*1e-12}  max={self._Um_r2.to_numpy().max()*1e-12}")

        # self.logger.info(f"fi:   {' '.join([f'{x:.{3}f}' for x in self.fi.to_numpy()[0, 0]])}")
        # self.logger.info(f"fi_0: {' '.join([f'{x:.{3}f}' for x in fi_0])}")
