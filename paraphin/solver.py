"""Класс содержит алгоритм расчета и хранение данных."""
from logging import basicConfig, INFO, getLogger
from sys import stdout
from time import perf_counter

import numpy as np
import psutil
import taichi as ti
from tqdm import tqdm

from paraphin import r1, r3, r4, r5, r6, fi_0
from .constants import (data_type, Nx, Ny, Nr, rw, results_path, data_path, logs_path, init_T, init_k, init_S, init_m,
                        init_p, init_qp, init_h_sloy, init_Wp, init_Wps, bar_to_pa, h, dt, day_to_sec, ro_p, ro_o,
                        max_eta, c_o, c_w, c_p, c_f, sol_time_step, Time_end, LOGGING, _re, geological_reserves)
from .equations import (calc_qp_m_k_fi, calc_pressure, saturation_equation, saturation_well, temperature_well,
                        temperature_equation, wps_wp_equation, wps_wp_wells, calc_velocitys_h, flows_in_cells,
                        calc_Um_r2)
from .utils import (calc_mu_o, calc_mu_w, preprocess_matrix_and_wells, convert_pkl_files, save_fields,
                    Bound, TypeBC, DataField, add_bc)
from .well import WellStruct, upd_q_and_eta


@ti.data_oriented
class Solver:
    def __init__(self):
        self.d_type = data_type

        # Граничные условия и скважины
        self.KIN = ti.field(dtype=data_type, shape=())
        self.n_wells = 0
        self._wells_buffer = []
        self._wells_names = []
        self.wells = WellStruct.field(shape=1)
        self.boundary_conditions = ti.field(dtype=data_type, shape=(4, 3, 2))  # Граница -> Поле -> Тип, Значение

        # Свойства флюидов
        self.mu_o = ti.field(dtype=data_type, shape=(Nx, Ny))  # Вязкость нефти, [Па*с]
        self.mu_w = ti.field(dtype=data_type, shape=(Nx, Ny))  # Вязкость воды, [Па*с]
        self.C_w  = ti.field(dtype=data_type, shape=(Nx, Ny))  # Теплоемкость воды, [Дж*кг/C]
        self.C_o  = ti.field(dtype=data_type, shape=(Nx, Ny))  # Теплоемкость нефти, [Дж*кг/C]
        self.C_f  = ti.field(dtype=data_type, shape=(Nx, Ny))  # Теплоемкость пласта, [Дж*кг/C]
        self.C_p  = ti.field(dtype=data_type, shape=(Nx, Ny))  # Теплоемкость парафина, [Дж*кг/C]

        # Поля данных
        self.p     = ti.field(dtype=data_type, shape=(Nx, Ny))  # Давление, [Па]
        self.S     = ti.field(dtype=data_type, shape=(Nx, Ny))  # Водонасыщенность, [-]
        self.S_0   = ti.field(dtype=data_type, shape=(Nx, Ny))
        self.Wo    = ti.field(dtype=data_type, shape=(Nx, Ny))  # Массовая доля маслянного компонента в нефти, [-]
        self.Wo_0  = ti.field(dtype=data_type, shape=(Nx, Ny))  # Массовая доля маслянного компонента в нефти, [-]
        self.Wp    = ti.field(dtype=data_type, shape=(Nx, Ny))  # Массовая доля растворенного парафина в нефти, [-]
        self.Wp_0  = ti.field(dtype=data_type, shape=(Nx, Ny))
        self.Wps   = ti.field(dtype=data_type, shape=(Nx, Ny))  # Массовая доля взвешенного парафина в нефти, [-]
        self.Wps_0 = ti.field(dtype=data_type, shape=(Nx, Ny))
        self.Wps_dep = ti.field(dtype=data_type, shape=(Nx, Ny))
        self.k     = ti.field(dtype=data_type, shape=(Nx, Ny))  # Проницаемость, [м^2]
        self.m     = ti.field(dtype=data_type, shape=(Nx, Ny))  # Пористость, [-]
        self.m_0   = ti.field(dtype=data_type, shape=(Nx, Ny))
        self.T     = ti.field(dtype=data_type, shape=(Nx, Ny))  # Температура, [С]
        self.T_0   = ti.field(dtype=data_type, shape=(Nx, Ny))

        # Динамика образования парафина (кольматация\суффозия)
        self.integr_r2_fi0 = ti.field(dtype=data_type, shape=())
        self.integr_r4_fi0 = ti.field(dtype=data_type, shape=())
        self._Um_r2  = ti.field(dtype=data_type, shape=(Nx, Ny))
        self.qp      = ti.field(dtype=data_type, shape=(Nx, Ny))  # Скорость отложения парафина в общем объеме
        self.fi      = ti.field(dtype=data_type, shape=(Nx, Ny, Nr))
        self.h_sloy  = ti.field(dtype=data_type, shape=(Nx, Ny, Nr))
        self.Ur      = ti.field(dtype=data_type, shape=(Nx, Ny, Nr))
        self.Ub      = ti.field(dtype=data_type, shape=(Nx, Ny, Nr))

        # Поля данный нового временного слоя
        self.new_h   = ti.field(dtype=data_type, shape=(Nx, Ny, Nr))
        self.new_Ur  = ti.field(dtype=data_type, shape=(Nx, Ny, Nr))
        self.new_Ub  = ti.field(dtype=data_type, shape=(Nx, Ny, Nr))
        self.new_fi  = ti.field(dtype=data_type, shape=(Nx, Ny, Nr))
        self.new_s   = ti.field(dtype=data_type, shape=(Nx, Ny))
        self.new_t   = ti.field(dtype=data_type, shape=(Nx, Ny))
        self.new_wps = ti.field(dtype=data_type, shape=(Nx, Ny))
        self.new_wp  = ti.field(dtype=data_type, shape=(Nx, Ny))
        self.new_qp  = ti.field(dtype=data_type, shape=(Nx, Ny))
        self.new_k   = ti.field(dtype=data_type, shape=(Nx, Ny))
        self.new_m   = ti.field(dtype=data_type, shape=(Nx, Ny))

        # Временные массивы перетоков через границы ячеек
        self.cells_T_eq  = ti.field(dtype=data_type, shape=(Nx, Ny))
        self.cells_Wp_eq = ti.field(dtype=data_type, shape=(Nx, Ny))
        self.cells_S_eq  = ti.field(dtype=data_type, shape=(Nx, Ny))

        # Вспомогательные поля класса
        self._i_img = 0
        self._paraphin = not np.isclose(init_Wp + init_Wps, 0.0)
        self.rows_indices = np.ndarray
        self.cols_indices = np.ndarray
        self.sort_mask = np.ndarray
        self.cols_ptr = np.ndarray
        results_path.mkdir(parents=True, exist_ok=True)
        data_path.mkdir(parents=True, exist_ok=True)

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
        def _wells_and_matrix_processing():
            self._wells_names = [i['name'] for i in self._wells_buffer]
            if self.n_wells > 0:
                self.wells = WellStruct.field(shape=self.n_wells)

            temp_data = preprocess_matrix_and_wells(self.wells, self._wells_buffer)
            self.sort_mask, self.rows_indices, self.cols_ptr, self.wells = temp_data

        @ti.kernel
        def _calc_integrals():
            """Вычисление интегралов от функций r^4*fi_o(r) и r^2*fi_o(r)"""
            self.integr_r2_fi0[None] = 0.0
            self.integr_r4_fi0[None] = 0.0

            for i in ti.ndrange((1, Nr)):
                dr = r1[i] - r1[i - 1]
                a = (fi_0[i - 1] * r1[i] - fi_0[i] * r1[i - 1]) / dr
                b = (fi_0[i] - fi_0[i - 1]) / dr
                self.integr_r2_fi0[None] += (r3[i] - r3[i-1]) * a / 3 + (r4[i] - r4[i-1]) * b / 4  # r^2 * fi
                self.integr_r4_fi0[None] += (r5[i] - r5[i-1]) * a / 5 + (r6[i] - r6[i-1]) * b / 6  # r^4 * fi

        @ti.kernel
        def _initialize_params_loop():
            for i, j in self.p:
                # Параметры пласта
                self.p[i, j]    = init_p
                self.S[i, j]    = init_S
                self.S_0[i, j]  = init_S
                self.Wo[i, j]   = 1.0 - init_Wp - init_Wps
                self.Wo_0[i, j] = self.Wo[i, j]
                self.Wp[i, j]   = init_Wp
                self.Wp_0[i, j] = init_Wp
                self.Wps[i, j]  = init_Wps
                self.Wps_dep[i, j] = 0.0
                self.k[i, j]    = init_k
                self.m[i, j]    = init_m
                self.m_0[i, j]  = init_m
                self.T[i, j]    = init_T
                self.T_0[i, j]  = init_T
                self.qp[i, j]   = init_qp

                # свойства флюидов
                self.mu_o[i, j] = calc_mu_o(init_T)
                self.mu_w[i, j] = calc_mu_w(init_T)
                self.C_w[i, j] = c_w  # calc_c_w(init_T)
                self.C_o[i, j] = c_o  # calc_c_o(init_T)
                self.C_f[i, j] = c_f  # calc_c_f(init_T)
                self.C_p[i, j] = c_p  # calc_c_p(init_T)

                # Поля данный нового временного слоя
                self.new_m[i, j]   = init_m
                self.new_k[i, j]   = init_k
                self.new_wps[i, j] = init_Wps
                self.new_wp[i, j]  = init_Wp

                for ij in ti.ndrange(Nr):
                    self.fi[i, j, ij]     = fi_0[ij]
                    self.new_fi[i, j, ij] = fi_0[ij]
                    self.h_sloy[i, j, ij] = init_h_sloy
                    self.Ur[i, j, ij] = 0.0
                    self.Ub[i, j, ij] = 0.0

        _calc_integrals()
        _initialize_params_loop()
        _wells_and_matrix_processing()
        for file_path in results_path.glob(f'*.pkl'):
            file_path.unlink()


    def add_bc(self, field: DataField, bound: Bound, type_bc: TypeBC, value: float) -> None:
        """Учет граничных условий для полей данных."""
        add_bc(self.boundary_conditions, bound.value, field.value, type_bc.value, value)


    def add_well(self, name: str, i: int, j: int, p: float, is_injector: bool = False, T: float = 0.0,
                 rw: float = rw, mult: float = 1.0) -> None:
        """Добавление скважин в расчет."""
        productivity_mult = 2.0 * np.pi * h / np.log(_re / rw) * mult
        well = WellStruct(i=i, j=j, p=p, T=T, rw=rw, is_injector=int(is_injector), productivity_mult=productivity_mult)
        self._wells_buffer.append({'well': well, 'name': name})
        self.n_wells += 1


    def upd_time_step(self, t: float) -> None:
        """Решение задачи на текущем временном слое."""
        self._process_time_step()
        self._logging_solution(t)
        self._swap_time_steps()

        # Запись данных в файл
        if t >= self._i_img * sol_time_step or np.isclose(t, Time_end) or self.wells[1].eta >= max_eta:
            save_fields(self, t)
            self._i_img += 1


    def _process_time_step(self):
        """Метод IMPES: явный по насыщенности неявный по давлению."""
        # Обновление давления
        calc_pressure(self.p, self.Wo, self.Wo_0, self.m, self.m_0, self.k, self.S, self.S_0, self.mu_o,
                      self.mu_w, self.wells, self.rows_indices, self.cols_ptr, self.sort_mask, self.boundary_conditions)
        # Решение уравнений по явной схеме
        self._equations_loop()


    @ti.kernel
    def _equations_loop(self):
        """Решение уравнений по явной схеме в цикле по ячейкам."""
        self._update_wells_data()  # Обновление данных скважин
        self._wells_loop()         # Учет скважин в уравнениях

        for i, j in self.p:
            # ---решение задачи кольматации\суффозии---
            if self._paraphin:
                # Средняя скорость в капилляре * r^2
                calc_Um_r2(i, j, self.p, self._Um_r2, self.mu_o)
                # Обновление концентраций парафина
                wps_wp_equation(i, j, self.qp, self.m, self.m_0, self.S, self.S_0, self.Wo, self.Wp, self.Wp_0, self.Wps, self.Wps_0, self.T, self.T_0, self.cells_Wp_eq, self.new_wp, self.new_wps)
                # Обновление толщины осадочного слоя, скорости изменения радиуса капилляра и скорости блокирования капилляров
                calc_velocitys_h(i, j, self.S, self._Um_r2, self.Wps, self.mu_o, self.fi, self.h_sloy, self.Ur, self.new_h, self.new_Ur, self.new_Ub)
                # Обновление функции пор по размерам, объема выделяемого парафина, пористости, проницаемости
                calc_qp_m_k_fi(i, j, self.Wps, self.m, self.fi, self.Ur, self.Ub, self.integr_r2_fi0[None], self.integr_r4_fi0[None], self.new_qp, self.new_fi, self.new_k, self.new_m)

            # ---решение гидродинамики---
            flows_in_cells(i, j, self.boundary_conditions, self.p, self.S, self.T, self.k, self.mu_o, self.mu_w, self.m, self.Wp, self.Wps, self.C_o, self.C_w, self.C_p, self.cells_T_eq, self.cells_Wp_eq, self.cells_S_eq)
            saturation_equation(i, j, self.S, self.m, self.m_0, self.cells_S_eq, self.new_m, self.new_s)
            temperature_equation(i, j, self.T, self.m, self.m_0, self.S, self.S_0, self.C_o, self.C_w, self.C_f, self.C_p, self.Wps, self.Wps_0, self.qp, self.cells_T_eq, self.new_t, self.new_m, self.new_s)


    @ti.func
    def _wells_loop(self):
        """Учет скважин в решении уравнений по явной схеме."""
        for i in self.wells:
            saturation_well(self.wells[i], self.m, self.new_m, self.new_s)
            wps_wp_wells(self.wells[i], self.m, self.S, self.T, self.Wp, self.Wps, self.new_wp)
            temperature_well(self.wells[i], self.T, self.m, self.S, self.C_o, self.C_w, self.C_f, self.C_p, self.Wps, self.new_t)


    @ti.func
    def _update_wells_data(self):
        """Обновление дебита и обводненности скважин."""
        Q_oil = 0.0
        for i in self.wells:
            self.wells[i] = upd_q_and_eta(self.wells[i], self.p, self.S, self.k, self.mu_o, self.mu_w)
            if self.wells[i].is_injector == 0:
                Q_oil += self.wells[1].Q[0]

        # Вычисление КИН
        self.KIN[None] = Q_oil / geological_reserves


    @ti.func
    def _update_mu_and_c_temp(self, i, j) -> None:
        self.mu_o[i, j] = calc_mu_o(self.T[i, j])
        self.mu_w[i, j] = calc_mu_w(self.T[i, j])
        self.C_w[i, j]  = c_w  # calc_c_w(self.T[i, j])
        self.C_o[i, j]  = c_o  # calc_c_o(self.T[i, j])
        self.C_f[i, j]  = c_f  # calc_c_f(self.T[i, j])
        self.C_p[i, j]  = c_p  # calc_c_p(self.T[i, j])


    @ti.kernel
    def _swap_time_steps(self):
        """Обновление полей данных на новом временном слое."""
        for i, j in self.p:
            self._update_mu_and_c_temp(i, j)  #  пересчет свойств флюидов из-за изменения температуры

            self.S_0[i, j] = self.S[i, j]
            self.S[i, j]   = self.new_s[i, j]
            self.new_s[i, j] = 0.0
            self.T_0[i, j] = self.T[i, j]
            self.T[i, j]   = self.new_t[i, j]
            self.new_t[i, j] = 0.0

            if self._paraphin:
                self.Wo_0[i, j]  = self.Wo[i, j]
                self.Wo[i, j]    = 1.0 - self.new_wp[i, j] - self.new_wps[i, j]
                self.Wp_0[i, j]  = self.Wp[i, j]
                self.Wp[i, j]    = self.new_wp[i, j]
                self.new_wp[i, j] = 0.0
                self.Wps_0[i, j] = self.Wps[i, j]
                self.Wps[i, j]   = self.new_wps[i, j]
                self.k[i, j]     = self.new_k[i, j]
                self.m[i, j]     = self.new_m[i, j]
                self.m_0[i, j]   = self.m[i, j]  # FIXME разобраться с производной
                self.Wps_dep[i, j] = ti.min(self.Wps_dep[i, j] - self.qp[i, j] * dt * ro_p /
                                            ((1.0-self.Wps[i,j]) * ro_o + self.Wps[i,j] * ro_p), init_Wp)
                self.qp[i, j]    = self.new_qp[i, j]

                for ij in ti.ndrange(Nr):
                    self.fi[i, j, ij]     = self.new_fi[i, j, ij]
                    self.h_sloy[i, j, ij] = self.new_h[i, j, ij]
                    self.Ur[i, j, ij]     = self.new_Ur[i, j, ij]
                    self.Ub[i, j, ij]     = self.new_Ub[i, j, ij]


    def start(self):
        try:
            times = np.linspace(0, Time_end, int(Time_end / dt + 1))
            tt = perf_counter()
            self.initialize()      # Задание начальных условий из файла const.py
            self.upd_time_step(0)  # При первом запуске компилируются модули
            print('Время компиляции:', perf_counter() - tt)

            with tqdm(iterable=times[1:], ncols=90, desc='Решение задачи', file=stdout, smoothing=0.05,
                      bar_format="{l_bar}{bar}[{elapsed}/{remaining}]  {n_fmt}/{total_fmt}{postfix}   ") as pbar:
                for _t in pbar:
                    # ti.profiler.print_kernel_profiler_info()
                    self.upd_time_step(_t)
                    pbar.set_postfix(день=_t / day_to_sec)
                    if self.wells[1].eta >= max_eta:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            print('KIN:', round(self.KIN[None], 5))
            print('eta:', round(self.wells[1].eta, 5))
            convert_pkl_files()


    def _logging_resources(self) -> None:
        # cpu_usage = psutil.cpu_percent(interval=None)  # , percpu=True
        memory_info = psutil.virtual_memory()
        memory_usage = round(memory_info.used / memory_info.total * 100 , 5)  # memory_info.percent
        # self.logger.info(f'Использование CPU: {cpu_usage}%')
        self.logger.info(f'Использование памяти: {memory_usage}%')


    def _logging_solution(self, t):
        """Логирование решения задачи."""
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
        self.logger.info(f"m_mult: min={(init_m / self.new_m.to_numpy()).min()}  max={(init_m / self.new_m.to_numpy()).max()}")
        self.logger.info(f"k_mult: min={(init_k / self.new_k.to_numpy()).min()}  max={(init_k / self.new_k.to_numpy()).max()}")

        self.logger.info(f"fi:   min={self.fi.to_numpy().min()}  max={self.fi.to_numpy().max()}")
        self.logger.info(f"Ur:   min={self.new_Ur.to_numpy().min()}  max={self.new_Ur.to_numpy().max()}")
        self.logger.info(f"Ub:   min={self.new_Ub.to_numpy().min()}  max={self.new_Ub.to_numpy().max()}")
        self.logger.info(f"Um:   min={self._Um_r2.to_numpy().min()*1e-12}  max={self._Um_r2.to_numpy().max()*1e-12}")

        # self.logger.info(f"fi:   {' '.join([f'{x:.{3}f}' for x in self.fi.to_numpy()[0, 0]])}")
        # self.logger.info(f"fi_0: {' '.join([f'{x:.{3}f}' for x in fi_0])}")
