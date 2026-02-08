"""Класс содержит алгоритм расчета и хранение данных."""
from logging import basicConfig, INFO, getLogger
from shutil import rmtree
from sys import stdout
from time import perf_counter

import numpy as np
import psutil
from numba import njit, prange
from tqdm import tqdm

from paraphin import N, NN, r1, r3, r4, r5, r6, fi_0
from .constants import (data_type, Nx, Ny, Nr, rw, results_path, data_path, logs_path, init_T, init_k, init_S, init_m,
                        init_p, init_qp, init_h_sloy, init_Wp, init_Wps, bar_to_pa, dt, day_to_sec, ro_p, ro_o,
                        max_eta, c_o, c_w, c_p, c_f, c_ff, sol_time_step, Time_end, LOGGING, geological_reserves)
from .equations import (calc_qp_m_k_fi, calc_pressure, saturation_equation, saturation_well, temperature_well,
                        temperature_equation, wps_wp_equation, wps_wp_wells, calc_velocitys_h, flows_in_cells,
                        calc_Um_r2)
from .utils import (calc_mu_o, calc_mu_w, preprocess_matrix_and_wells, convert_pkl_files, save_fields,
                    Bound, TypeBC, DataField, add_bc, WellStruct, upd_q_and_eta)


class Solver:
    def __init__(self):
        # Граничные условия и скважины
        self.KIN = data_type(0.0)
        self.n_wells = 0
        self._wells_buffer = []
        self._wells_names = []
        self.wells = np.array([], dtype=WellStruct)
        self.boundary_conditions = np.zeros(dtype=data_type, shape=(4, 3, 2))  # Граница -> Поле -> Тип, Значение
        # Свойства флюидов
        self.mu_o = np.full((Nx, Ny), calc_mu_o(init_T), data_type)  # Вязкость нефти, [Па*с]
        self.mu_w = np.full((Nx, Ny), calc_mu_w(init_T), data_type)  # Вязкость воды, [Па*с]
        self.C_w  = np.full((Nx, Ny), c_w, data_type)  # Теплоемкость воды, [Дж*кг/C]
        self.C_o  = np.full((Nx, Ny), c_o, data_type)  # Теплоемкость нефти, [Дж*кг/C]
        self.C_f  = np.full((Nx, Ny), c_f, data_type)  # Теплоемкость пласта, [Дж*кг/C]
        self.C_ff = np.full((Nx, Ny), c_f, data_type)  # Теплоемкость окружающих пород пласта, [Дж*кг/C]
        self.C_p  = np.full((Nx, Ny), c_p, data_type)  # Теплоемкость парафина, [Дж*кг/C]
        # Поля данных пласта
        self.p     = np.full((Nx, Ny), init_p, data_type)  # Давление, [Па]
        self.S     = np.full((Nx, Ny), init_S, data_type)  # Водонасыщенность, [-]
        self.S_0   = np.full((Nx, Ny), init_S, data_type)  # Водонасыщенность на прошлом временном слое, [-]
        self.T     = np.full((Nx, Ny), init_T, data_type)  # Температура, [С]
        self.T_0   = np.full((Nx, Ny), init_T, data_type)  # Температура на прошлом временном слое, [С]
        self.Wo    = np.full((Nx, Ny), 1.0 - init_Wp - init_Wps, data_type)  # Массовая доля маслянного компонента в нефти, [-]
        self.Wo_0  = np.full((Nx, Ny), 1.0 - init_Wp - init_Wps, data_type)  # Массовая доля маслянного компонента в нефти, [-]
        self.Wp    = np.full((Nx, Ny), init_Wp, data_type)   # Массовая доля растворенного парафина в нефти, [-]
        self.Wp_0  = np.full((Nx, Ny), init_Wp, data_type)   # Массовая доля растворенного парафина в нефти на прошлом временном слое, [-]
        self.Wps   = np.full((Nx, Ny), init_Wps, data_type)  # Массовая доля взвешенного парафина в нефти, [-]
        self.Wps_0 = np.full((Nx, Ny), init_Wps, data_type)  # Массовая доля взвешенного парафина в нефти на прошлом временном слое, [-]
        self.Wps_dep = np.full((Nx, Ny), 0.0, data_type)# Массовая доля осевшего на порах парафина, [-]
        self.k     = np.full((Nx, Ny), init_k, data_type)  # Проницаемость, [м^2]
        self.m     = np.full((Nx, Ny), init_m, data_type)  # Пористость, [-]
        self.m_0   = np.full((Nx, Ny), init_m, data_type)  # Пористость на прошлом временном слое, [-]
        # Динамика образования парафина (кольматация\суффозия)
        self.integr_r2_fi0 = data_type(0.0)
        self.integr_r4_fi0 = data_type(0.0)
        self.fi      = np.ones((Nx, Ny, Nr), data_type) * fi_0               # Функция пор по размерам, [-]
        self.h_sloy  = np.full((Nx, Ny, Nr), init_h_sloy, data_type)  # Толщина осадочного слоя парафина, [м]
        self.qp      = np.full((Nx, Ny), init_qp, data_type) # Скорость отложения парафина в общем объеме, [1/сек]
        self.grad_p  = np.zeros((Nx, Ny), data_type)         # Градиент давления, [Па/м]
        self._Um_r2  = np.zeros((Nx, Ny), data_type)         # Компонент скорости фильтрации в капилляре радиуса r, [1/(м·сек)]
        self.Ur      = np.zeros((Nx, Ny, Nr), data_type)     # Скорость сужения капилляров, [м/сек]
        self.Ub      = np.zeros((Nx, Ny, Nr), data_type)     # Скорость блокирования капилляров, [1/сек]
        # Поля данный нового временного слоя
        self.new_h   = np.full((Nx, Ny, Nr), init_h_sloy, data_type)  # Толщина осадочного слоя парафина на новом временном слое, [м]
        self.new_Ur  = np.zeros((Nx, Ny, Nr), data_type)  # Скорость сужения капилляров на новом временном слое, [м/сек]
        self.new_Ub  = np.zeros((Nx, Ny, Nr), data_type)  # Скорость блокирования капилляров на новом временном слое, [1/сек]
        self.new_fi  = np.ones((Nx, Ny, Nr), data_type) * fi_0   # Функция пор по размерам на новом временном слое, [-]
        self.new_s   = np.zeros((Nx, Ny), data_type)      # Водонасыщенность на новом временном слое, [-]
        self.new_t   = np.zeros((Nx, Ny), data_type)      # Температура на новом временном слое, [С]
        self.new_wps = np.full((Nx, Ny), init_Wps, data_type)  # Массовая доля взвешенного парафина на новом временном слое, [-]
        self.new_wp  = np.full((Nx, Ny), init_Wp, data_type)   # Массовая доля растворенного парафина на новом временном слое, [-]
        self.new_qp  = np.full((Nx, Ny), init_qp, data_type)   # Скорость отложения парафина на новом временном слое, [1/сек]
        self.new_k   = np.full((Nx, Ny), init_k, data_type)    # Пористость на новом временном слое, [м^2]
        self.new_m   = np.full((Nx, Ny), init_m, data_type)    # Проницаемость на новом временном слое, [-]
        # Временные массивы перетоков через границы ячеек
        self.cells_T_eq  = np.zeros((Nx, Ny), data_type)  # Суммарный переток температуры в ячейке
        self.cells_Wp_eq = np.zeros((Nx, Ny), data_type)  # Суммарный переток растворенного в ячейке
        self.cells_S_eq  = np.zeros((Nx, Ny), data_type)  # Суммарный переток водонасыщенности в ячейке
        # Вспомогательные поля класса
        self._t = 0.0
        self._i_img = 0
        self._paraphin = not np.isclose(init_Wp + init_Wps, 0.0)
        self.a_tdma = np.zeros(Nr, data_type)
        self.b_tdma = np.zeros(Nr, data_type)
        self.rhs = np.zeros(N, data_type)
        self.data = np.zeros(NN, data_type)
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
                self.wells = self.n_wells * [WellStruct]
            else:
                self.wells[1].eta = 0.0

            self.sort_mask, self.rows_indices, self.cols_ptr, self.wells = (
                preprocess_matrix_and_wells(self.wells, self._wells_buffer))

        @njit
        def _calc_integrals():
            """Вычисление интегралов от функций r^4*fi_o(r) и r^2*fi_o(r)"""
            r2_fi0, r4_fi0 = 0.0, 0.0
            for i in range(1, Nr):
                dr = r1[i] - r1[i - 1]
                a = (fi_0[i - 1] * r1[i] - fi_0[i] * r1[i - 1]) / dr
                b = (fi_0[i] - fi_0[i - 1]) / dr
                r2_fi0 += (r3[i] - r3[i-1]) * a / 3 + (r4[i] - r4[i-1]) * b / 4  # r^2 * fi
                r4_fi0 += (r5[i] - r5[i-1]) * a / 5 + (r6[i] - r6[i-1]) * b / 6  # r^4 * fi

            return r2_fi0, r4_fi0

        self.integr_r2_fi0, self.integr_r4_fi0 = _calc_integrals()
        _wells_and_matrix_processing()


    def add_bc(self, field: DataField, bound: Bound, type_bc: TypeBC, value: float) -> None:
        """Учет граничных условий для полей данных."""
        add_bc(self.boundary_conditions, bound.value, field.value, type_bc.value, value)


    def add_well(self, name: str, i: int, j: int, p: float, is_injector: bool = False, T: float = 0.0,
                 rw: float = rw, mult: float = 1.0) -> None:
        """Добавление скважин в расчет."""
        well = WellStruct(i=i, j=j, p=p, T=T, rw=rw, is_injector=int(is_injector), mult=mult)
        self._wells_buffer.append({'well': well, 'name': name})
        self.n_wells += 1


    def start(self) -> None:
        try:
            times = np.linspace(0, Time_end, int(Time_end / dt + 1))
            tt = perf_counter()
            self.initialize()        # Задание начальных условий из файла const.py
            self.upd_time_step(0.0)  # При первом запуске компилируются модули
            print('Время компиляции:', perf_counter() - tt)

            with tqdm(iterable=times[1:], ncols=90, desc='Решение задачи', file=stdout, smoothing=0.05,
                      bar_format="{l_bar}{bar}[{elapsed}/{remaining}]  {n_fmt}/{total_fmt}{postfix}   ") as pbar:
                for _t in pbar:
                    self.upd_time_step(_t)
                    pbar.set_postfix(день=_t / day_to_sec)
                    if self.wells[1].eta >= max_eta:
                        break
        except (KeyboardInterrupt, SystemError):
            pass
        finally:
            print('KIN:', round(self.KIN, 5))
            print('eta:', round(self.wells[1].eta, 5))
            convert_pkl_files()
            rmtree(results_path)


    def upd_time_step(self, t: float) -> None:
        """Решение задачи на текущем временном слое."""
        self._t = t
        data = _process_time_step(self._t, self._paraphin, self.boundary_conditions, self.grad_p, self._Um_r2, self.qp, self.new_qp, self.k, self.new_k, self.m, self.m_0, self.new_m, self.S, self.S_0, self.new_s, self.Wo, self.Wo_0, self.Wp, self.Wp_0, self.new_wp, self.Wps, self.Wps_0, self.new_wps, self.T, self.T_0, self.new_t,
                       self.fi, self.new_fi, self.h_sloy, self.new_h, self.Ur, self.new_Ur, self.Ub, self.new_Ub, self.integr_r2_fi0, self.integr_r4_fi0, self.a_tdma, self.b_tdma, self.C_o, self.C_w, self.C_p, self.C_f, self.C_ff, self.cells_T_eq, self.cells_Wp_eq, self.cells_S_eq, self.mu_o, self.mu_w,
                       self.n_wells, self.wells, self.rows_indices, self.cols_ptr, self.sort_mask, self.data, self.rhs)
        self.p, self.KIN, self.new_t, self.new_s, self.new_m, self.new_k, self.new_h, self.new_Ur, self.new_Ub, self.new_fi, self.new_wps, self.new_wp, self.new_qp = data

        _swap_time_steps(self._t, self._paraphin, self.Wps_dep, self._Um_r2, self.qp, self.new_qp, self.k, self.new_k, self.m, self.m_0, self.new_m, self.S, self.S_0, self.new_s, self.Wo, self.Wo_0, self.Wp, self.Wp_0,
                     self.new_wp, self.Wps, self.Wps_0, self.new_wps, self.T, self.T_0, self.new_t, self.fi, self.new_fi, self.h_sloy, self.new_h, self.Ur, self.new_Ur, self.Ub, self.new_Ub, self.mu_o, self.mu_w, self.C_w, self.C_o, self.C_f, self.C_p)

        _logging_solution(self, t)

        # Запись данных в файл
        if t >= self._i_img * sol_time_step or np.isclose(t, Time_end) or self.wells[1].eta >= max_eta:
            # TODO создать массив в который кешируются данные (5-7 врем слоев) и записывать в файл только при заполнении кеша !!!!!
            save_fields(self, t)
            self._i_img += 1


@njit
def _process_time_step(_t, _paraphin, boundary_conditions, grad_p, _Um_r2, qp, new_qp, k, new_k, m, m_0, new_m, S, S_0, new_s, Wo, Wo_0, Wp, Wp_0, new_wp, Wps, Wps_0, new_wps, T, T_0, new_t,
                       fi, new_fi, h_sloy, new_h, Ur, new_Ur, Ub, new_Ub, integr_r2_fi0, integr_r4_fi0, a_tdma, b_tdma, C_o, C_w, C_p, C_f, C_ff, cells_T_eq, cells_Wp_eq, cells_S_eq, mu_o, mu_w,
                       n_wells, wells, rows_indices, cols_ptr, sort_mask, data, rhs):
    """Метод IMPES: явный по насыщенности неявный по давлению."""
    # Обновление давления
    p = calc_pressure(Wo, Wo_0, m, m_0, k, S, S_0, mu_o, mu_w, wells, rows_indices, cols_ptr, sort_mask, data, rhs, boundary_conditions)
    # Обновление данных скважин
    KIN = _update_wells_data(n_wells, wells, p, S, k, mu_o, mu_w)
    # Учет скважин в уравнениях
    _wells_loop(n_wells, wells, m, new_m, S, new_s, T, new_t, Wp, new_wp, Wps, C_o, C_w, C_f, C_p)
    # Решение уравнений по явной схеме
    _equations_loop(_t, _paraphin, boundary_conditions, p, grad_p, _Um_r2, qp, new_qp, k, new_k, m, m_0, new_m, S, S_0, new_s, Wo, Wp, Wp_0, new_wp, Wps, Wps_0, new_wps, T, T_0, new_t,
                fi, new_fi, h_sloy, new_h, Ur, new_Ur, Ub, new_Ub, integr_r2_fi0, integr_r4_fi0, a_tdma, b_tdma, C_o, C_w, C_p, C_f, C_ff, cells_T_eq, cells_Wp_eq, cells_S_eq, mu_o, mu_w)

    return p, KIN, new_t, new_s, new_m, new_k, new_h, new_Ur, new_Ub, new_fi, new_wps, new_wp, new_qp


@njit()  # nogil=True, parallel=True, boundscheck=False, fastmath=True
def _equations_loop(_t, _paraphin, boundary_conditions, p, grad_p, _Um_r2, qp, new_qp, k, new_k, m, m_0, new_m, S, S_0, new_s, Wo, Wp, Wp_0, new_wp, Wps, Wps_0, new_wps, T, T_0, new_t,
                    fi, new_fi, h_sloy, new_h, Ur, new_Ur, Ub, new_Ub, integr_r2_fi0, integr_r4_fi0, a_tdma, b_tdma, C_o, C_w, C_p, C_f, C_ff, cells_T_eq, cells_Wp_eq, cells_S_eq, mu_o, mu_w):
    """Решение уравнений по явной схеме в цикле по ячейкам."""
    for i in prange(Nx):
        for j in range(Ny):
            # ---решение задачи кольматации\суффозии---
            if _paraphin:
                # Средняя скорость в капилляре * r^2
                calc_Um_r2(i, j, p, grad_p, _Um_r2, mu_o)
                # Обновление концентраций парафина
                wps_wp_equation(i, j, qp, m, m_0, S, S_0, Wo, Wp, Wp_0, Wps, Wps_0, T, T_0, cells_Wp_eq, new_wp, new_wps)
                # Обновление толщины осадочного слоя, скорости изменения радиуса капилляра и скорости блокирования капилляров
                calc_velocitys_h(i, j, S, _Um_r2, Wps, mu_o, fi, h_sloy, Ur, new_h, new_Ur, new_Ub)
                # Обновление функции пор по размерам, объема выделяемого парафина, пористости, проницаемости
                calc_qp_m_k_fi(i, j, Wps, m, fi, Ur, Ub, integr_r2_fi0, integr_r4_fi0, a_tdma, b_tdma, new_qp, new_fi, new_k, new_m)

            # ---решение гидродинамики---
            flows_in_cells(i, j, boundary_conditions, p, S, T, k, mu_o, mu_w, m, Wp, Wps, C_o, C_w, C_p, cells_T_eq, cells_Wp_eq, cells_S_eq)
            saturation_equation(i, j, S, m, m_0, cells_S_eq, new_m, new_s)
            temperature_equation(i, j, T, m, S, C_o, C_w, C_f, C_ff, C_p, Wps, qp, cells_T_eq, _t, k, mu_o, mu_w, grad_p, new_t, new_m, new_s)


@njit
def _wells_loop(n_wells, wells, m, new_m, S, new_s, T, new_t, Wp, new_wp, Wps, C_o, C_w, C_f, C_p):
    """Учет скважин в решении уравнений по явной схеме."""
    for i in range(n_wells):
        saturation_well(wells[i], m, new_m, new_s)
        wps_wp_wells(wells[i], m, S, T, Wp, Wps, new_wp)
        temperature_well(wells[i], T, m, S, C_o, C_w, C_f, C_p, Wps, new_t)


@njit
def _update_wells_data(n_wells, wells, p, S, k, mu_o, mu_w):
    """Обновление дебита и обводненности скважин."""
    Q_oil = 0.0
    for i in range(n_wells):
        wells[i] = upd_q_and_eta(wells[i], p, S, k, mu_o, mu_w)
        if wells[i].is_injector == 0:
            Q_oil += wells[1].Q[0]

    # Вычисление КИН
    return Q_oil / geological_reserves


@njit()  # nogil=True, parallel=True
def _swap_time_steps(_t, _paraphin, Wps_dep, _Um_r2, qp, new_qp, k, new_k, m, m_0, new_m, S, S_0, new_s, Wo, Wo_0, Wp, Wp_0,
                     new_wp, Wps, Wps_0, new_wps, T, T_0, new_t, fi, new_fi, h_sloy, new_h, Ur, new_Ur, Ub, new_Ub, mu_o, mu_w, C_w, C_o, C_f, C_p):
    """Обновление полей данных на новом временном слое."""
    for i in prange(Nx):
        for j in range(Ny):
            # пересчет свойств флюидов из-за изменения температуры
            _update_mu_and_c_temp(i, j, new_t, mu_o, mu_w, C_w, C_o, C_f, C_p)

            S_0[i, j] = S[i, j]
            S[i, j]   = new_s[i, j]
            new_s[i, j] = 0.0
            T_0[i, j] = T[i, j]
            T[i, j]   = new_t[i, j]
            new_t[i, j] = 0.0

            if _paraphin:
                Wo_0[i, j]  = Wo[i, j]
                Wo[i, j]    = 1.0 - new_wp[i, j] - new_wps[i, j]
                Wp_0[i, j]  = Wp[i, j]
                Wp[i, j]    = new_wp[i, j]
                new_wp[i, j] = 0.0
                Wps_0[i, j] = Wps[i, j]
                Wps[i, j]   = new_wps[i, j]
                k[i, j]     = new_k[i, j]
                m_0[i, j]   = m[i, j]
                m[i, j]     = new_m[i, j]
                Wps_dep[i, j] = min(Wps_dep[i, j] - qp[i, j] * dt * ro_p /
                                            ((1.0-Wps[i,j]) * ro_o + Wps[i,j] * ro_p), init_Wp)
                qp[i, j]    = new_qp[i, j]
                for ij in range(Nr):
                    fi[i, j, ij]     = new_fi[i, j, ij]
                    h_sloy[i, j, ij] = new_h[i, j, ij]
                    Ur[i, j, ij]     = new_Ur[i, j, ij]
                    Ub[i, j, ij]     = new_Ub[i, j, ij]


@njit
def _update_mu_and_c_temp(i, j, T, mu_o, mu_w, C_w, C_o, C_f, C_p):
    """Обновление свойств флюидов, вызванных изменением температуры."""
    mu_o[i, j] = calc_mu_o(T[i, j])
    mu_w[i, j] = calc_mu_w(T[i, j])
    C_w[i, j]  = c_w  # calc_c_w(self.T[i, j])
    C_o[i, j]  = c_o  # calc_c_o(self.T[i, j])
    C_f[i, j]  = c_f  # calc_c_f(self.T[i, j])
    C_p[i, j]  = c_p  # calc_c_p(self.T[i, j])


def _logging_solution(solver, t):
    """Логирование решения задачи."""
    if not LOGGING:
        return None

    solver.logger.info('')
    _logging_resources(solver.logger)
    solver.logger.info(f"ВРЕМЕННОЙ СЛОЙ t = {round(t / day_to_sec, 5)} день ({int(t / dt)} итерация)")
    solver.logger.info(f"Обновлено давление (бар): min={solver.p.min() / bar_to_pa}  max={solver.p.max() / bar_to_pa}")
    solver.logger.info(f"Обновлена насыщенность:   min={solver.new_s.min()}  max={solver.new_s.max()}")
    solver.logger.info(f"Обновлена температура:    min={solver.new_t.min()}  max={solver.new_t.max()}")
    for i in range(solver.n_wells):
        solver.logger.info(f"Дебет скважины {solver._wells_buffer[i]['name']} (м^3/сут): q_o={solver.wells[i].q[0] * day_to_sec}  q_w={solver.wells[i].q[1] * day_to_sec}")

    if not solver._paraphin:
        return None
    solver.logger.info(f"Wps:  min={solver.new_wps.min()}  max={solver.new_wps.max()}")
    solver.logger.info(f"Wp:   min={solver.new_wp.min()}  max={solver.new_wp.max()}")
    oil_components = solver.new_wp + solver.new_wps + solver.Wo
    solver.logger.info(f"Wp+Wps+Wo:   min={oil_components.min()}  max={oil_components.max()}")

    solver.logger.info(f"qp:     min={solver.new_qp.min()}  max={solver.new_qp.max()}")
    solver.logger.info(f"m_mult: min={solver.new_m.min()}  max={solver.new_m.max()}")
    solver.logger.info(f"k_mult: min={solver.new_k.min()}  max={solver.new_k.max()}")

    solver.logger.info(f"fi:   min={solver.fi.min()}  max={solver.fi.max()}")
    solver.logger.info(f"Ur:   min={solver.new_Ur.min()}  max={solver.new_Ur.max()}")
    solver.logger.info(f"Ub:   min={solver.new_Ub.min()}  max={solver.new_Ub.max()}")
    solver.logger.info(f"Um:   min={solver._Um_r2.min()*1e-12}  max={solver._Um_r2.max()*1e-12}")

    # self.logger.info(f"fi:   {' '.join([f'{x:.{3}f}' for x in self.fi)[0, 0]])}")
    # self.logger.info(f"fi_0: {' '.join([f'{x:.{3}f}' for x in fi_0])}")


def _logging_resources(logger) -> None:
    # cpu_usage = psutil.cpu_percent(interval=None)  # , percpu=True
    memory_info = psutil.virtual_memory()
    memory_usage = round(memory_info.used / memory_info.total * 100 , 5)  # memory_info.percent
    # self.logger.info(f'Использование CPU: {cpu_usage}%')
    logger.info(f'Использование памяти: {memory_usage}%')
