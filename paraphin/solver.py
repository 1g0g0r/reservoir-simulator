"""Класс содержит алгоритм расчета и хранение данных."""
from logging import INFO, getLogger, Formatter, FileHandler
from shutil import rmtree
from sys import stdout
from time import perf_counter

import numpy as np
import psutil
from numba import njit, prange
from tqdm import tqdm

from paraphin import N, r1, r3, r4, r5, r6, fi_0
from .constants import (data_type, Nx, Ny, Nr, rw, results_path, data_path, logs_path, init_T, init_k, init_S, init_m,
                        init_p, init_qp, init_h_sloy, init_Wp, init_Wps, bar_to_pa, dt, day_to_sec,
                        max_eta, c_o, c_w, c_p, c_f, c_ff, sol_time_step, Time_end, LOGGING, geological_reserves,
                        volume, S_min, S_max, CFL_target, dt_growth, dt_max, dt_min,
                        min_Wps_bound)
from .equations import (calc_qp_m_k_fi, calc_pressure, saturation_equation, saturation_well, temperature_well,
                        temperature_equation, wps_wp_equation, wps_wp_wells, calc_velocities_h, flows_in_cells,
                        calc_Um_r2)
from .utils import (calc_mu_o, calc_mu_w, preprocess_wells, convert_pkl_files, save_fields,
                    Bound, TypeBC, DataField, add_bc, WellStruct, upd_q_and_eta, Buckley_Leverett,
                    calc_mobility)

_process = psutil.Process()  # для логирования памяти, см. _logging_resources


class Solver:
    def __init__(self):
        # Граничные условия и скважины
        self.KIN = data_type(0.0)
        self.n_wells = 0
        self._wells_buffer = []
        self._wells_names = []
        self.wells = []  # заполняется в initialize() из _wells_buffer
        self.boundary_conditions = np.zeros(dtype=data_type, shape=(4, 3, 2))  # Граница -> Поле -> Тип, Значение
        # Свойства флюидов
        self.mu_o = np.full((Nx, Ny), calc_mu_o(init_T), data_type)  # Вязкость нефти, [Па*с]
        self.mu_w = np.full((Nx, Ny), calc_mu_w(init_T), data_type)  # Вязкость воды, [Па*с]
        self.C_w  = np.full((Nx, Ny), c_w, data_type)  # Теплоемкость воды, [Дж*кг/C]
        self.C_o  = np.full((Nx, Ny), c_o, data_type)  # Теплоемкость нефти, [Дж*кг/C]
        self.C_f  = np.full((Nx, Ny), c_f, data_type)  # Теплоемкость пласта, [Дж*кг/C]
        self.C_ff = np.full((Nx, Ny), c_ff, data_type) # Теплоемкость окружающих пород пласта, [Дж*кг/C]
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
        self.cells_Q_out = np.zeros((Nx, Ny), data_type)  # Суммарный отток через грани ячейки, [м^3/с]
        # Вспомогательные поля класса
        self._t = 0.0
        self._i_img = 0
        self._layers_file = None  # общий файл слоев, открывается при первом сохранении
        self.dt = dt              # Текущий шаг по времени, подбирается по CFL каждую итерацию, [с]
        self.max_dfw = 1.0        # max|df_w/dS|, задается в initialize()
        self.clip_stats = np.zeros(4, data_type)  # [число обрезаний S, макс. выход, i, j] за шаг
        self.clip_field = np.zeros((Nx, Ny), data_type)  # выход S за границы по ячейкам за шаг
        self._clip_total = 0.0    # Накопленное число обрезаний S за весь расчет
        self._p_bhp_min = 0.0     # Диапазон забойных давлений скважин, [Па]
        self._p_bhp_max = 0.0
        self._p_warned = False
        self._paraphin = not np.isclose(init_Wp + init_Wps, 0.0)
        # Прогоночные коэффициенты для fi: своя строка на каждый i, иначе гонка в prange по ячейкам
        self.a_tdma = np.zeros((Nx, Nr), data_type)
        self.b_tdma = np.zeros((Nx, Nr), data_type)
        # Подвижности фаз: считаются один раз за шаг и переиспользуются сборкой матрицы и перетоками
        self.lam_o = np.zeros((Nx, Ny), data_type)
        self.lam_w = np.zeros((Nx, Ny), data_type)
        # Матрица давления в трех диагоналях (idx = i + j*Nx) и буферы ленточного решателя
        self.diag  = np.zeros(N, data_type)
        self.ex    = np.zeros(N, data_type)
        self.ey    = np.zeros(N, data_type)
        self.rhs   = np.zeros(N, data_type)
        self.band_w = np.zeros((N, Nx + 1), data_type)  # фактор Холецкого, живет между шагами
        self.p_vec = np.full(N, init_p, data_type)      # решение и начальное приближение для PCG
        self.pcg_r = np.zeros(N, data_type)
        self.pcg_z = np.zeros(N, data_type)
        self.pcg_p = np.zeros(N, data_type)
        self.pcg_q = np.zeros(N, data_type)
        self._band_age = 0  # 0 - фактора еще нет, считаем точно
        results_path.mkdir(parents=True, exist_ok=True)
        data_path.mkdir(parents=True, exist_ok=True)

        if LOGGING:
            self.logger = getLogger(__name__)
            self.logger.setLevel(INFO)
            self.logger.propagate = False
            for old in self.logger.handlers[:]:
                self.logger.removeHandler(old)
                old.close()
            handler = FileHandler(logs_path, mode='w')
            handler.setFormatter(Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
            self.logger.addHandler(handler)
        else:
            self.logger = getLogger(__name__)
            self.logger.disabled = True


    def initialize(self):
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

        # Критерий останова и расчет КИН читают wells[1], поэтому набор скважин фиксирован:
        # ровно две, добывающая под индексом 1. Проверяем здесь, а не падаем по IndexError в цикле.
        if self.n_wells != 2 or self._wells_buffer[1]['well'].is_injector == 1:
            raise ValueError('Ожидаются ровно две скважины, добывающая - вторая по порядку '
                             f'add_well; сейчас их {self.n_wells}')

        self.integr_r2_fi0, self.integr_r4_fi0 = _calc_integrals()
        self._wells_names = [item['name'] for item in self._wells_buffer]
        self.wells = preprocess_wells(self._wells_buffer)
        self.max_dfw = _calc_max_dfw()

        bhp = [well['well'].p for well in self._wells_buffer]
        self._p_bhp_min, self._p_bhp_max = min(bhp), max(bhp)


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
            tt = perf_counter()
            self.initialize()        # Задание начальных условий из файла const.py
            self.upd_time_step(0.0)  # При первом запуске компилируются модули
            print('Время компиляции:', perf_counter() - tt)

            _t = 0.0
            with tqdm(total=Time_end, ncols=90, desc='Решение задачи', file=stdout, smoothing=0.05,
                      bar_format="{l_bar}{bar}[{elapsed}/{remaining}]{postfix}   ") as pbar:  # {n_fmt}/{total_fmt}
                while _t < Time_end:
                    self.dt = min(self.dt, Time_end - _t)  # последний шаг подрезаем ровно до Time_end
                    _t += self.dt
                    pbar.update(self.dt)
                    self.upd_time_step(_t)
                    pbar.set_postfix(день=_t / day_to_sec, шаг_сут=round(self.dt / day_to_sec, 5))
                    if self.wells[1].eta >= max_eta:
                        break
        except (KeyboardInterrupt, SystemError):
            pass
        finally:
            print('KIN:', round(self.KIN, 5))
            print('eta:', round(self.wells[1].eta, 5))
            if self._layers_file is not None:
                self._layers_file.close()
                self._layers_file = None
            convert_pkl_files(self._i_img)
            rmtree(results_path)


    def upd_time_step(self, t: float) -> None:
        """Решение задачи на текущем временном слое."""
        self._t = t
        step_dt = self.dt
        self.clip_stats[:] = 0.0

        # Метод IMPES: явный по насыщенности, неявный по давлению.
        # Подвижности фаз - общие для сборки матрицы давления и для перетоков
        calc_mobility(self.k, self.S, self.mu_o, self.mu_w, self.lam_o, self.lam_w)
        # Обновление давления
        self._band_age = calc_pressure(self.Wo, self.Wo_0, self.m, self.m_0, self.k, self.S, self.S_0, self.mu_o, self.mu_w, self.lam_o, self.lam_w, self.wells,
                                       self.diag, self.ex, self.ey, self.rhs, self.band_w, self.p_vec,
                                       self.pcg_r, self.pcg_z, self.pcg_p, self.pcg_q,
                                       self.boundary_conditions, self._band_age, self.p, step_dt)
        # Обновление данных скважин
        self.KIN = _update_wells_data(self.n_wells, self.wells, self.p, self.S, self.k, self.mu_o, self.mu_w, step_dt)
        # Учет скважин в уравнениях
        _wells_loop(self.n_wells, self.wells, self.m, self.S, self.new_s, self.T, self.new_t, self.Wp, self.new_wp, self.Wps, self.C_o, self.C_w, self.C_f, self.C_p, step_dt)
        # Решение уравнений по явной схеме
        dt_cells = _equations_loop(self._t, self._paraphin, self.boundary_conditions, self.p, self.grad_p, self._Um_r2, self.qp, self.new_qp, self.k, self.new_k, self.m, self.m_0, self.new_m, self.S, self.S_0, self.new_s, self.Wp, self.new_wp, self.Wps, self.new_wps, self.Wps_dep, self.T, self.T_0, self.new_t,
                        self.fi, self.new_fi, self.h_sloy, self.new_h, self.Ur, self.new_Ur, self.Ub, self.new_Ub, self.integr_r2_fi0, self.integr_r4_fi0, self.a_tdma, self.b_tdma, self.C_o, self.C_w, self.C_p, self.C_f, self.C_ff, self.cells_T_eq, self.cells_Wp_eq, self.cells_S_eq, self.cells_Q_out, self.mu_o, self.mu_w, self.lam_o, self.lam_w, self.clip_field, self.clip_stats, self.max_dfw, step_dt)
        # Шаг для следующей итерации из фактического условия устойчивости
        dt_next = _calc_dt(self.n_wells, self.wells, self.m, self.cells_Q_out, self.max_dfw, step_dt, dt_cells)

        self._clip_total += self.clip_stats[0]
        self._check_pressure()
        if self.clip_stats[0] > 0:
            _logging_clip(self)

        # Полный лог и сохранение полей идут по одному условию и обязательно до _swap_time_steps:
        # он обнуляет new_* поля, которые логируются.
        dump_now = (t >= self._i_img * sol_time_step or np.isclose(t, Time_end)
                    or self.wells[1].eta >= max_eta)
        if dump_now:
            _logging_solution(self, t)

        _swap_time_steps(self)

        self.dt = dt_next

        # Запись данных в файл
        if dump_now:
            save_fields(self, t)
            self._i_img += 1


    def _check_pressure(self) -> None:
        """Проверка поля давления на физичность.

        При отключенном парафине правая часть уравнения давления тождественно нулевая, матрица -
        симметричная M-матрица, а скважины входят источниками с забойными давлениями. Тогда дискретный
        принцип максимума гарантирует p в пределах [min(p_заб), max(p_заб)]. Выход за эти границы
        означает не физику, а NaN в матрице или сбой решателя, поэтому сообщаем о нем явно.
        """
        if not np.isfinite(self.p.sum()):
            raise FloatingPointError('В поле давления появились NaN/Inf')

        if self._paraphin or self.n_wells == 0:
            return None

        tol = 1e-6 * max(abs(self._p_bhp_max), 1.0)
        if self.p.min() < self._p_bhp_min - tol or self.p.max() > self._p_bhp_max + tol:
            msg = (f'Давление вне диапазона забойных давлений скважин: '
                   f'min={self.p.min() / bar_to_pa:.4f} max={self.p.max() / bar_to_pa:.4f} бар '
                   f'при допустимых [{self._p_bhp_min / bar_to_pa:.4f}, {self._p_bhp_max / bar_to_pa:.4f}] бар')
            self.logger.warning(msg)
            if not self._p_warned:
                self._p_warned = True
                print('\n' + msg)


@njit(parallel=True, cache=True)
def _equations_loop(_t, _paraphin, boundary_conditions, p, grad_p, _Um_r2, qp, new_qp, k, new_k, m, m_0, new_m, S, S_0, new_s, Wp, new_wp, Wps, new_wps, Wps_dep, T, T_0, new_t,
                    fi, new_fi, h_sloy, new_h, Ur, new_Ur, Ub, new_Ub, integr_r2_fi0, integr_r4_fi0, a_tdma, b_tdma, C_o, C_w, C_p, C_f, C_ff, cells_T_eq, cells_Wp_eq, cells_S_eq, cells_Q_out, mu_o, mu_w, lam_o, lam_w, clip_field, clip_stats, max_dfw, dt):
    """Решение уравнений по явной схеме в цикле по ячейкам.

    Ячейки независимы - каждая пишет только в свои [i, j], поэтому внешний цикл идет в prange.
    Статистика обрезаний насыщенности собирается поячеечно в clip_field и сворачивается уже после
    параллельного цикла. Прогоночные буферы a_tdma, b_tdma нарезаются по i: один общий буфер на
    все ячейки давал бы гонку - потоки затирали друг другу коэффициенты, и fi считалась по мусору.
    """
    for i in prange(Nx):
        for j in range(Ny):
            calc_Um_r2(i, j, p, grad_p, _Um_r2, mu_o)  # Средняя скорость в капилляре * r^2

            # ---решение задачи кольматации\суффозии---
            if _paraphin:
                # Обновление концентраций парафина
                wps_wp_equation(i, j, qp, m, m_0, S, S_0, Wp, Wps, T, T_0, cells_Wp_eq, new_wp, new_wps, Wps_dep, dt)
                # Обновление толщины осадочного слоя, скорости изменения радиуса капилляра и скорости блокирования капилляров
                calc_velocities_h(i, j, S, _Um_r2, Wps, mu_o, fi, h_sloy, Ur, new_h, new_Ur, new_Ub, dt)
                # Обновление функции пор по размерам, объема выделяемого парафина, пористости, проницаемости
                calc_qp_m_k_fi(i, j, Wps, qp, m, k, fi, Ur, Ub, integr_r2_fi0, integr_r4_fi0, a_tdma[i], b_tdma[i], new_qp, new_fi, new_k, new_m, dt)

            # ---решение гидродинамики---
            flows_in_cells(i, j, boundary_conditions, p, S, T, k, mu_o, mu_w, lam_o, lam_w, m, Wp, Wps, C_o, C_w, C_p, cells_T_eq, cells_Wp_eq, cells_S_eq, cells_Q_out)
            saturation_equation(i, j, S, m, cells_S_eq, new_m, new_s, dt, clip_field)
            temperature_equation(i, j, T, m, S, C_o, C_w, C_f, C_ff, C_p, Wps, qp, cells_T_eq, _t, lam_o, lam_w, grad_p, new_t, new_m, new_s, dt)

    # Свертка по ячейкам, одним проходом: статистика обрезаний [число, макс. выход, i, j] и
    # ограничение на шаг по числу Куранта. Раньше это были два прохода по одним и тем же 2500
    # ячейкам, причем второй - отдельным njit-вызовом со скважинами в аргументах.
    # Свои имена ci, cj: переиспользование i, j из prange сбивает numba типизацию индексов.
    cnt, worst, wi, wj = 0.0, 0.0, 0.0, 0.0
    dt_cells = dt_max
    for ci in range(Nx):
        for cj in range(Ny):
            v = clip_field[ci, cj]
            if v > 0.0:
                cnt += 1.0
                if v > worst:
                    worst = v
                    wi = ci
                    wj = cj

            q_out = cells_Q_out[ci, cj]
            if q_out > 1e-30:
                dt_cell = CFL_target * m[ci, cj] * volume / (max_dfw * q_out)
                if dt_cell < dt_cells:
                    dt_cells = dt_cell
    clip_stats[0] = cnt
    clip_stats[1] = worst
    clip_stats[2] = wi
    clip_stats[3] = wj

    return dt_cells


@njit
def _calc_dt(n_wells, wells, m, cells_Q_out, max_dfw, dt_prev, dt_cells):
    """Шаг по времени из условия устойчивости явной схемы по насыщенности.

    Ограничение на ячейку: dt <= CFL * m*V / (max|df_w/dS| * Q_отток). По самим ячейкам минимум
    уже посчитан в `_equations_loop` (там же идет свертка обрезаний, проход по сетке один на двоих)
    и приходит сюда как `dt_cells`. Здесь остаются только скважины: их отбор идет мимо граней,
    поэтому для ячеек скважин он добавляется к оттоку отдельно. Считается каждый шаг, потому что
    расход растет вместе с суммарной подвижностью по мере обводнения - в разы к концу расчета.
    """
    dt_new = dt_cells
    for w in range(n_wells):
        i, j = wells[w].i, wells[w].j
        q_out = cells_Q_out[i, j] + abs(wells[w].q[2])
        if q_out > 1e-30:
            dt_cell = CFL_target * m[i, j] * volume / (max_dfw * q_out)
            if dt_cell < dt_new:
                dt_new = dt_cell

    return min(max(dt_new, dt_min), dt_prev * dt_growth, dt_max)


@njit
def _wells_loop(n_wells, wells, m, S, new_s, T, new_t, Wp, new_wp, Wps, C_o, C_w, C_f, C_p, dt):
    """Учет скважин в решении уравнений по явной схеме."""
    for i in range(n_wells):
        saturation_well(wells[i], m, new_s, dt)
        wps_wp_wells(wells[i], m, S, T, Wp, Wps, new_wp, dt)
        temperature_well(wells[i], T, m, S, C_o, C_w, C_f, C_p, Wps, new_t, dt)


@njit
def _update_wells_data(n_wells, wells, p, S, k, mu_o, mu_w, dt):
    """Обновление дебита и обводненности скважин."""
    Q_oil = 0.0
    for i in range(n_wells):
        wells[i] = upd_q_and_eta(wells[i], p, S, k, mu_o, mu_w, dt)
        if wells[i].is_injector == 0:
            Q_oil += wells[i].Q[0]

    # Вычисление КИН
    return Q_oil / geological_reserves


def _swap_time_steps(solver) -> None:
    """Обновление полей данных на новом временном слое.

    Копирования полей здесь нет: буферы меняются ссылками (ping-pong). Поле с тремя буферами
    (X_0, X, new_X) прокручивается по кругу - new_X становится текущим слоем, текущий уезжает
    в X_0, а освободившийся X_0 уходит под new_X, его содержимое все равно затирается целиком.
    Данные при этом не двигаются, двигаются только имена.

    **Условие корректности: уравнение обязано писать new_X в каждой ячейке, а не только там,
    где поле изменилось.** Пропущенная ячейка получит не текущее значение, а позапрошлое: ее
    держит второй буфер, который к следующему шагу встанет на место текущего слоя. Раньше на это
    работало копирование в цикле, и ячейку можно было не трогать; теперь `calc_qp_m_k_fi` и
    `wps_wp_equation` в своих else-ветках пишут `new_* = текущее`. Заводишь новое поле - или
    пиши его в каждой ячейке, или оставляй копию, как у полей по радиусам пор (`fi`, `h_sloy`,
    `Ur`, `Ub`): те копируются в `_finish_swap` только выше порога кольматации.
    """
    s = solver
    s.S_0, s.S, s.new_s = s.S, s.new_s, s.S_0
    s.T_0, s.T, s.new_t = s.T, s.new_t, s.T_0

    if s._paraphin:
        s.Wo_0, s.Wo = s.Wo, s.Wo_0
        s.Wp_0, s.Wp, s.new_wp = s.Wp, s.new_wp, s.Wp_0
        s.Wps_0, s.Wps, s.new_wps = s.Wps, s.new_wps, s.Wps_0
        s.m_0, s.m, s.new_m = s.m, s.new_m, s.m_0
        s.k, s.new_k = s.new_k, s.k
        s.qp, s.new_qp = s.new_qp, s.qp

    _finish_swap(s._paraphin, s.T, s.mu_o, s.mu_w, s.new_s, s.new_t, s.Wo, s.Wp, s.Wps, s.new_wp,
                 s.Wps_0, s.fi, s.new_fi, s.h_sloy, s.new_h, s.Ur, s.new_Ur, s.Ub, s.new_Ub)


@njit(parallel=True, cache=True)
def _finish_swap(_paraphin, T, mu_o, mu_w, new_s, new_t, Wo, Wp, Wps, new_wp,
                 Wps_0, fi, new_fi, h_sloy, new_h, Ur, new_Ur, Ub, new_Ub):
    """Поячеечный хвост обмена слоев: то, что ссылками не решается.

    Имена полей здесь уже после ping-pong: `Wp` - это новый слой, `new_wp` - отработавший буфер
    под обнуление (уравнения накапливают new_* через `+=`).

    Вязкости пересчитываются от новой температуры. Векторной записи тут нет намеренно:
    параллельный цикл считает их за 25 мкс против 60-99 мкс у numpy на 50x50 - `mu_w` идет
    через `10**x`, а SIMD-реализации `pow` у numpy нет. Теплоемкости C_w, C_o, C_f, C_p раньше
    переприсваивались здесь же константами на каждом шаге: они выставляются в `Solver.__init__`
    и не меняются, а корреляции от температуры лежат в `utils/math_utils/fluids_correlations.py`.

    Поля по радиусам пор - единственные, что здесь копируются: ссылками их не обменять, потому что
    `calc_velocities_h` и `calc_qp_m_k_fi` пишут их только выше порога кольматации, а тождественная
    запись в пропущенных ячейках стоила бы Nr элементов вместо одного. В этих ячейках new_* равны
    текущим с прошлого активного шага, поэтому копия под порогом - no-op и пропускается. Порог
    проверяется по Wps_0: сам Wps к этому моменту уже заменен новым, а уравнения смотрели на старый.
    Срезами, а не поячеечным циклом по Nr: numba разворачивает их в memcpy подряд лежащих Nr
    элементов, поячеечный цикл - в четыре чередующихся потока записи.
    """
    for i in prange(Nx):
        for j in range(Ny):
            mu_o[i, j] = calc_mu_o(T[i, j])
            mu_w[i, j] = calc_mu_w(T[i, j])
            new_s[i, j] = 0.0
            new_t[i, j] = 0.0

            if _paraphin:
                Wo[i, j]  = 1.0 - Wp[i, j] - Wps[i, j]
                new_wp[i, j] = 0.0

                if Wps_0[i, j] > min_Wps_bound:
                    fi[i, j, :]     = new_fi[i, j, :]
                    h_sloy[i, j, :] = new_h[i, j, :]
                    Ur[i, j, :]     = new_Ur[i, j, :]
                    Ub[i, j, :]     = new_Ub[i, j, :]


def _calc_max_dfw() -> float:
    """Максимум производной функции Баклея-Леверетта на рабочем диапазоне насыщенности.

    Именно эта величина задает предел устойчивости явной схемы по насыщенности. Считается по
    фактическим вязкостям, а не по номинальным mu_o, mu_w из constants.py: при 70 C отношение
    подвижностей равно 14.4 против заявленных там 5, а max|df_w/dS| - 6.5 против 2.2.
    """
    # ponytail: производная берется при init_T; пересчитать, если диапазон температур расширится
    s = np.linspace(S_min, S_max, 2001)
    mu_w_ref, mu_o_ref = calc_mu_w(init_T), calc_mu_o(init_T)
    f_w = np.array([Buckley_Leverett(x, mu_w_ref, mu_o_ref) for x in s])

    return float(np.abs(np.gradient(f_w, s)).max())


def _logging_solution(solver, t):
    """Логирование решения задачи."""
    if not LOGGING:
        return None

    solver.logger.info('')
    _logging_resources(solver.logger)
    solver.logger.info(f"ВРЕМЕННОЙ СЛОЙ t = {round(t / day_to_sec, 5)} день, шаг {solver.dt / day_to_sec} сут")
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


def _logging_clip(solver) -> None:
    """Предупреждение об обрезании насыщенности. Пишется на каждом шаге, где оно случилось."""
    if not LOGGING:
        return None
    solver.logger.warning(f"Насыщенность обрезана в {int(solver.clip_stats[0])} ячейках, "
                          f"максимальный выход {solver.clip_stats[1]} "
                          f"в ячейке ({int(solver.clip_stats[2])}, {int(solver.clip_stats[3])}); "
                          f"всего с начала расчета {int(solver._clip_total)}")


def _logging_resources(logger) -> None:
    # psutil.virtual_memory() на Windows стоит 3.5 мс - больше, чем весь шаг по времени.
    # Память самого процесса и полезнее, и дешевле в тысячу раз.
    logger.info(f'Память процесса: {round(_process.memory_info().rss / 1024 ** 2, 1)} МБ')
