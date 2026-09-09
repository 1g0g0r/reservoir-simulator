"""Класс содержит алгоритм расчета и хранение данных."""
from logging import INFO, getLogger, Formatter, FileHandler
from shutil import rmtree
from sys import stdout
from time import perf_counter

import numpy as np
from numba import njit, prange
from tqdm import tqdm

from paraphin import N, r1, r3, r4, r5, r6, fi_0
from .constants import (data_type, Nx, Ny, Nr, rw, results_path, data_path, logs_path, init_T, init_k, init_S, init_m,
                        init_p, init_qp, init_h_sloy, init_Wp, init_Wps, bar_to_pa, dt, day_to_sec,
                        max_eta, c_o, c_w, c_p, c_f, sol_time_step, Time_end, LOGGING, geological_reserves,
                        volume, S_min, S_max, CFL_target, dt_growth, dt_max, dt_min,
                        min_Wps_bound)
from .equations import (calc_qp_m_k_fi, calc_pressure, saturation_equation, temperature_source,
                        temperature_equation, wp_equation, calc_velocities_h, flows_in_cells,
                        calc_Um_r2)
from .utils import (calc_mu_o, calc_mu_w, preprocess_wells, convert_pkl_files, save_fields,
                    Bound, TypeBC, DataField, add_bc, WellStruct, upd_q_and_eta, Buckley_Leverett,
                    calc_mobility)


class Solver:
    def __init__(self):
        # Граничные условия и скважины
        self.KIN = data_type(0.0)
        self.n_wells = 0
        self._wells_buffer = []
        self._wells_names = []
        self.wells = []
        self.boundary_conditions = np.zeros(dtype=data_type, shape=(4, 3, 2))  # Граница -> Поле -> Тип, Значение
        # Свойства флюидов
        self.mu_o = np.full((Nx, Ny), calc_mu_o(init_T, init_Wps), data_type)  # Вязкость нефти, [Па*с]
        self.mu_w = np.full((Nx, Ny), calc_mu_w(init_T), data_type)  # Вязкость воды, [Па*с]
        self.C_w  = np.full((Nx, Ny), c_w, data_type)  # Теплоемкость воды, [Дж*кг/C]
        self.C_o  = np.full((Nx, Ny), c_o, data_type)  # Теплоемкость нефти, [Дж*кг/C]
        self.C_f  = np.full((Nx, Ny), c_f, data_type)  # Теплоемкость пласта, [Дж*кг/C]
        self.C_p  = np.full((Nx, Ny), c_p, data_type)  # Теплоемкость парафина, [Дж*кг/C]
        # Поля данных пласта
        self.p     = np.full((Nx, Ny), init_p, data_type)  # Давление, [Па]
        self.S     = np.full((Nx, Ny), init_S, data_type)  # Водонасыщенность, [-]
        self.T     = np.full((Nx, Ny), init_T, data_type)  # Температура, [С]
        self.T_0   = np.full((Nx, Ny), init_T, data_type)  # Температура на прошлом временном слое, [С]
        # Доли компонентов нефтяной фазы - объемные (w_o + w_p + w_ps = 1)
        self.Wo    = np.full((Nx, Ny), 1.0 - init_Wp - init_Wps, data_type)  # Объемная доля масляного компонента, [-]
        self.Wp    = np.full((Nx, Ny), init_Wp, data_type)   # Объемная доля растворенного парафина, [-]
        self.Wps   = np.full((Nx, Ny), init_Wps, data_type)  # Объемная доля взвешенного парафина, [-]
        self.k     = np.full((Nx, Ny), init_k, data_type)  # Проницаемость, [м^2]
        self.m     = np.full((Nx, Ny), init_m, data_type)  # Пористость, [-]
        # Динамика образования парафина (кольматация\суффозия)
        self.integr_r2_fi0 = data_type(0.0)
        self.integr_r4_fi0 = data_type(0.0)
        self.fi      = np.ones((Nx, Ny, Nr), data_type) * fi_0               # Функция пор по размерам, [-]
        self.h_sloy  = np.full((Nx, Ny, Nr), init_h_sloy, data_type)  # Толщина осадочного слоя парафина, [м]
        self.qp1     = np.full((Nx, Ny), init_qp, data_type) # Скорость осаждения парафина на стенках пор, [1/сек]
        self.qp2     = np.full((Nx, Ny), init_qp, data_type) # Скорость блокирования поровых каналов, [1/сек]
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
        self.new_qp1 = np.full((Nx, Ny), init_qp, data_type)   # Скорость осаждения на новом временном слое, [1/сек]
        self.new_qp2 = np.full((Nx, Ny), init_qp, data_type)   # Скорость блокирования на новом временном слое, [1/сек]
        self.new_k   = np.full((Nx, Ny), init_k, data_type)    # Пористость на новом временном слое, [м^2]
        self.new_m   = np.full((Nx, Ny), init_m, data_type)    # Проницаемость на новом временном слое, [-]
        # Временные массивы перетоков через границы ячеек
        self.cells_T_eq  = np.zeros((Nx, Ny), data_type)  # Суммарный переток температуры в ячейке
        self.cells_Wp_eq = np.zeros((Nx, Ny), data_type)  # Суммарный переток растворенного в ячейке
        self.cells_S_eq  = np.zeros((Nx, Ny), data_type)  # Суммарный переток водонасыщенности в ячейке
        self.cells_Q_out = np.zeros((Nx, Ny), data_type)  # Суммарный отток через грани ячейки, [м^3/с]
        # Источники скважин в тех же единицах, что и перетоки через грани. Скважины неподвижны,
        # поэтому обнулять буферы не нужно: ячейка со скважиной перезаписывается каждый шаг,
        # остальные так и остаются нулями.
        self.src_S  = np.zeros((Nx, Ny), data_type)  # Дебит по воде, [м^3/с]
        self.src_Wp = np.zeros((Nx, Ny), data_type)  # Вынос парафина нефтью, [м^3/с]
        self.src_T  = np.zeros((Nx, Ny), data_type)  # Приток энергии со скважиной, [Вт]
        # Состояние метода Винсома-Вестервельда: накопленный интеграл перегрева пород, [C*м].
        self.E_ff = np.zeros((Nx, Ny), data_type)
        # Вспомогательные поля класса
        self._t = 0.0
        self._i_img = 0
        self._layers_file = None  # общий файл слоев, открывается при первом сохранении
        self.dt = dt              # Текущий шаг по времени, подбирается по CFL каждую итерацию, [с]
        self.max_dfw = 1.0        # max|df_w/dS|, задается в initialize()
        self._producer = -1       # Индекс добывающей скважины, ищется в initialize() по is_injector
        self._paraphin = not np.isclose(init_Wp + init_Wps, 0.0)
        # Прогоночные коэффициенты для fi: своя строка на каждый i, иначе гонка в prange по ячейкам
        self.a_tdma = np.zeros((Nx, Nr), data_type)
        self.b_tdma = np.zeros((Nx, Nr), data_type)
        # Подвижности фаз: считаются один раз за шаг и переиспользуются сборкой матрицы и перетоками
        self.lam_o = np.zeros((Nx, Ny), data_type)
        self.lam_w = np.zeros((Nx, Ny), data_type)
        self.lam_h = np.zeros((Nx, Ny), data_type)  # эффективная теплопроводность ячейки, [Вт/(м*C)]
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

        # Критерий останова и расписание сохранения читают обводненность добывающей скважины,
        # поэтому она нужна ровно одна; порядок вызовов add_well при этом не важен.
        producers = [idx for idx, item in enumerate(self._wells_buffer)
                     if item['well'].is_injector == 0]
        if len(producers) != 1:
            raise ValueError('Ожидается ровно одна добывающая скважина (is_injector=False), '
                             f'сейчас их {len(producers)} из {self.n_wells}')
        self._producer = producers[0]

        self.integr_r2_fi0, self.integr_r4_fi0 = _calc_integrals()
        self._wells_names = [item['name'] for item in self._wells_buffer]
        self.wells = preprocess_wells(self._wells_buffer)
        self.max_dfw = _calc_max_dfw(init_T, self.wells)


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
                    if self.wells[self._producer].eta >= max_eta:
                        break
        except (KeyboardInterrupt, SystemError):
            pass
        finally:
            print('KIN:', round(self.KIN, 5))
            print('eta:', round(self.wells[self._producer].eta, 5))
            if self._layers_file is not None:
                self._layers_file.close()
                self._layers_file = None
            convert_pkl_files(self._i_img)
            rmtree(results_path)


    def upd_time_step(self, t: float) -> None:
        """Решение задачи на текущем временном слое. Метод IMPES: явный по насыщенности, неявный по давлению."""
        self._t = t
        step_dt = self.dt

        # Подвижности фаз - общие для сборки матрицы давления и для перетоков
        calc_mobility(self.k, self.S, self.m, self.Wo, self.Wp, self.Wps,
                      self.mu_o, self.mu_w, self.lam_o, self.lam_w, self.lam_h)
        # Обновление давления. Проницаемость берется с текущего слоя: блок кольматации идет ниже,
        # в общем цикле по ячейкам, поэтому k отстает от m на полшага.
        self._band_age = calc_pressure(self.k, self.S, self.mu_o, self.mu_w, self.lam_o, self.lam_w, self.wells,
                                       self.diag, self.ex, self.ey, self.rhs, self.band_w, self.p_vec, self.pcg_r, self.pcg_z, self.pcg_p, self.pcg_q,
                                       self.boundary_conditions, self._band_age, self.p)
        # Обновление данных скважин
        self.KIN = _update_wells_data(self.n_wells, self.wells, self.p, self.S, self.mu_o, self.mu_w, step_dt)
        # Источники скважин в тех же единицах, что и перетоки через грани
        _wells_loop(self.n_wells, self.wells, self.T, self.C_o, self.C_w, self.C_p,
                    self.Wo, self.Wp, self.Wps, self.src_S, self.src_Wp, self.src_T)
        # Решение уравнений по явной схеме
        dt_cells = _equations_loop(self._t, self._paraphin, self.boundary_conditions, self.p, self.grad_p, self._Um_r2, self.qp1, self.qp2, self.new_qp1, self.new_qp2, self.k, self.new_k, self.m, self.new_m, self.S, self.new_s, self.Wo, self.Wp, self.new_wp, self.Wps, self.new_wps, self.T, self.T_0, self.new_t,
                        self.fi, self.new_fi, self.h_sloy, self.new_h, self.Ur, self.new_Ur, self.Ub, self.new_Ub, self.integr_r2_fi0, self.integr_r4_fi0, self.a_tdma, self.b_tdma, self.C_o, self.C_w, self.C_p, self.C_f, self.E_ff, self.cells_T_eq, self.cells_Wp_eq, self.cells_S_eq, self.cells_Q_out, self.src_S, self.src_Wp, self.src_T, self.mu_o, self.mu_w, self.lam_o, self.lam_w, self.lam_h, self.max_dfw, step_dt)
        # Шаг для следующей итерации из фактического условия устойчивости
        dt_next = _calc_dt(self.n_wells, self.wells, self.m, self.cells_Q_out, self.max_dfw, step_dt, dt_cells)

        if not np.isfinite(self.p.sum()):
            raise FloatingPointError('В поле давления появились NaN/Inf')

        _swap_time_steps(self._paraphin, self.qp1, self.new_qp1, self.qp2, self.new_qp2, self.k, self.new_k, self.m, self.new_m,
                         self.S, self.new_s, self.Wo, self.Wp, self.new_wp,self.Wps, self.new_wps, self.T, self.T_0, self.new_t,
                         self.fi, self.new_fi, self.h_sloy, self.new_h, self.Ur, self.new_Ur, self.Ub, self.new_Ub, self.mu_o, self.mu_w)
        self.dt = dt_next

        # Запись данных в файл
        if t >= self._i_img * sol_time_step or np.isclose(t, Time_end) or self.wells[self._producer].eta >= max_eta:
            _logging_solution(self, t)
            save_fields(self, t)
            self._i_img += 1


@njit(parallel=True, cache=True)
def _equations_loop(_t, _paraphin, boundary_conditions, p, grad_p, _Um_r2, qp1, qp2, new_qp1, new_qp2, k, new_k, m, new_m, S, new_s, Wo, Wp, new_wp, Wps, new_wps, T, T_0, new_t,
                    fi, new_fi, h_sloy, new_h, Ur, new_Ur, Ub, new_Ub, integr_r2_fi0, integr_r4_fi0, a_tdma, b_tdma, C_o, C_w, C_p, C_f, E_ff, cells_T_eq, cells_Wp_eq, cells_S_eq, cells_Q_out, src_S, src_Wp, src_T, mu_o, mu_w, lam_o, lam_w, lam_h, max_dfw, dt):
    """Решение уравнений по явной схеме в цикле по ячейкам.

    Порядок повторяет порядок вычислений на шаге из постановки задачи: кольматация (fi -> m, k,
    q_p1, q_p2), перетоки и дебиты, насыщенность, перенос парафина, температура. Перенос парафина
    делит на (m*S_o) нового слоя, а температуре нужен w_p нового слоя для скрытой теплоты, поэтому
    переставлять эти три вызова нельзя.

    Ячейки независимы - каждая пишет только в свои [i, j], поэтому внешний цикл идет в prange.
    Прогоночные буферы a_tdma, b_tdma нарезаются по i: один общий буфер на все ячейки давал бы гонку потоков.
    """
    dt_cells = dt_max
    for i in prange(Nx):
        for j in range(Ny):
            calc_Um_r2(i, j, p, grad_p, _Um_r2, mu_o)  # Средняя скорость в капилляре * r^2

            # ---решение задачи кольматации\суффозии---
            if _paraphin:
                # Обновление толщины осадочного слоя, скорости изменения радиуса капилляра и коэффициента блокирования
                calc_velocities_h(i, j, S, _Um_r2, Wps, mu_o, h_sloy, Ur, new_h, new_Ur, new_Ub, dt)
                # Обновление функции пор по размерам, скоростей потери порового объема, пористости, проницаемости
                calc_qp_m_k_fi(i, j, Wps, m, k, fi, Ur, Ub, integr_r2_fi0, integr_r4_fi0, a_tdma[i], b_tdma[i], new_qp1, new_qp2, new_fi, new_k, new_m, dt)

            # ---решение гидродинамики---
            qo_out, t_out = flows_in_cells(i, j, boundary_conditions, p, S, T, k, mu_o, mu_w, lam_o, lam_w, lam_h, m, Wo, Wp, Wps, C_o, C_w, C_p, cells_T_eq, cells_Wp_eq, cells_S_eq, cells_Q_out)
            # Скважины входят в уравнения наравне с перетоками через грани, поэтому делятся на те же поля нового слоя.
            # Буферы src_* заполнены в `_wells_loop` до цикла.
            cells_S_eq[i, j] += src_S[i, j]
            cells_Wp_eq[i, j] += src_Wp[i, j]
            cells_T_eq[i, j] += src_T[i, j]

            saturation_equation(i, j, S, m, cells_S_eq, new_m, new_s, dt)
            if _paraphin:
                wp_equation(i, j, qp1, qp2, m, S, Wp, Wps, T, cells_Wp_eq, new_m, new_s, new_wp, new_wps, dt)
            psi = temperature_equation(i, j, T, T_0, m, S, C_o, C_w, C_f, C_p, Wo, Wp, Wps, new_wp, new_wps, cells_T_eq, _t, E_ff, new_t, new_m, new_s, dt)

            # Три ограничения на шаг по числу Куранта: по насыщенности, по переносу парафина и по температуре.
            # Скважинная часть первого добирается в `_calc_dt`.
            q_out = cells_Q_out[i, j]
            if q_out > 1e-30:
                dt_cells = min(dt_cells, CFL_target * m[i, j] * volume / (max_dfw * q_out))

            if _paraphin and qo_out > 1e-30:
                dt_cells = min(dt_cells, CFL_target * m[i, j] * (1.0 - S[i, j]) * volume / qo_out)

            if t_out > 1e-30:
                dt_cells = min(dt_cells, CFL_target * psi * volume / t_out)

    return dt_cells


@njit
def _calc_dt(n_wells, wells, m, cells_Q_out, max_dfw, dt_prev, dt_cells):
    """Шаг по времени из условия устойчивости явной схемы по насыщенности.

    Ограничение на ячейку: dt <= CFL * m*V / (max|df_w/dS| * Q_отток). По самим ячейкам минимум
    уже посчитан в `_equations_loop` и приходит сюда как `dt_cells`. Здесь остаются только скважины: их отбор идет мимо граней,
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
def _wells_loop(n_wells, wells, T, C_o, C_w, C_p, Wo, Wp, Wps, src_S, src_Wp, src_T):
    """Источники скважин в тех же единицах, что и перетоки через грани ячейки.

    Дебиты знаковые (q > 0 - закачка), поэтому источники складываются с перетоками как есть.
    Буферы не обнуляются: скважины неподвижны, их ячейки перезаписываются каждый шаг, а остальные
    так и остаются нулями с момента создания.
    """
    for i in range(n_wells):
        well = wells[i]
        src_S[well.i, well.j] = well.q[1]
        src_Wp[well.i, well.j] = (Wp[well.i, well.j] + Wps[well.i, well.j]) * well.q[0]
        src_T[well.i, well.j] = temperature_source(well, T, C_o, C_w, C_p, Wo, Wp, Wps)


@njit
def _update_wells_data(n_wells, wells, p, S, mu_o, mu_w, dt):
    """Обновление дебита и обводненности скважин.

    Зовется после `calc_pressure`: дебиты берутся по давлению нового слоя и по коэффициентам
    продуктивности, уже ушедшим в матрицу, то есть неявно.
    """
    Q_oil = 0.0
    for i in range(n_wells):
        wells[i] = upd_q_and_eta(wells[i], p, S, mu_o, mu_w, dt)
        if wells[i].is_injector == 0:
            Q_oil -= wells[i].Q[0]  # у добывающей q < 0, а добыча положительна

    # Вычисление КИН
    return Q_oil / geological_reserves


@njit(parallel=True, cache=True)
def _swap_time_steps(_paraphin, qp1, new_qp1, qp2, new_qp2, k, new_k, m, new_m, S, new_s,
                     Wo, Wp, new_wp, Wps, new_wps, T, T_0, new_t,
                     fi, new_fi, h_sloy, new_h, Ur, new_Ur, Ub, new_Ub, mu_o, mu_w):
    """Обновление полей данных на новом временном слое."""
    for i in prange(Nx):
        for j in range(Ny):
            # Пересчет свойств флюидов из-за изменения температуры
            mu_o[i, j] = calc_mu_o(new_t[i, j], new_wps[i, j])
            mu_w[i, j] = calc_mu_w(new_t[i, j])
            # C_w[i, j] = c_w  # calc_c_w(self.T[i, j])
            # C_o[i, j] = c_o  # calc_c_o(self.T[i, j])
            # C_f[i, j] = c_f  # calc_c_f(self.T[i, j])
            # C_p[i, j] = c_p  # calc_c_p(self.T[i, j])

            S[i, j]   = new_s[i, j]
            T_0[i, j] = T[i, j]
            T[i, j]   = new_t[i, j]

            if _paraphin:
                was_clogging = Wps[i, j] > min_Wps_bound
                Wp[i, j]  = new_wp[i, j]
                Wps[i, j] = new_wps[i, j]
                Wo[i, j]  = 1.0 - new_wp[i, j] - new_wps[i, j]
                k[i, j]   = new_k[i, j]
                m[i, j]   = new_m[i, j]
                qp1[i, j] = new_qp1[i, j]
                qp2[i, j] = new_qp2[i, j]

                # Ниже порога кольматации поля по радиусам не пишутся, копия была бы тождественной
                if was_clogging:
                    fi[i, j, :]     = new_fi[i, j, :]
                    h_sloy[i, j, :] = new_h[i, j, :]
                    Ur[i, j, :]     = new_Ur[i, j, :]
                    Ub[i, j, :]     = new_Ub[i, j, :]


def _calc_max_dfw(init_T, wells) -> float:
    """Максимум производной функции Баклея-Леверетта на рабочем диапазоне насыщенности.

    Именно эта величина задает предел устойчивости явной схемы по насыщенности: занизишь ее -
    `_calc_dt` разрешит слишком большой шаг. Функция зависит от отношения вязкостей, а оно - от
    температуры, и пласт по мере закачки остывает от init_T до температуры нагнетаемой воды.
    """
    # Диапазон температур расчета: от начальной пластовой до самой холодной закачиваемой воды
    temps = [init_T] + [well.T for well in wells if well.is_injector == 1]

    s = np.linspace(S_min, S_max, 2001)
    max_dfw = 0.0
    for t in np.linspace(min(temps), max(temps), 11):
        # Вязкость нефти зависит еще и от доли выпавшего парафина
        for w_ps in (0.0, init_Wp):
            f_w = np.array([Buckley_Leverett(x, calc_mu_w(t), calc_mu_o(t, w_ps)) for x in s])
            max_dfw = max(max_dfw, float(np.abs(np.gradient(f_w, s)).max()))

    return max_dfw


def _logging_solution(solver, t):
    """Логирование решения задачи."""
    if not LOGGING:
        return None

    solver.logger.info('')
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
    solver.logger.info(f"Wo:   min={solver.Wo.min()}  max={solver.Wo.max()}")

    solver.logger.info(f"qp1:    min={solver.new_qp1.min()}  max={solver.new_qp1.max()}")
    solver.logger.info(f"qp2:    min={solver.new_qp2.min()}  max={solver.new_qp2.max()}")
    solver.logger.info(f"m_mult: min={solver.new_m.min()}  max={solver.new_m.max()}")
    solver.logger.info(f"k_mult: min={solver.new_k.min()}  max={solver.new_k.max()}")

    solver.logger.info(f"fi:   min={solver.fi.min()}  max={solver.fi.max()}")
    solver.logger.info(f"Ur:   min={solver.new_Ur.min()}  max={solver.new_Ur.max()}")
    solver.logger.info(f"Ub:   min={solver.new_Ub.min()}  max={solver.new_Ub.max()}")
    solver.logger.info(f"Um:   min={solver._Um_r2.min()*1e-12}  max={solver._Um_r2.max()*1e-12}")

    # self.logger.info(f"fi:   {' '.join([f'{x:.{3}f}' for x in self.fi)[0, 0]])}")
    # self.logger.info(f"fi_0: {' '.join([f'{x:.{3}f}' for x in fi_0])}")
