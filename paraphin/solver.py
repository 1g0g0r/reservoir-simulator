"""Класс содержит алгоритм расчета и хранение данных."""
from logging import INFO, getLogger, Formatter, FileHandler
from shutil import rmtree
from sys import stdout
from time import perf_counter

import math

import numpy as np
from numba import njit, prange
from tqdm import tqdm

from paraphin.geometry import r1, r3, r4, r5, r6, fi_0, surf_0
from .constants import (data_type, Nx, Ny, Nr, rw, results_path, data_path, logs_path, init_T, init_k, init_S, init_m,
                        init_p, init_Wp, init_Wps, bar_to_pa, dt, day_to_sec,
                        max_eta, c_o, c_w, c_p, c_f, sol_time_step, Time_end, LOGGING, geological_reserves,
                        volume, S_min, S_max, CFL_target, dt_growth, dt_max, dt_min,
                        min_Wps_bound, wax_components, asphaltenes, gelation, pressure_viscosity,
                        gel_time, gel_mobility_min, alpha_p_visc, P_ref_wax, sara_asphaltenes, case_name,
                        ro_asph_dep, ro_asph, deposition_kinetics, deposition_model, wax_kinetics,
                        asph_aggregation, wettability, thermal_nonequilibrium, adsorption,
                        ads_init_equilibrium, R, pore_network, p_guess_m)
from .utils.math_utils import MG_ROWS, MG_TOTAL, MG_COARSE, MG_COARSE_KD, P_STATE
from .equations import (calc_qp_m_k_fi, calc_pressure, saturation_equation, temperature_source,
                        temperature_equation, wp_equation, calc_velocities_h, flows_in_cells,
                        calc_Um_r2, components_equation, calc_qp_m_k_fi_2, calc_velocity_asph, oil_viscosity,
                        sle_split)
from .equations.Asphaltene import asph_soluble
from .equations.Deposition import calc_deposition, calc_filtration, network_g0
from .kinetics_params import (default_kin, AGG_D0, OW_S_MIN, OW_S_MAX, OW_N_O, OW_N_W, ADS_GMAX, ADS_K, ADS_DH,
                              ADS_T_REF, ADS_RESIN, PERM_BETA, PERM_SMAX)
from .layout import (N_W, NC, IA_D, IA_F, I_R, IS0, IN_F, NKX, KX_WEQ, KX_TS, KX_GA, KX_GR, KX_GMAX, KX_SIG0, KX_UA, KX_QPA,
                     NROWS, ROW_A, ROW_B, ROW_FO, ROW_UE, ROW_TMP, NP)
from .equations.Kinetics_math import langmuir_constant
from .equations.Thermal_ltne import temperature_equation_ltne
from .oil_composition import WAX_W0, F_SAT_REST, initial_components
from .utils import (calc_mu_o, calc_mu_p, calc_mu_w, preprocess_wells, convert_pkl_files,
                    save_fields, Bound, TypeBC, DataField, add_bc, new_well, upd_q_and_eta, Buckley_Leverett,
                    calc_mobility, calc_mobility_w)


class Solver:
    def __init__(self):
        # Граничные условия и скважины
        self.KIN = data_type(0.0)  # Коэффициент извлечения нефти (КИН), [-]
        self.n_wells = 0         # Число добавленных скважин
        self._wells_buffer = []  # Буфер скважин до обработки
        self._wells_names = []   # Имена скважин по порядку add_well, для логирования и сохранения полей
        self.well_data = None    # Скважины для ядер: структурный массив с dtype `WELL` (`utils/well.py`)
        self.wells = []          # Они же для Python-кода: вид np.recarray на тот же буфер, `wells[k].q`
        self.boundary_conditions = np.zeros(dtype=data_type, shape=(4, 4, 2))  # Граница -> Поле (DataField) -> Тип, Значение
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
        self.Wo    = np.full((Nx, Ny), 1.0 - init_Wp - init_Wps, data_type)  # Массовая доля масляного компонента, [-]
        self.Wp    = np.full((Nx, Ny), init_Wp, data_type)   # Массовая доля растворенного парафина, [-]
        self.Wps   = np.full((Nx, Ny), init_Wps, data_type)  # Массовая доля взвешенного парафина, [-]
        self.k     = np.full((Nx, Ny), init_k, data_type)  # Проницаемость, [м^2]
        self.m     = np.full((Nx, Ny), init_m, data_type)  # Пористость, [-]
        # Динамика образования парафина (кольматация\суффозия)
        self.integr_r2_fi0 = data_type(0.0)  # Интеграл r^2*fi_0(r) по сетке радиусов пор, считается в initialize()
        self.integr_r4_fi0 = data_type(0.0)  # Интеграл r^4*fi_0(r) по сетке радиусов пор, считается в initialize()
        self.fi      = np.ones((Nx, Ny, Nr), data_type) * fi_0               # Функция пор по размерам, [-]
        self.h_sloy  = np.zeros((Nx, Ny, Nr), data_type)     # Толщина осадочного слоя парафина, [м]
        self.qp1     = np.zeros((Nx, Ny), data_type)         # Скорость осаждения парафина на стенках пор, [1/сек]
        self.qp2     = np.zeros((Nx, Ny), data_type)         # Скорость блокирования поровых каналов, [1/сек]
        self.grad_p  = np.zeros((Nx, Ny), data_type)         # Градиент давления, [Па/м]
        self._Um_r2  = np.zeros((Nx, Ny), data_type)         # Компонент скорости фильтрации в капилляре радиуса r, [1/(м·сек)]
        self.Ur      = np.zeros((Nx, Ny, Nr), data_type)     # Скорость сужения капилляров, [м/сек]
        self.Ub      = np.zeros((Nx, Ny, Nr), data_type)     # Скорость блокирования капилляров, [1/сек]
        # Поля данный нового временного слоя
        self.new_h   = np.zeros((Nx, Ny, Nr), data_type)  # Толщина осадочного слоя парафина на новом временном слое, [м]
        self.new_Ur  = np.zeros((Nx, Ny, Nr), data_type)  # Скорость сужения капилляров на новом временном слое, [м/сек]
        self.new_Ub  = np.zeros((Nx, Ny, Nr), data_type)  # Скорость блокирования капилляров на новом временном слое, [1/сек]
        self.new_fi  = np.ones((Nx, Ny, Nr), data_type) * fi_0   # Функция пор по размерам на новом временном слое, [-]
        self.new_s   = np.zeros((Nx, Ny), data_type)      # Водонасыщенность на новом временном слое, [-]
        self.new_t   = np.zeros((Nx, Ny), data_type)      # Температура на новом временном слое, [С]
        self.new_wps = np.full((Nx, Ny), init_Wps, data_type)  # Массовая доля взвешенного парафина на новом временном слое, [-]
        self.new_wp  = np.full((Nx, Ny), init_Wp, data_type)   # Массовая доля растворенного парафина на новом временном слое, [-]
        self.new_qp1 = np.zeros((Nx, Ny), data_type)      # Скорость осаждения на новом временном слое, [1/сек]
        self.new_qp2 = np.zeros((Nx, Ny), data_type)      # Скорость блокирования на новом временном слое, [1/сек]
        self.new_k   = np.full((Nx, Ny), init_k, data_type)    # Пористость на новом временном слое, [м^2]
        self.new_m   = np.full((Nx, Ny), init_m, data_type)    # Проницаемость на новом временном слое, [-]
        # Временные массивы перетоков через границы ячеек
        self.cells_T_eq  = np.zeros((Nx, Ny), data_type)  # Суммарный переток температуры в ячейке
        self.cells_Wp_eq = np.zeros((Nx, Ny), data_type)  # Суммарный переток растворенного в ячейке
        self.cells_S_eq  = np.zeros((Nx, Ny), data_type)  # Суммарный переток водонасыщенности в ячейке
        self.cells_Q_out = np.zeros((Nx, Ny), data_type)  # Суммарный отток через грани ячейки, [м^3/с]
        # Источники скважин
        self.src_S  = np.zeros((Nx, Ny), data_type)  # Дебит по воде, [м^3/с]
        self.src_Wp = np.zeros((Nx, Ny), data_type)  # Вынос парафина нефтью, [м^3/с]
        self.src_T  = np.zeros((Nx, Ny), data_type)  # Приток энергии со скважиной, [Вт]
        # Подвижности фаз: считаются один раз за шаг и переиспользуются сборкой матрицы и перетоками
        self.lam_o = np.zeros((Nx, Ny), data_type)  # Подвижность нефтяной фазы, [м^2/(Па*с)]
        self.lam_w = np.zeros((Nx, Ny), data_type)  # Подвижность водной фазы, [м^2/(Па*с)]
        self.lam_h = np.zeros((Nx, Ny), data_type)  # эффективная теплопроводность ячейки, [Вт/(м*C)]
        # Состояние метода Винсома-Вестервельда: накопленный интеграл перегрева пород, [C*м].
        self.E_ff = np.zeros((Nx, Ny), data_type)
        # Вспомогательные поля класса
        self._t = 0.0    # Физическое время текущего слоя, [с]
        self._i_img = 0  # Индекс следующего сохраняемого слоя (расписание sol_time_step)
        self._results_file = None  # общий файл слоев, открывается при первом сохранении
        self.dt = dt               # Текущий шаг по времени, подбирается по CFL каждую итерацию, [с]
        self.max_dfw = 1.0         # max|df_w/dS|, задается в initialize()
        self._producer = -1        # Индекс добывающей скважины, ищется в initialize() по is_injector
        self._paraphin = not np.isclose(init_Wp + init_Wps, 0.0)  # Флаг: включен ли блок кольматации/суффозии
        # Скретч: рабочие профили ячейки (прогонка fi, профили по радиусам, потоки граней) - свои строки на каждый i,
        # иначе гонка в prange по ячейкам; имена строк - `layout.ROW_*`
        self.rows = np.zeros((Nx, NROWS, Nr), data_type)
        # Уравнение давления (`calc_pressure`): PCG с многосеточным предобуславливателем. Векторы - в раскладке `layout`
        # (ячейка - P0 + i + j*PX, с рамкой фиктивных ячеек, как уровни решателя). Матрица - мелкий уровень `mg_buf`
        # (diag, ex, ey - его строки), невязка и предобусловленная невязка PCG - его строки b и x
        self.mg_buf = np.zeros((MG_ROWS, MG_TOTAL), data_type)  # уровни многосеточного решателя
        self.mg_wc = np.zeros((MG_COARSE, MG_COARSE_KD + 1), data_type)  # фактор самого грубого уровня
        self.diag, self.ex, self.ey = self.mg_buf[0, :NP], self.mg_buf[1, :NP], self.mg_buf[2, :NP]
        self.rhs   = np.zeros(NP, data_type)
        self.pcg_p = np.zeros(NP, data_type)  # PCG: направление поиска
        self.pcg_q = np.zeros(NP, data_type)  # PCG: M*p
        # возраст грубых уровней, сумма итераций с пересборки, флаг пересборки, голова кольца прошлых решений
        self.p_state = np.zeros(P_STATE, data_type)
        self.p_vec = np.full(NP, init_p, data_type)  # решение; перед решением - начальное приближение
        self.p_hist = np.tile(self.p_vec, (p_guess_m, 1))  # кольцо прошлых решений - для начального приближения
        self.p_iters = 0                # итераций PCG на последнем шаге
        # Детальный состав нефти (флаг `wax_components`, см. `paraphin/oil_composition.py`). Массивы есть всегда,
        # с выключенными флагами они не используются, а `Hl` и `mu_p` - просто другие имена `Wp` и `mu_o`.
        self.Wc     = np.tile(initial_components(), (Nx, Ny, 1))  # доли компонентов в нефтяной фазе, [-]
        self.new_Wc = self.Wc.copy()
        self.Ws     = np.zeros((Nx, Ny, N_W), data_type)  # взвешенные кристаллы по группам парафина, [-]
        self.new_Ws = np.zeros((Nx, Ny, N_W), data_type)  # на новом слое; он же буфер `sle_split`
        self.Dep    = np.zeros((Nx, Ny, NC), data_type)   # накопленные отложения по компонентам, [кг/м^3 породы]
        self.src_Qo = np.zeros((Nx, Ny), data_type)       # дебит нефти скважины, [м^3/с]
        self.bc_Wc  = np.zeros((4, NC), data_type)        # состав втекающей нефти по границам (ГУ Дирихле)
        self.Phi = np.ones((Nx, Ny), data_type)           # множитель подвижности нефти от геля, [-]
        # Кинетика осаждения (`deposition_kinetics`, `equations/Deposition.py`): параметры - runtime-вектор (калибровка
        # меняет его без перекомпиляции), поля моделей - один массив с именованными индексами (`layout`)
        self.kin = default_kin()
        self.kx = np.zeros((Nx, Ny, NKX), data_type)
        self.kx[..., KX_TS] = init_T
        self.new_kx = self.kx.copy()
        self.Ua = self.kx[..., KX_UA]    # коэффициент сужения капилляров флокулами (вид на kx), [м^(2/3)/с]
        self.qpa = self.kx[..., KX_QPA]  # скорость потери порового объема на осадок асфальтенов (вид на kx), [1/с]
        self.WAT = np.zeros((Nx, Ny), data_type)          # температура начала кристаллизации - только выгрузка, [C]
        if wax_components:
            self.Hl, self.new_Hl = np.zeros((Nx, Ny), data_type), np.zeros((Nx, Ny), data_type)
            self._init_composition()
        else:
            self.Hl, self.new_Hl = self.Wp, self.new_wp  # носитель скрытой теплоты - сам растворенный парафин
        self.mu_p = self.mu_o.copy() if gelation else self.mu_o  # вязкость без геля (пластическая)
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
        producers = [idx for idx, item in enumerate(self._wells_buffer) if not item['well']['is_injector']]
        if len(producers) != 1:
            raise ValueError('Ожидается ровно одна добывающая скважина (is_injector=False), '
                             f'сейчас их {len(producers)} из {self.n_wells}')
        self._producer = producers[0]

        # Если все скважины с заданным дебитом, задача с непроницаемыми границами чисто нейманнова:
        # матрица вырождена, а решение определено с точностью до константы.
        if all(item['well']['rate_control'] for item in self._wells_buffer):
            raise ValueError('Хотя бы одна скважина должна работать на заданном забойном давлении '
                             '(аргумент p в add_well): иначе уровень давления ничем не закреплен и матрица вырождена')

        self.integr_r2_fi0, self.integr_r4_fi0 = _calc_integrals()
        if adsorption and ads_init_equilibrium:
            self._init_retention()
        # Состав втекающей нефти по ГУ Дирихле для парафина: если не задан явно (`add_inflow_composition`),
        # группы - в начальной пропорции с заданной суммой, асфальтены и смолы - как в пласте
        for bound in range(4):
            if self.boundary_conditions[bound, 3, 0] == 1 and not self.bc_Wc[bound].any():
                self.bc_Wc[bound] = initial_components()
                self.bc_Wc[bound, :N_W] *= self.boundary_conditions[bound, 3, 1] / WAX_W0.sum()
                if wax_kinetics:  # втекающая нефть - с равновесной взвесью при температуре границы
                    t_bc = (self.boundary_conditions[bound, 2, 1] if self.boundary_conditions[bound, 2, 0] == 1
                            else init_T)
                    sus = np.zeros(N_W, data_type)
                    sle_split(self.bc_Wc[bound, :N_W].copy(), float(t_bc), float(init_p), sus)
                    self.bc_Wc[bound, IS0:IS0 + N_W] = sus
        self._wells_names = [item['name'] for item in self._wells_buffer]
        self.well_data = preprocess_wells(self._wells_buffer)
        self.wells = self.well_data.view(np.recarray)  # в ядра - только well_data: recarray numba типизирует медленно
        self.max_dfw = _calc_max_dfw(init_T, self.wells)


    def _init_retention(self) -> None:
        """Начальное удержание смол и асфальтенов в равновесии с пластовой нефтью при init_T (`ads_init_equilibrium`).

        Порода и нефть в пласте в контакте геологическое время, поэтому init_m и init_k относятся к породе уже с
        удержанным слоем: пористость не уменьшается, а повреждение проницаемости считается от начального объема
        sigma_0 (`KX_SIG0`). Нужен параметр кинетики `solver.kin`, поэтому вызывается из `initialize`, а не из
        конструктора. Тождество пористости тогда - от начального удержания: m0 - m = ... + (G - G_0)/ro_ad."""
        kin = self.kin
        k_l = langmuir_constant(kin[ADS_K], kin[ADS_DH], init_T + 273.15, kin[ADS_T_REF] + 273.15, R)
        g_max = kin[ADS_GMAX] * surf_0
        for idx, comp, mult in ((KX_GA, IA_D, 1.0), (KX_GR, I_R, kin[ADS_RESIN])):
            c = self.Wc[..., comp]
            self.kx[..., idx] = g_max * mult * k_l * c / (1.0 + k_l * c)  # изотерма Ленгмюра (`langmuir_eq`)
        self.kx[..., KX_GMAX] = g_max
        self.kx[..., KX_SIG0] = (self.kx[..., KX_GA] + self.kx[..., KX_GR]) / ro_asph_dep
        self.new_kx[:] = self.kx
        rel = float(kin[PERM_BETA] * self.kx[..., KX_SIG0].max() / (kin[PERM_SMAX] * init_m))
        if rel >= 1.0:  # D(sigma_0) = 0: от полностью поврежденного начального состояния повреждение не отсчитать
            raise ValueError(f'Равновесное удержание при {init_T:g} C исчерпывает функцию повреждения '
                             f'(beta*sigma_0/(sigma_max*m0) = {rel:.2f} >= 1): параметры удержания не согласованы '
                             f'с пластовой температурой')
        print(f'Начальное удержание в равновесии при {init_T:g} C: асфальтены {self.kx[0, 0, KX_GA]:.3g}, смолы '
              f'{self.kx[0, 0, KX_GR]:.3g} кг/м^3 породы, {self.kx[0, 0, KX_SIG0] / init_m:.2e} порового объема')


    def add_bc(self, field: DataField, bound: Bound, type_bc: TypeBC, value: float) -> None:
        """Учет граничных условий для полей данных."""
        add_bc(self.boundary_conditions, bound.value, field.value, type_bc.value, value)


    def add_inflow_composition(self, bound: Bound, wc) -> None:
        """Состав втекающей через границу нефти по компонентам `Wc` (детальный состав, `wax_components`).

        Ставит и ГУ Дирихле `DataField.Paraffin` на сумму групп парафина, чтобы старый путь расчета видел ту же долю.
        """
        wc = np.asarray(wc, data_type)
        if wc.shape != (NC,):
            raise ValueError(f'Состав втекающей нефти: ожидается {NC} компонентов, передано {wc.shape}')
        self.bc_Wc[bound.value] = wc
        self.add_bc(DataField.Paraffin, bound, TypeBC.Dirichlet, float(wc[:N_W].sum()))


    def _init_composition(self) -> None:
        """Начальное равновесие детального состава: группы парафина - растворенные и взвешенные при init_T,
        init_p, асфальтены - равновесная доля флокул. Неустойчивые в пласте асфальтены - признак того, что
        давление начала осаждения задано выше пластового, об этом выводится предупреждение."""
        wc = self.Wc[0, 0].copy()
        sus = np.zeros(N_W, data_type)
        w_dis, w_sus, hl = sle_split(wc[:N_W], float(init_T), float(init_p), sus)
        if asphaltenes:
            rest = 1.0 - wc[:N_W].sum() - wc[IA_D] - wc[IA_F] - wc[I_R]
            w_max = asph_soluble(rest * F_SAT_REST + w_dis, rest * (1.0 - F_SAT_REST), wc[I_R], float(init_p), float(init_T))
            a_tot = wc[IA_D] + wc[IA_F]
            wc[IA_F] = max(a_tot - w_max, 0.0)
            wc[IA_D] = a_tot - wc[IA_F]
            if wc[IA_F] > 0.0:
                print(f'Предупреждение: при init_p = {init_p / 1e6:.2f} МПа асфальтены неустойчивы уже в начальном '
                      f'состоянии ({wc[IA_F] / sara_asphaltenes:.1%} флокулировано): P_onset_asph выше пластового давления')
        if wax_kinetics:  # в начале взвесь равновесна
            wc[IS0:IS0 + N_W] = sus
            self.kx[..., KX_WEQ:KX_WEQ + N_W] = sus
            self.new_kx[:] = self.kx
        if asph_aggregation and wc[IA_F] > 0.0:  # начальные флокулы - первичные частицы
            d0 = self.kin[AGG_D0]
            wc[IN_F] = wc[IA_F] / (ro_asph * math.pi * d0 ** 3 / 6.0)
        self.Wc[:] = wc
        self.new_Wc[:] = wc
        self.Ws[:] = sus
        self.new_Ws[:] = sus
        self.Wp[:], self.new_wp[:] = w_dis, w_dis
        self.Wps[:], self.new_wps[:] = w_sus, w_sus
        self.Wo[:] = 1.0 - w_dis - w_sus
        self.Hl[:], self.new_Hl[:] = hl, hl
        self.mu_o[:] = calc_mu_p(init_T, w_sus, init_p)


    def add_well(self, name: str, i: int, j: int, p: float | None = None, q: float | None = None,
                 is_injector: bool = False, T: float = 0.0, rw: float = rw, mult: float = 1.0) -> None:
        """Добавление скважины в расчет.

        Режим работы задается тем, что передано: `p` - забойное давление, [Па] (дебит считается по формуле Писмана),
        либо `q` - суммарный дебит, [м^3/с] (забойное давление считается по формуле Писмана). Только одно из двух.

        Дебит задается положительным для любой скважины - это расход, а не знаковый источник.
        Знак дебита определяется в соответствии с `is_injector`: (`q > 0` - закачка, `q < 0` - отбор),
        поэтому у добывающей он меняется на противоположный, а у нагнетательной остается как есть.
        """
        if (q is None) == (p is None):
            raise ValueError(f'Скважина {name}: задайте ровно одно - забойное давление p '
                             f'или дебит q (сейчас p={p}, q={q})')

        if q is not None and q <= 0.0:
            raise ValueError(f'Скважина {name}: дебит задается положительным для любой скважины, '
                             f'знак ставится по is_injector; задано q={q}')

        # Внутрь идет знаковый источник: у добывающей расход меняет знак
        q_set = 0.0 if q is None else (q if is_injector else -q)
        p_set = 0.0 if q is not None else p
        well = new_well(i=i, j=j, p=p_set, q_set=q_set, rate_control=int(q is not None), T=T, rw=rw,
                        is_injector=int(is_injector), mult=mult)
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
                    _t += self.dt
                    pbar.update(self.dt)
                    self.upd_time_step(_t)
                    # Без refresh=False tqdm перерисовывал строку каждый шаг: 79 мкс, ~14 % шага 75x75. Строка и так
                    # обновляется раз в mininterval (0.1 с) и подхватит последний postfix
                    pbar.set_postfix(день=_t / day_to_sec, шаг_сут=round(self.dt / day_to_sec, 5), refresh=False)
                    if self.wells[self._producer].eta >= max_eta:
                        break
        except (KeyboardInterrupt, SystemError):
            pass
        finally:
            print('KIN:', round(self.KIN, 5))
            print('eta:', round(self.wells[self._producer].eta, 5))
            if self._results_file is not None:
                self._results_file.close()
            if wax_components:
                # Полный детальный состав конца расчета: в слоях он хранится только компактно (`save_fields`)
                np.savez_compressed(data_path / f'{case_name}_final_composition.npz', Wc=self.Wc, Ws=self.Ws,
                                    Dep=self.Dep, WAT=self.WAT, m=self.m, k=self.k, Phi=self.Phi, T=self.T, p=self.p)
            convert_pkl_files(self._i_img)
            rmtree(results_path)


    def upd_time_step(self, t: float) -> None:
        """Решение задачи на текущем временном слое. Метод IMPES: явный по насыщенности, неявный по давлению."""
        self._t = t
        step_dt = self.dt

        # Подвижности фаз - общие для сборки матрицы давления и для перетоков
        if wettability:  # ОФП - смесь водо- и нефтесмачиваемых по адсорбированным асфальтенам
            kin = self.kin
            calc_mobility_w(self.k, self.S, self.m, self.Wo, self.Wp, self.Wps, self.mu_o, self.mu_w,
                            self.lam_o, self.lam_w, self.lam_h, self.kx[..., KX_GA], self.kx[..., KX_GMAX],
                            kin[OW_S_MIN], kin[OW_S_MAX], kin[OW_N_O], kin[OW_N_W])
        else:
            calc_mobility(self.k, self.S, self.m, self.Wo, self.Wp, self.Wps,
                          self.mu_o, self.mu_w, self.lam_o, self.lam_w, self.lam_h)
        # Обновление давления. Проницаемость берется с текущего слоя: блок кольматации идет ниже,
        # в общем цикле по ячейкам, поэтому k отстает от m на полшага.
        self.p_iters = calc_pressure(self.k, self.S, self.mu_o, self.mu_w, self.lam_o, self.lam_w, self.well_data,
                                     self.diag, self.ex, self.ey, self.rhs, self.mg_buf, self.mg_wc, self.p_state,
                                     self.p_vec, self.p_hist, self.pcg_p, self.pcg_q, self.boundary_conditions, self.p)
        # Обновление данных скважин
        self.KIN = _update_wells_data(self.n_wells, self.well_data, self.p, self.S, self.mu_o, self.mu_w, step_dt)
        # Источники скважин в тех же единицах, что и перетоки через грани
        _wells_loop(self.n_wells, self.well_data, self.T, self.C_o, self.C_w, self.C_p,
                    self.Wo, self.Wp, self.Wps, self.Hl, self.src_S, self.src_Wp, self.src_T, self.src_Qo)
        # Решение уравнений по явной схеме
        dt_cells = _equations_loop(self._t, self._paraphin, self.boundary_conditions, self.p, self.grad_p, self._Um_r2, self.qp1, self.qp2, self.new_qp1, self.new_qp2, self.k, self.new_k, self.m, self.new_m, self.S, self.new_s, self.Wo, self.Wp, self.new_wp, self.Wps, self.new_wps, self.T, self.T_0, self.new_t,
                                   self.fi, self.new_fi, self.h_sloy, self.new_h, self.Ur, self.new_Ur, self.Ub, self.new_Ub, self.integr_r2_fi0, self.integr_r4_fi0, self.C_o, self.C_w, self.C_p, self.C_f, self.E_ff, self.cells_T_eq, self.cells_Wp_eq, self.cells_S_eq, self.cells_Q_out, self.src_S, self.src_Wp, self.src_T, self.mu_o, self.mu_w, self.lam_o, self.lam_w, self.lam_h, self.max_dfw, step_dt,
                                   self.bc_Wc, self.Wc, self.new_Wc, self.Ws, self.new_Ws, self.Hl, self.new_Hl, self.Dep, self.src_Qo, self.mu_p,
                                   self.kin, self.kx, self.new_kx, self.rows)
        # Шаг для следующей итерации из фактического условия устойчивости
        dt_next = _calc_dt(self.n_wells, self.well_data, self.m, self.cells_Q_out, self.max_dfw, step_dt, dt_cells)

        if not np.isfinite(self.p.sum()):
            raise FloatingPointError('В поле давления появились NaN/Inf')

        _swap_time_steps(self._paraphin, self.qp1, self.new_qp1, self.qp2, self.new_qp2, self.k, self.new_k, self.m, self.new_m,
                         self.S, self.new_s, self.Wo, self.Wp, self.new_wp,self.Wps, self.new_wps, self.T, self.T_0, self.new_t,
                         self.fi, self.new_fi, self.h_sloy, self.new_h, self.Ur, self.new_Ur, self.Ub, self.new_Ub, self.mu_o, self.mu_w,
                         self.p, self.grad_p, self.Wc, self.new_Wc, self.Ws, self.new_Ws, self.Hl, self.new_Hl,
                         self.mu_p, self.Phi, self.Dep, self.kx, self.new_kx,
                         step_dt)
        self.dt = dt_next

        # Запись данных в файл
        if (t >= self._i_img * sol_time_step or abs(t - Time_end) <= 1e-6 or
        self.well_data[self._producer]['eta'] >= max_eta):
            _logging_solution(self, t)
            save_fields(self, t)
            self._i_img += 1


@njit(parallel=True, cache=True)
def _equations_loop(_t, _paraphin, boundary_conditions, p, grad_p, _Um_r2, qp1, qp2, new_qp1, new_qp2, k, new_k, m, new_m, S, new_s, Wo, Wp, new_wp, Wps, new_wps, T, T_0, new_t,
                    fi, new_fi, h_sloy, new_h, Ur, new_Ur, Ub, new_Ub, integr_r2_fi0, integr_r4_fi0, C_o, C_w, C_p, C_f, E_ff, cells_T_eq, cells_Wp_eq, cells_S_eq, cells_Q_out, src_S, src_Wp, src_T, mu_o, mu_w, lam_o, lam_w, lam_h, max_dfw, dt,
                    bc_Wc, Wc, new_Wc, Ws, new_Ws, Hl, new_Hl, Dep, src_Qo, mu_p, kin, kx, new_kx, rows):
    """Решение уравнений по явной схеме в цикле по ячейкам.

    Порядок повторяет порядок вычислений на шаге из постановки задачи: перетоки (по текущему слою, от
    кольматации не зависят), кольматация (fi -> m, k, q_p1, q_p2), дебиты, насыщенность, перенос парафина,
    температура. Перенос парафина делит на (m*S_o) нового слоя, а температуре нужен w_p нового слоя для
    скрытой теплоты, поэтому переставлять эти три вызова нельзя.

    Ячейки независимы - каждая пишет только в свои [i, j], поэтому внешний цикл идет в prange.
    Скретч `rows` нарезан по i (прогоночные коэффициенты, профили по радиусам, потоки граней - строки `layout.ROW_*`):
    один общий буфер на все ячейки давал бы гонку потоков. Буфер групп парафина - сама ячейка `new_Ws[i, j]`.

    Детальный состав (флаг `wax_components`) заменяет `wp_equation` на `components_equation`, а
    `calc_qp_m_k_fi` - на `calc_qp_m_k_fi_2`: ограничитель подводом учитывает отток нефти, а асфальтены
    (`asphaltenes`) добавляют сужение капилляров флокулами. Флаги - константы модуля, выключенные ветки
    numba выбрасывает.
    """
    dt_cells = dt_max
    net_g0 = 1.0
    if pore_network:  # проводимость исходной сети пор одна на все ячейки - раз за шаг, до параллельного цикла
        net_g0 = network_g0(kin, rows[0, ROW_TMP], rows[0, ROW_UE])
    # Кольматация без детального состава - отдельным циклом по списку ячеек выше порога. Все они у нагнетательной
    # скважины, в первых строках i, и в общем цикле по строкам их считали один-два потока.
    # Ячейки списка раздаются кускам по кругу, скретч - строка куска `rows[c]`. Перетоков эта
    # кольматация не читает, поэтому идет до общего цикла; ячейки независимы, результат побитово прежний.
    if _paraphin and not deposition_kinetics and not wax_components:
        n_act = 0
        act = np.empty(Nx * Ny, np.int64)
        for ia in range(Nx):
            for ja in range(Ny):
                if Wps[ia, ja] > min_Wps_bound:
                    act[n_act] = ia * Ny + ja
                    n_act += 1
        if n_act > 0:
            n_ch = min(n_act, Nx)  # строк скретча - Nx
            for c in prange(n_ch):
                for ka in range(c, n_act, n_ch):
                    ia = act[ka] // Ny
                    ja = act[ka] % Ny
                    calc_Um_r2(ia, ja, p, grad_p, _Um_r2, mu_o)
                    calc_velocities_h(ia, ja, S, T, _Um_r2, Wps, mu_p, h_sloy, Ur, new_h, new_Ur, new_Ub, dt)
                    calc_qp_m_k_fi(ia, ja, S, Wp, Wps, m, k, fi, Ur, Ub, integr_r2_fi0, integr_r4_fi0, rows[c, ROW_A],
                                   rows[c, ROW_B], new_qp1, new_qp2, new_fi, new_k, new_m, dt)
    for i in prange(Nx):
        for j in range(Ny):
            calc_Um_r2(i, j, p, grad_p, _Um_r2, mu_o)  # Средняя скорость в капилляре * r^2

            # ---перетоки через грани---
            # Читают только текущий слой и от кольматации не зависят; считаются до нее, потому что ограничителю
            # осаждения в детальном составе нужен отток нефти из ячейки за шаг.
            qo_out, t_out = flows_in_cells(i, j, boundary_conditions, bc_Wc, p, S, T, k, mu_o, mu_w, lam_o, lam_w, lam_h, m, Wo, Wp, Wps, Hl, C_o, C_w, C_p, cells_T_eq, cells_Wp_eq, cells_S_eq, cells_Q_out, rows[i, ROW_FO])

            # ---решение задачи кольматации\суффозии---
            if _paraphin:
                if deposition_kinetics:
                    # Кинетика осаждения (`equations/Deposition.py`): скорости переноса к стенке - по текущему слою
                    out_o = qo_out + max(-src_Qo[i, j], 0.0)
                    if deposition_model == 'filtration':
                        calc_filtration(i, j, S, T, p, m, k, fi, Wc, Ws, Wps, Dep, grad_p, lam_o, kx, kin, integr_r2_fi0,
                                        rows[i], new_qp1, new_qp2, new_fi, new_k, new_m, new_kx, out_o, dt)
                    else:
                        calc_deposition(i, j, S, T, p, m, k, fi, h_sloy, Wc, Ws, Wps, Dep, _Um_r2, grad_p, mu_p, kx, kin,
                                        integr_r2_fi0, integr_r4_fi0, rows[i], net_g0,
                                        new_qp1, new_qp2, new_fi, new_h, new_k, new_m, new_kx, out_o, dt)
                elif not wax_components:
                    # Выше порога кольматация посчитана циклом по списку активных ячеек до этого цикла. Ниже порога
                    # `calc_velocities_h` и `calc_qp_m_k_fi` ничего не считают, а вызов с двумя десятками
                    # массивов-аргументов стоит ~100 нс на ячейку (docs/PERFORMANCE_FINDINGS.md, «Раунд 6»), поэтому
                    # здесь ровно то, что сделал бы `calc_qp_m_k_fi` ниже порога.
                    if Wps[i, j] <= min_Wps_bound:
                        new_qp1[i, j] = 0.0
                        new_qp2[i, j] = 0.0
                        new_m[i, j] = m[i, j]
                        new_k[i, j] = k[i, j]
                else:
                    # Обновление толщины осадочного слоя, скорости изменения радиуса капилляра и коэффициента блокирования
                    # Броуновская диффузия частиц - в жидкой основе: вязкость без геля `mu_p` (без флага `gelation` это mu_o)
                    calc_velocities_h(i, j, S, T, _Um_r2, Wps, mu_p, h_sloy, Ur, new_h, new_Ur, new_Ub, dt)
                    # Обновление функции пор по размерам, скоростей потери порового объема, пористости, проницаемости
                    if asphaltenes:
                        calc_velocity_asph(i, j, S, T, _Um_r2, Wc, mu_p, new_kx)
                    # Взвесь и флокулы за шаг уходят и в осадок, и с оттоком нефти (грани и добывающая скважина):
                    # без учета оттока ограничитель подводом пропускал отрицательные доли тяжелых групп и флокул.
                    out_o = qo_out + max(-src_Qo[i, j], 0.0)
                    calc_qp_m_k_fi_2(i, j, S, Wps, Wc, m, k, fi, Ur, Ub, kx, integr_r2_fi0, integr_r4_fi0, rows[i], new_qp1, new_qp2, new_fi, new_k, new_m, new_kx, out_o, dt)

            # ---гидродинамика и перенос---
            # Скважины входят в уравнения наравне с перетоками через грани, поэтому делятся на те же поля нового слоя.
            cells_S_eq[i, j] += src_S[i, j]
            cells_Wp_eq[i, j] += src_Wp[i, j]
            cells_T_eq[i, j] += src_T[i, j]

            saturation_equation(i, j, S, m, cells_S_eq, new_m, new_s, dt)
            # Стоки q_p1, q_p2 - нового слоя: именно они дают убыль пористости m - new_m на этом шаге. С qp1, qp2
            # прошлого слоя парафин уходил из фазы на шаг позже, чем терялся поровый объем.
            if wax_components:
                components_equation(i, j, _paraphin, boundary_conditions, bc_Wc, rows[i, ROW_FO], p, T, m, S, new_m, new_s, Wc, new_Wc, Ws,
                                    new_Ws, src_Qo, new_qp1, new_qp2, new_wp, new_wps, new_Hl, Dep, kin, kx, new_kx, mu_p, dt)
            else:
                wp_equation(i, j, new_qp1, new_qp2, m, S, Wp, Wps, T, cells_Wp_eq, new_m, new_s, new_wp, new_wps, dt, _paraphin)
            if thermal_nonequilibrium:
                u_abs = (lam_o[i, j] + lam_w[i, j]) * grad_p[i, j]  # скорость фильтрации для теплообмена с породой
                psi = temperature_equation_ltne(i, j, T, T_0, m, S, C_o, C_w, C_f, C_p, Wo, Wp, Wps, new_wp, new_wps, Hl, new_Hl,
                                                cells_T_eq, _t, E_ff, new_t, new_m, new_s, kx, new_kx, u_abs, mu_o, kin, dt)
            else:
                psi = temperature_equation(i, j, T, T_0, m, S, C_o, C_w, C_f, C_p, Wo, Wp, Wps, new_wp, new_wps, Hl, new_Hl, cells_T_eq, _t, E_ff, new_t, new_m, new_s, dt)

            # Три ограничения на шаг по числу Куранта: по насыщенности, по переносу парафина и по температуре.
            # Осаждение шаг не ограничивает: его сток зажат подводом взвеси в `calc_qp_m_k_fi`.
            q_out = cells_Q_out[i, j]
            if q_out > 1e-30: # Скважинная часть добирается в `_calc_dt`.
                dt_cells = min(dt_cells, CFL_target * m[i, j] * volume / (max_dfw * q_out))
            if _paraphin and qo_out > 1e-30:
                dt_cells = min(dt_cells, CFL_target * m[i, j] * (1.0 - S[i, j]) * volume / qo_out)
            if t_out > 1e-30:
                dt_cells = min(dt_cells, CFL_target * psi * volume / t_out)

    return dt_cells


@njit(cache=True)
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


@njit(cache=True)
def _wells_loop(n_wells, wells, T, C_o, C_w, C_p, Wo, Wp, Wps, Hl, src_S, src_Wp, src_T, src_Qo):
    """Источники скважин в тех же единицах, что и перетоки через грани ячейки.

    Дебиты знаковые (q > 0 - закачка), поэтому источники складываются с перетоками как есть.
    Буферы не обнуляются: скважины неподвижны, их ячейки перезаписываются каждый шаг, а остальные
    так и остаются нулями с момента создания.
    """
    for w in range(n_wells):
        wl = wells[w]
        i, j = wl.i, wl.j
        src_S[i, j] = wl.q[1]
        src_Wp[i, j] = (Wp[i, j] + Wps[i, j]) * wl.q[0]
        src_T[i, j] = temperature_source(wells, w, T, C_o, C_w, C_p, Wo, Wp, Wps, Hl)
        src_Qo[i, j] = wl.q[0]  # перенос компонентов детального состава: у каждого свой множитель w_c


@njit(cache=True)
def _update_wells_data(n_wells, wells, p, S, mu_o, mu_w, dt):
    """Обновление дебита и обводненности скважин.

    Зовется после `calc_pressure`: дебиты берутся по давлению нового слоя и по коэффициентам
    продуктивности, уже ушедшим в матрицу, то есть неявно.
    """
    Q_oil = 0.0
    for w in range(n_wells):
        upd_q_and_eta(wells, w, p, S, mu_o, mu_w, dt)
        if not wells[w].is_injector:
            Q_oil -= wells[w].Q[0]  # у добывающей q < 0, а добыча положительна

    # Вычисление КИН
    return Q_oil / geological_reserves


@njit(parallel=True, cache=True)
def _swap_time_steps(_paraphin, qp1, new_qp1, qp2, new_qp2, k, new_k, m, new_m, S, new_s,
                     Wo, Wp, new_wp, Wps, new_wps, T, T_0, new_t,
                     fi, new_fi, h_sloy, new_h, Ur, new_Ur, Ub, new_Ub, mu_o, mu_w,
                     p, grad_p, Wc, new_Wc, Ws, new_Ws, Hl, new_Hl, mu_p, Phi, Dep, kx, new_kx,
                     dt):
    """Обновление полей данных на новом временном слое.

    С гелем (`gelation`) или зависимостью от давления вязкость нефти считает `Gel.oil_viscosity`: эффективная
    mu_o = mu_p/Phi, Phi релаксирует к равновесному с временем `gel_time` (множитель за шаг - `relax`). Через mu_o
    гель согласованно попадает в матрицу давления, продуктивность скважин и перетоки.
    """
    relax = math.exp(-dt / gel_time)
    for i in prange(Nx):
        for j in range(Ny):
            # Пересчет свойств флюидов из-за изменения температуры
            if not (gelation or pressure_viscosity):
                mu_o[i, j] = calc_mu_o(new_t[i, j], new_wps[i, j])
            mu_w[i, j] = calc_mu_w(new_t[i, j])

            S[i, j]   = new_s[i, j]
            T_0[i, j] = T[i, j]
            T[i, j]   = new_t[i, j]

            if _paraphin:
                was_clogging = Wps[i, j] > min_Wps_bound
                # Асфальтены меняли fi, если в этом шаге было сужение флокулами (`calc_qp_m_k_fi_2` читал Ua)
                asph_clogging = kx[i, j, KX_UA] < 0.0 if asphaltenes else False
                Wp[i, j]  = new_wp[i, j]
                Wps[i, j] = new_wps[i, j]
                Wo[i, j]  = 1.0 - new_wp[i, j] - new_wps[i, j]
                k[i, j]   = new_k[i, j]
                m[i, j]   = new_m[i, j]
                qp1[i, j] = new_qp1[i, j]
                qp2[i, j] = new_qp2[i, j]
                if wax_components:
                    Wc[i, j, :] = new_Wc[i, j, :]
                    Ws[i, j, :] = new_Ws[i, j, :]
                    Hl[i, j] = new_Hl[i, j]
                if asphaltenes or deposition_kinetics:  # поля механизмов, в том числе Ua и qpa
                    kx[i, j, :] = new_kx[i, j, :]

                if deposition_kinetics:
                    # Кинетическое ядро пишет fi и h каждой ячейки на каждом шаге
                    fi[i, j, :] = new_fi[i, j, :]
                    h_sloy[i, j, :] = new_h[i, j, :]
                # Ниже порога кольматации поля по радиусам не пишутся, копия была бы тождественной
                elif was_clogging:
                    fi[i, j, :]     = new_fi[i, j, :]
                    h_sloy[i, j, :] = new_h[i, j, :]
                    Ur[i, j, :]     = new_Ur[i, j, :]
                    Ub[i, j, :]     = new_Ub[i, j, :]
                elif asph_clogging:
                    fi[i, j, :] = new_fi[i, j, :]

            if gelation or pressure_viscosity:  # после обмена: множителю геля нужны m, S и fi нового слоя
                oil_viscosity(i, j, new_t[i, j], new_wps[i, j], p, S, m, fi, grad_p, Dep, Phi, mu_p, mu_o, relax)


def _calc_max_dfw(init_T, wells) -> float:
    """Максимум производной функции Баклея-Леверетта на рабочем диапазоне насыщенности.

    Именно эта величина задает предел устойчивости явной схемы по насыщенности: занизишь ее -
    `_calc_dt` разрешит слишком большой шаг. Функция зависит от отношения вязкостей, а оно - от
    температуры, и пласт по мере закачки остывает от init_T до температуры нагнетаемой воды.
    """
    # Диапазон температур расчета: от начальной пластовой до самой холодной закачиваемой воды
    temps = [init_T] + [well.T_inj for well in wells if well.is_injector]

    s = np.linspace(S_min, S_max, 2001)
    max_dfw = 0.0
    for t in np.linspace(min(temps), max(temps), 11):
        # Вязкость нефти зависит еще и от доли выпавшего парафина
        for w_ps in (0.0, init_Wp):
            f_w = np.array([Buckley_Leverett(x, calc_mu_w(t), calc_mu_o(t, w_ps)) for x in s])
            max_dfw = max(max_dfw, float(np.abs(np.gradient(f_w, s)).max()))

    # Гель и давление поднимают вязкость нефти выше calc_mu_o: гель - до mu/gel_mobility_min, давление -
    # множителем Баруса по диапазону забойных давлений. Без этих флагов цикл не выполняется.
    mults = []
    if pressure_viscosity:
        pressures = [init_p] + [well.p for well in wells if well.rate_control == 0]
        mults += [np.exp(alpha_p_visc * (pp - P_ref_wax)) for pp in (min(pressures), max(pressures))]
    if gelation:
        mults += [m_ / gel_mobility_min for m_ in (mults or [1.0])]
    for t in np.linspace(min(temps), max(temps), 11) if mults else ():
        for w_ps in (0.0, init_Wp):
            for mult in mults:
                f_w = np.array([Buckley_Leverett(x, calc_mu_w(t), calc_mu_o(t, w_ps) * mult) for x in s])
                max_dfw = max(max_dfw, float(np.abs(np.gradient(f_w, s)).max()))

    if wettability:
        # Смена смачиваемости меняет и форму функции Баклея-Леверетта: скан по доле нефтесмачиваемой поверхности
        from .utils.math_utils import pf_o_mix, pf_w_mix
        from .kinetics_params import DEFAULTS
        ow_S_min, ow_S_max, ow_n_o, ow_n_w = (DEFAULTS[k] for k in ('OW_S_MIN', 'OW_S_MAX', 'OW_N_O', 'OW_N_W'))
        s_scan = np.linspace(min(S_min, ow_S_min), max(S_max, ow_S_max), 2001)
        for t in np.linspace(min(temps), max(temps), 11):
            mu_w_t, mu_o_t = calc_mu_w(t), calc_mu_o(t, 0.0)
            for omega in (0.25, 0.5, 0.75, 1.0):
                lw = np.array([pf_w_mix(x, omega, ow_S_min, ow_S_max, ow_n_w) for x in s_scan]) / mu_w_t
                lo = np.array([pf_o_mix(x, omega, ow_S_min, ow_S_max, ow_n_o) for x in s_scan]) / mu_o_t
                f_w = lw / np.maximum(lw + lo, 1e-300)
                max_dfw = max(max_dfw, float(np.abs(np.gradient(f_w, s_scan)).max()))

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
        mode = 'задан дебит' if solver.wells[i].rate_control == 1 else 'задано P_заб'
        solver.logger.info(f"Скважина {solver._wells_buffer[i]['name']} ({mode}): "
                           f"q_o={solver.wells[i].q[0] * day_to_sec} м^3/сут  "
                           f"q_w={solver.wells[i].q[1] * day_to_sec} м^3/сут  "
                           f"P_заб={solver.wells[i].p / bar_to_pa} бар")

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
