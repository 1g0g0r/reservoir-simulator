"""Числовые параметры кинетики осаждения и поля кинетических моделей: именованные индексы.

Флаги моделей (`wax_kinetics`, `entrainment`, ...) - константы `constants.py`, numba сворачивает по ним
ветки при компиляции. А числовые параметры тех же моделей живут в runtime-векторе `solver.kin`: ядра
получают его аргументом. Так калибровка по опытам (`experiments/`) меняет константы скоростей в уже
скомпилированном процессе - без копии пакета и 40 с компиляции на каждую оценку целевой функции.
Начальные значения берутся из `constants.py` (там же - источник каждого значения).

Поля кинетических моделей (равновесная взвесь по группам, скорости новых стоков, адсорбированные
количества, объем гель-отложения, температура породы) собраны в один массив `kx` формы (Nx, Ny, NKX),
а не в отдельные массивы: иначе каждое поле добавляло бы по два аргумента во все ядра шага.
Индексы - константы ниже; `kx` - текущий слой, `new_kx` - новый, обмен в `_swap_time_steps`.
"""
import numpy as np

from paraphin import constants as c
from paraphin.oil_composition import N_W

# --- Вектор параметров solver.kin ---------------------------------------------------------------------------
_PARAMS = (
    # 1. кинетика кристаллизации
    ('K_CRYST', c.k_cryst), ('K_DISS', c.k_diss), ('K_WALL', c.k_wall),
    # 2. перенос к стенке
    ('SHEAR_DISP', c.shear_disp), ('GRAV_EFF', c.grav_eff),
    # 3. вынос
    ('ENT_RATE', c.ent_rate), ('ENT_TAU', c.ent_tau),
    # 4-5. агрегация и снежный ком
    ('AGG_D0', c.agg_d0), ('AGG_DF', c.agg_df), ('AGG_W', c.agg_W), ('SNOW_A', c.snow_a),
    # 6-7. адсорбция и смачиваемость
    ('ADS_GMAX', c.ads_gmax), ('ADS_K', c.ads_K), ('ADS_DH', c.ads_dH), ('ADS_T_REF', c.ads_T_ref),
    ('ADS_RATE', c.ads_rate), ('ADS_RESIN', c.ads_resin),
    ('OW_S_MIN', c.ow_S_min), ('OW_S_MAX', c.ow_S_max), ('OW_N_O', c.ow_n_o), ('OW_N_W', c.ow_n_w),
    # 8. старение гель-отложения
    ('AGE_C0', c.age_c0), ('AGE_CMAX', c.age_cmax), ('AGE_RATE', c.age_rate),
    # 9-10. глубинная фильтрация и k(phi, sigma)
    ('FILT_KD', c.filt_kd), ('FILT_KPL', c.filt_kpl), ('FILT_KE', c.filt_ke), ('FILT_UCR', c.filt_ucr),
    ('PERM_N', c.perm_n), ('PERM_BETA', c.perm_beta), ('PERM_SMAX', c.perm_smax), ('PERM_GAMMA', c.perm_gamma),
    ('PERM_ALPHA', c.perm_alpha),
    # 11. тепловое неравновесие
    ('LTNE_DG', c.ltne_dg), ('LTNE_DM', c.ltne_Dm),
)
KIN_NAMES = tuple(name for name, _ in _PARAMS)
(K_CRYST, K_DISS, K_WALL, SHEAR_DISP, GRAV_EFF, ENT_RATE, ENT_TAU, AGG_D0, AGG_DF, AGG_W, SNOW_A,
 ADS_GMAX, ADS_K, ADS_DH, ADS_T_REF, ADS_RATE, ADS_RESIN, OW_S_MIN, OW_S_MAX, OW_N_O, OW_N_W,
 AGE_C0, AGE_CMAX, AGE_RATE, FILT_KD, FILT_KPL, FILT_KE, FILT_UCR, PERM_N, PERM_BETA, PERM_SMAX, PERM_GAMMA,
 PERM_ALPHA, LTNE_DG, LTNE_DM) = range(len(_PARAMS))
NK = len(_PARAMS)


def default_kin() -> np.ndarray:
    """Вектор параметров со значениями из `constants.py`."""
    return np.array([value for _, value in _PARAMS], c.data_type)


def kin_index(name: str) -> int:
    """Индекс параметра по имени (для калибровки: `solver.kin[kin_index('K_WALL')] = ...`)."""
    return KIN_NAMES.index(name.upper())


# --- Поля kx -----------------------------------------------------------------------------------------------
KX_WEQ = 0            # [KX_WEQ, +N_W): равновесная взвесь групп на слое (`wax_kinetics`), [-]
KX_WSH = N_W          # [KX_WSH, +N_W): доли групп в стеночной кристаллизации этого шага, [-]
KX_GSH = 2 * N_W      # [KX_GSH, +N_W): доли групп в старении гель-отложения этого шага, [-]
KX_QW = 3 * N_W       # скорость потери порового объема на стеночную кристаллизацию, [1/с]
KX_QG = 3 * N_W + 1   # то же на старение гель-отложения, [1/с]
KX_QADA = 3 * N_W + 2  # то же на адсорбцию асфальтенов, [1/с]
KX_QADR = 3 * N_W + 3  # то же на адсорбцию смол, [1/с]
KX_GA = 3 * N_W + 4   # адсорбированные асфальтены, [кг/м^3 породы]
KX_GR = 3 * N_W + 5   # адсорбированные смолы, [кг/м^3 породы]
KX_VGEL = 3 * N_W + 6  # объем гель-отложения парафина в проводящих каналах, [м^3/м^3 породы]
KX_TS = 3 * N_W + 7   # температура породы (`thermal_nonequilibrium`), [C]
KX_GMAX = 3 * N_W + 8  # предельная адсорбция асфальтенов G_max = ads_gmax*a_v (для смачиваемости), [кг/м^3 породы]
NKX = 3 * N_W + 9
