"""Модуль содержит класс объектов скважин"""
import numpy as np
from numba import njit, int32, float32, float64
from numba.experimental import jitclass
from numba.types import Array

from paraphin.constants import data_type, h, _re, Nx
from .math_utils import pf_w, pf_o, Buckley_Leverett

if data_type == np.float32:
    data_type_nb = float32
else:
    data_type_nb = float64

well_spec = [
    ('i', int32),
    ('j', int32),
    ('idx_rhs', int32),
    ('rw', data_type_nb),
    ('p', data_type_nb),
    ('T', data_type_nb),
    ('is_injector', int32),
    ('productivity_mult', data_type_nb),
    ('q', Array(data_type_nb, 1, 'C')),  # Вектор размера 3
    ('Q', Array(data_type_nb, 1, 'C')),  # Вектор размера 3
    ('eta', data_type_nb),
    ('dp', data_type_nb),
]


@jitclass(well_spec)
class WellStruct:
    def __init__(self, i, j, p, T, rw, is_injector, mult):
        # Инициализация полей
        self.i = i
        self.j = j
        self.idx_rhs = 0
        self.rw = 0.0
        self.p = p
        self.T = T
        self.is_injector = is_injector
        self.productivity_mult = 2.0 * np.pi * h / np.log(_re / rw) * mult
        # Инициализация массивов размером 3
        self.q = np.zeros(3, dtype=np.float64)
        self.Q = np.zeros(3, dtype=np.float64)
        self.eta = 0.0
        self.dp = 0.0


@njit(cache=True)
def upd_q_and_eta(well, p, S, k, mu_o, mu_w, dt) -> WellStruct:
    """Вычисление дебета скважины."""
    well.dp = p[well.i, well.j] - well.p
    mult = well.dp * well.productivity_mult * k[well.i, well.j]

    if well.is_injector == 1:
        well.q[0] = 0.0
        well.q[1] = mult / mu_w[well.i, well.j]
        well.eta = 1.0

    else:
        well.q[0] = mult * pf_o(S[well.i, well.j]) / mu_o[well.i, well.j]
        well.q[1] = mult * pf_w(S[well.i, well.j]) / mu_w[well.i, well.j]
        well.eta = Buckley_Leverett(S[well.i, well.j], mu_w[well.i, well.j], mu_o[well.i, well.j])

    well.q[2] = well.q[0] + well.q[1]

    well.Q[0] += well.q[0] * dt
    well.Q[1] += well.q[1] * dt
    well.Q[2] += well.q[2] * dt

    return well


@njit(cache=True)
def calc_well_mult(well, S, k, mu_o, mu_w) -> float:
    """Вычисление множителя дебета скважины."""
    ret = 0.0
    mult = well.productivity_mult * k[well.i, well.j]

    if well.is_injector == 1:
        ret = mult / mu_w[well.i, well.j]
    else:
        ret = mult * (pf_o(S[well.i, well.j]) / mu_o[well.i, well.j] +
                      pf_w(S[well.i, well.j]) / mu_w[well.i, well.j])

    return ret


def preprocess_matrix_and_wells(wells, wells_buffer):
    """Раскладка скважин из буфера в массив с вычислением индекса неизвестной.

    Профиль разреженности матрицы больше не нужен: уравнение давления собирается сразу в три
    диагонали ленты (`equations/Pressure.py`), поэтому скважина правит `diag[idx]` и `rhs[idx]`
    по одному и тому же индексу `idx = i + j*Nx`.
    """
    for i in range(len(wells)):
        wells[i] = wells_buffer[i]['well']
        wells[i].idx_rhs = wells[i].i + wells[i].j * Nx

    return wells
