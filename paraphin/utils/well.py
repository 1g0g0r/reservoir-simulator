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
    ('prod_o', data_type_nb),  # Коэффициенты продуктивности Писмана по фазам, [м^3/(Па*с)]
    ('prod_w', data_type_nb),
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
        self.prod_o = 0.0
        self.prod_w = 0.0


@njit(cache=True)
def calc_well_pi(well, S, k, mu_o, mu_w) -> WellStruct:
    """Коэффициенты продуктивности Писмана по фазам, [м^3/(Па*с)]:

        prod_a = 2*pi*k^(t+1)*h/ln(r_o/r_w) * (f_a/mu_a)^t,   q_a = prod_a * (P_w - P_i^(t+1)).

    Считаются по слою t до решения СЛАУ и остаются на скважине: матрица берет их сумму на
    диагональ и сумму*P_w в правую часть (`equations/Pressure.py`), а `upd_q_and_eta` после
    решения умножает те же коэффициенты на перепад. Один источник формулы вместо двух - иначе
    матрица и дебиты расходятся молча, а вместе с ними разъезжаются закачка и отбор.

    Проницаемость берется текущая: кольматация призабойной зоны - основной эффект задачи.
    В нагнетательную идет только вода, поэтому ее приемистость считается по полной подвижности
    воды, без ОФП.
    """
    mult = well.productivity_mult * k[well.i, well.j]

    if well.is_injector == 1:
        well.prod_o = 0.0
        well.prod_w = mult / mu_w[well.i, well.j]
    else:
        well.prod_o = mult * pf_o(S[well.i, well.j]) / mu_o[well.i, well.j]
        well.prod_w = mult * pf_w(S[well.i, well.j]) / mu_w[well.i, well.j]

    return well


@njit(cache=True)
def upd_q_and_eta(well, p, S, mu_o, mu_w, dt) -> WellStruct:
    """Дебиты скважины по неявной формуле Писмана и обводненность.

        q_a = prod_a * (P_w - P_i^(t+1))

    Давление берется с нового слоя, а `prod_a` - те самые коэффициенты, что уже ушли в матрицу
    (`calc_well_pi`). Поэтому дебиты согласованы с решенной системой тождественно, и сумма
    закачки и отбора равна нулю с точностью решателя, а не схемы.

    Знак такой же, как у источников в уравнениях баланса: q > 0 - закачка, q < 0 - отбор.
    """
    dp = well.p - p[well.i, well.j]

    well.q[0] = well.prod_o * dp
    well.q[1] = well.prod_w * dp
    well.q[2] = well.q[0] + well.q[1]

    if well.is_injector == 1:
        well.eta = 1.0
    else:
        well.eta = Buckley_Leverett(S[well.i, well.j], mu_w[well.i, well.j], mu_o[well.i, well.j])

    well.Q[0] += well.q[0] * dt
    well.Q[1] += well.q[1] * dt
    well.Q[2] += well.q[2] * dt

    return well


def preprocess_wells(wells_buffer):
    """Раскладка скважин из буфера в список с вычислением индекса неизвестной.

    Профиль разреженности матрицы не нужен: уравнение давления собирается сразу в
    диагонали ленты (`equations/Pressure.py`), поэтому скважина правит `diag[idx]` и `rhs[idx]`
    по одному и тому же индексу `idx = i + j*Nx`.
    """
    wells = [item['well'] for item in wells_buffer]
    for well in wells:
        well.idx_rhs = well.i + well.j * Nx

    return wells
