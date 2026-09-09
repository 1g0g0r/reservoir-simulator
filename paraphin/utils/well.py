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
    ('prod_mult', data_type_nb),
    ('q', Array(data_type_nb, 1, 'C')),    # Дебит: (q_o, q_w, q_t), [м^3/с]
    ('Q', Array(data_type_nb, 1, 'C')),    # Накопленный дебит: (Q_o, Q_w, Q_t), [м^3]
    ('prod', Array(data_type_nb, 1, 'C')), # Коэффициенты продуктивности: (J_o, J_w, J_t),  [м^3/(Па*с)]
    ('eta', data_type_nb),
    ('rate_control', int32),   # 1 - задан дебит, забойное давление считается; 0 - наоборот
]


@jitclass(well_spec)
class WellStruct:
    def __init__(self, i, j, p, q_set, rate_control, T, rw, is_injector, mult):
        # Инициализация полей
        self.i = i
        self.j = j
        self.idx_rhs = 0
        self.rw = rw
        self.p = p
        self.rate_control = rate_control
        self.T = T
        self.is_injector = is_injector
        self.prod_mult = 2.0 * np.pi * h / np.log(_re / rw) * mult
        self.q = np.array([0.0, 0.0, q_set], dtype=data_type)
        self.Q = np.zeros(3, dtype=data_type)
        self.prod = np.zeros(3, dtype=data_type)
        self.eta = 0.0


@njit(cache=True)
def calc_well_prod(well, S, k, mu_o, mu_w) -> WellStruct:
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
    mult = well.prod_mult * k[well.i, well.j]

    if well.is_injector == 1:
        well.prod[0] = 0.0
        well.prod[1] = mult / mu_w[well.i, well.j]
    else:
        well.prod[0] = mult * pf_o(S[well.i, well.j]) / mu_o[well.i, well.j]
        well.prod[1] = mult * pf_w(S[well.i, well.j]) / mu_w[well.i, well.j]

    well.prod[2] = well.prod[0] + well.prod[1]

    return well


@njit(cache=True)
def upd_q_and_eta(well, p, S, mu_o, mu_w, dt) -> WellStruct:
    """Дебиты скважины по неявной формуле Писмана и обводненность.
        q_a = prod_a * (P_w - P_i^(t+1))

    Давление берется с нового слоя, а `prod_a` - коэффициенты, что уже ушли в матрицу (`calc_well_prod`).
    Поэтому дебиты согласованы, сумма закачки и отбора равна нулю с точностью решателя, а не схемы.

    В режиме заданного дебита (`rate_control`) неизвестной становится забойное давление:
    перепад берется из формулы Писмана, `P_w - P_i = q / prod[2]`, и `well.p` пересчитывается на каждом шаге.
    Суммарный дебит при этом равен заданному тождественно, а по фазам он делится в тех же долях подвижности - ровно так,
    как источник вошел в матрицу. Знак такой же, как у источников в уравнениях баланса: q > 0 - закачка, q < 0 - отбор.
    """
    if well.rate_control == 1:
        # Нулевая суммарная подвижность. Матрица в такой ячейке вырождена и решатель сообщит об этом.
        dp = well.q[2] / well.prod[2] if well.prod[2] > 0.0 else 0.0
        well.p = p[well.i, well.j] + dp
    else:
        dp = well.p - p[well.i, well.j]

    well.q[0] = well.prod[0] * dp
    well.q[1] = well.prod[1] * dp
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
