"""Скважины: структурный массив `wells[n_wells]` с dtype `WELL`.

Структурный dtype кешируется и компилируется так же быстро, как 2D float64 со столбцами-индексами (замер в
docs/PERFORMANCE_FINDINGS.md, «Скважины»). Дебиты, накопленные дебиты и продуктивность по фазам - скалярные поля
(`q_o`, `q_w`, `q_t`, `Q_o` ..., `J_t`), а не подмассивы: векторное выражение над подмассивом записи стоило +2.4 с
компиляции, а поэлементный доступ по номеру фазы был нечитаем. В ядра передавать обычный ndarray, а не `np.recarray`:
подкласс numba типизирует медленным путем, +3 мкс на вызов. Python-коду (выгрузка, тесты, критерий останова) тот же
буфер виден как `np.recarray`: `solver.wells[k].q_t`, `.p`.
"""
import numpy as np
from numba import njit

from paraphin.constants import h, _re, wettability
from paraphin.layout import P0, PX
from .math_utils import pf_w, pf_o, Buckley_Leverett

WELL = np.dtype([
    ('i', np.int64), ('j', np.int64),  # ячейка скважины
    ('idx_rhs', np.int64),             # индекс неизвестной в СЛАУ давления idx = P0 + i + j*PX (`layout`)
    ('rw', np.float64),                # радиус скважины, [м]
    ('p', np.float64),                 # забойное давление, [Па]: задано или, в режиме заданного дебита, считается
    ('T_inj', np.float64),             # температура закачки, [С] (не 'T': у записи numpy это транспонирование)
    ('is_injector', np.bool_),         # нагнетательная (иначе добывающая)
    ('rate_control', np.bool_),        # задан дебит, забойное давление считается (иначе наоборот)
    ('prod_mult', np.float64),         # множитель Писмана 2*pi*h/ln(r_o/r_w)*mult, [м]
    ('eta', np.float64),               # обводненность, [-]
    # дебит нефти, воды и суммарный, [м^3/с]; q > 0 - закачка, q < 0 - отбор
    ('q_o', np.float64), ('q_w', np.float64), ('q_t', np.float64),
    ('Q_o', np.float64), ('Q_w', np.float64), ('Q_t', np.float64),  # накопленный дебит, [м^3]
    ('J_o', np.float64), ('J_w', np.float64), ('J_t', np.float64),  # продуктивность, [м^3/(Па*с)] (не 'prod': метод записи)
], align=True)


@njit(cache=True)
def _prod_mult(rw, mult):
    """Множитель Писмана, [м]."""
    return 2.0 * np.pi * h / np.log(_re / rw) * mult


def new_well(i, j, p, q_set, rate_control, T, rw, is_injector, mult) -> np.ndarray:
    """Запись новой скважины; индекс неизвестной ставит `preprocess_wells`."""
    well = np.zeros((), WELL)
    well['i'], well['j'], well['rw'], well['p'], well['T_inj'] = i, j, rw, p, T
    well['is_injector'], well['rate_control'] = is_injector, rate_control
    well['prod_mult'] = _prod_mult(rw, mult)
    well['q_t'] = q_set
    return well


@njit(cache=True)
def calc_well_prod(wells, w, S, k, mu_o, mu_w, lam_o, lam_w) -> None:
    """Коэффициенты продуктивности Писмана скважины w по фазам, [м^3/(Па*с)]:

        prod_a = 2*pi*k^(t+1)*h/ln(r_o/r_w) * (f_a/mu_a)^t,   q_a = prod_a * (P_w - P_i^(t+1)).

    Считаются по слою t до решения СЛАУ и остаются в записи скважины: матрица берет их сумму на
    диагональ и сумму*P_w в правую часть (`equations/Pressure.py`), а `upd_q_and_eta` после
    решения умножает те же коэффициенты на перепад. Один источник формулы вместо двух - иначе
    матрица и дебиты расходятся молча, а вместе с ними разъезжаются закачка и отбор.

    Проницаемость берется текущая: кольматация призабойной зоны - основной эффект задачи.
    В нагнетательную идет только вода, поэтому ее приемистость считается по полной подвижности воды, без ОФП.
    """
    wl = wells[w]
    i, j = wl.i, wl.j
    mult = wl.prod_mult * k[i, j]

    if wl.is_injector:
        wl.J_o = 0.0
        wl.J_w = mult / mu_w[i, j]
    elif wettability:
        # ОФП со сменой смачиваемости уже в подвижностях ячейки (`calc_mobility_w`): lam = k*pf/mu
        wl.J_o = wl.prod_mult * lam_o[i, j]
        wl.J_w = wl.prod_mult * lam_w[i, j]
    else:
        wl.J_o = mult * pf_o(S[i, j]) / mu_o[i, j]
        wl.J_w = mult * pf_w(S[i, j]) / mu_w[i, j]

    wl.J_t = wl.J_o + wl.J_w


@njit(cache=True)
def upd_q_and_eta(wells, w, p, S, mu_o, mu_w, dt) -> None:
    """Дебиты скважины w по неявной формуле Писмана и обводненность.

        q_a = prod_a * (P_w - P_i^(t+1))

    Давление берется с нового слоя, а `J_a` - коэффициенты, что уже ушли в матрицу (`calc_well_prod`).
    Поэтому дебиты согласованы, сумма закачки и отбора равна нулю с точностью решателя, а не схемы.

    В режиме заданного дебита неизвестной становится забойное давление: перепад берется из формулы Писмана,
    `P_w - P_i = q / J[2]`, и забойное давление пересчитывается на каждом шаге. Суммарный дебит при этом
    равен заданному тождественно, а по фазам он делится в тех же долях подвижности - ровно так, как источник
    вошел в матрицу. Знак такой же, как у источников в уравнениях баланса: q > 0 - закачка, q < 0 - отбор.
    """
    wl = wells[w]
    i, j = wl.i, wl.j
    if wl.rate_control:
        # Нулевая суммарная подвижность. Матрица в такой ячейке вырождена и решатель сообщит об этом.
        dp = wl.q_t / wl.J_t if wl.J_t > 0.0 else 0.0
        wl.p = p[i, j] + dp
    else:
        dp = wl.p - p[i, j]

    wl.q_o = wl.J_o * dp
    wl.q_w = wl.J_w * dp
    wl.q_t = wl.q_o + wl.q_w

    if wl.is_injector:
        wl.eta = 1.0
    else:
        wl.eta = Buckley_Leverett(S[i, j], mu_w[i, j], mu_o[i, j])

    wl.Q_o += wl.q_o * dt

    wl.Q_w += wl.q_w * dt

    wl.Q_t += wl.q_t * dt


def preprocess_wells(wells_buffer) -> np.ndarray:
    """Массив скважин из буфера `add_well` с индексом неизвестной.

    Уравнение давления собирается сразу в диагонали ленты (`equations/Pressure.py`),
    поэтому скважина правит `diag[idx]` и `rhs[idx]` по одному и тому же индексу `idx = P0 + i + j*PX` (`layout`).
    """
    wells = np.array([item['well'] for item in wells_buffer], WELL)
    wells['idx_rhs'] = P0 + wells['i'] + wells['j'] * PX
    return wells
