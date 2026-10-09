"""Проверки решателя уравнения давления: ориентация поля, принцип максимума и сам решатель.

Неизвестная в СЛАУ нумеруется как idx = i + j*Nx (быстрый индекс - i), поэтому развернутое решение
нужно транспонировать, иначе поле давления оказывается зеркальным относительно главной диагонали по
отношению ко всем остальным полям. На штатной постановке (обе скважины на главной диагонали,
однородная проницаемость, квадратная сетка) точное решение симметрично и ошибка не видна, поэтому
скважины здесь ставятся несимметрично.
"""
import numpy as np

from paraphin.geometry import N
from paraphin.constants import (Nx, Ny, Pw, Po, Twater, rw, init_S, init_k, init_m, init_p, init_T, init_Wp, init_Wps,
                                data_type, p_guess_m)
from paraphin.layout import NP, PX
from paraphin.equations import calc_pressure
from paraphin.utils import BC, new_well, preprocess_wells, calc_mu_o, calc_mu_w, calc_mobility
from paraphin.utils.math_utils import MG_ROWS, MG_TOTAL, MG_COARSE, MG_COARSE_KD, P_STATE, solve_mg_system
from paraphin.utils.math_utils.mg_solver import mg_pad, mg_unpad

INJ = (3, 7)
PROD = (Nx - 5, Ny - 10)


def _solver_buffers():
    """Буферы решателя, как в `Solver.__init__`: уровни многосеточного, фактор грубого уровня, состояние."""
    return (np.zeros((MG_ROWS, MG_TOTAL), data_type), np.zeros((MG_COARSE, MG_COARSE_KD + 1), data_type),
            np.zeros(P_STATE, data_type))


def _solve_pressure():
    """Однократное решение уравнения давления на однородном пласте с двумя скважинами."""
    injector = new_well(i=INJ[0], j=INJ[1], p=Pw, q_set=0.0, rate_control=0, T=Twater, rw=rw,
                        is_injector=1, mult=0.25)
    producer = new_well(i=PROD[0], j=PROD[1], p=Po, q_set=0.0, rate_control=0, T=0.0, rw=rw,
                        is_injector=0, mult=0.25)
    buffer = [{'well': injector, 'name': 'inj'}, {'well': producer, 'name': 'prod'}]
    wells = preprocess_wells(buffer)

    def field(value):
        return np.full((Nx, Ny), value, data_type)

    def vec():
        return np.zeros(NP, data_type)

    k, S = field(init_k), field(init_S)
    mu_o, mu_w = field(calc_mu_o(init_T, init_Wps)), field(calc_mu_w(init_T))
    lam_o, lam_w, lam_h = field(0.0), field(0.0), field(0.0)
    calc_mobility(k, S, field(init_m), field(init_Wp),
                  field(init_Wps), mu_o, mu_w, lam_o, lam_w, lam_h)

    mg_buf, mg_wc, state = _solver_buffers()
    diag, ex, ey, rhs = mg_buf[0, :NP], mg_buf[1, :NP], mg_buf[2, :NP], vec()  # матрица - мелкий уровень решателя
    p = np.zeros((Nx, Ny), data_type)
    x0 = np.full(NP, init_p, data_type)
    calc_pressure(k, S, mu_o, mu_w, lam_o, lam_w, wells, diag, ex, ey, rhs, mg_buf, mg_wc, state, x0,
                  np.tile(x0, (p_guess_m, 1)), vec(), vec(), np.zeros((4, 4), BC), p)

    # матрица в раскладке подряд (idx = i + j*Nx) - для сравнения с SuperLU
    unpad = lambda v: v.reshape(Ny + 2, PX)[1:-1, 1:-1].ravel()
    return p, unpad(diag), unpad(ex), unpad(ey), unpad(rhs)


def test_pressure_orientation():
    """Экстремумы давления обязаны попадать ровно в те ячейки, где заданы скважины."""
    p, _, _, _, _ = _solve_pressure()

    assert np.unravel_index(np.argmax(p), p.shape) == INJ, 'максимум давления не в ячейке нагнетательной'
    assert np.unravel_index(np.argmin(p), p.shape) == PROD, 'минимум давления не в ячейке добывающей'


def test_pressure_maximum_principle():
    """Без источников в правой части давление не выходит за диапазон забойных давлений скважин."""
    p, _, _, _, _ = _solve_pressure()

    assert np.all(np.isfinite(p)), 'в поле давления появились NaN/Inf'
    assert p.min() >= Po - 1e-6, f'давление ниже забойного давления добывающей: {p.min()}'
    assert p.max() <= Pw + 1e-6, f'давление выше забойного давления нагнетательной: {p.max()}'


def test_solver_matches_superlu():
    """Многосеточный PCG обязан давать то же, что SuperLU: и в расчете (`calc_pressure`), и со старта издалека.

    Сборка идет сразу в три диагонали, минуя CSC, поэтому независимая проверка нужна: PCG останавливается по невязке
    `p_pcg_rtol`, а предобуславливатель опирается на положительную определенность матрицы.
    """
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    p, diag, ex, ey, rhs = _solve_pressure()
    n = N
    M = (np.diag(diag) + np.diag(ex[:n - 1], 1) + np.diag(ex[:n - 1], -1)
         + np.diag(ey[:n - Nx], Nx) + np.diag(ey[:n - Nx], -Nx))
    x_ref = splu(csc_matrix(M)).solve(rhs)
    rel = lambda x: np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)

    assert rel(p.T.ravel()) < 1e-8, f'решатель расчета разошелся с SuperLU: {rel(p.T.ravel()):.2e}'
    mg_buf, mg_wc, state = _solver_buffers()
    rhs_p, x_p = np.zeros(MG_TOTAL), np.zeros(MG_TOTAL)
    for row, v in enumerate((diag, ex, ey)):
        mg_pad(v, mg_buf[row])
    mg_pad(rhs, rhs_p)
    mg_pad(np.full(n, init_p), x_p)
    it = solve_mg_system(rhs_p, x_p, np.zeros(MG_TOTAL), np.zeros(MG_TOTAL), mg_buf, mg_wc, state)
    x = mg_unpad(x_p)
    assert rel(x) < 1e-8, f'многосеточный PCG разошелся с SuperLU: {rel(x):.2e} за {it} итераций'


if __name__ == '__main__':
    test_pressure_orientation()
    test_pressure_maximum_principle()
    test_solver_matches_superlu()
    print('OK')
