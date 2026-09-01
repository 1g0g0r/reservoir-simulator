"""Проверки решателя уравнения давления: ориентация поля, принцип максимума и сам решатель.

Неизвестная в СЛАУ нумеруется как idx = i + j*Nx (быстрый индекс - i), поэтому развернутое решение
нужно транспонировать, иначе поле давления оказывается зеркальным относительно главной диагонали по
отношению ко всем остальным полям. На штатной постановке (обе скважины на главной диагонали,
однородная проницаемость, квадратная сетка) точное решение симметрично и ошибка не видна, поэтому
скважины здесь ставятся несимметрично.
"""
import numpy as np

from paraphin import N
from paraphin.constants import Nx, Ny, Pw, Po, Twater, rw, init_S, init_k, init_m, init_p, init_T, data_type
from paraphin.equations import calc_pressure
from paraphin.utils import WellStruct, preprocess_matrix_and_wells, calc_mu_o, calc_mu_w, calc_mobility

INJ = (3, 7)
PROD = (Nx - 5, Ny - 10)


def _solve_pressure():
    """Однократное решение уравнения давления на однородном пласте с двумя скважинами."""
    injector = WellStruct(i=INJ[0], j=INJ[1], p=Pw, T=Twater, rw=rw, is_injector=1, mult=0.25)
    producer = WellStruct(i=PROD[0], j=PROD[1], p=Po, T=0.0, rw=rw, is_injector=0, mult=0.25)
    buffer = [{'well': injector, 'name': 'inj'}, {'well': producer, 'name': 'prod'}]
    wells = preprocess_matrix_and_wells([WellStruct, WellStruct], buffer)

    def field(value):
        return np.full((Nx, Ny), value, data_type)

    def vec():
        return np.zeros(N, data_type)

    k, S = field(init_k), field(init_S)
    mu_o, mu_w = field(calc_mu_o(init_T)), field(calc_mu_w(init_T))
    lam_o, lam_w = field(0.0), field(0.0)
    calc_mobility(k, S, mu_o, mu_w, lam_o, lam_w)

    diag, ex, ey, rhs = vec(), vec(), vec(), vec()
    p, _ = calc_pressure(field(1.0), field(1.0), field(init_m), field(init_m), k, S, S, mu_o, mu_w,
                         lam_o, lam_w, wells, diag, ex, ey, rhs,
                         np.zeros((N, Nx + 1), data_type), np.full(N, init_p, data_type),
                         vec(), vec(), vec(), vec(),
                         np.zeros((4, 3, 2), data_type), 0, 4320.0)

    return p, diag, ex, ey, rhs


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


def test_band_solver_matches_superlu():
    """Ленточный Холецкий обязан давать то же, что и SuperLU на той же матрице.

    Сборка идет сразу в три диагонали, минуя CSC, поэтому независимая проверка нужна: разложение
    делается без выбора главного элемента и опирается на положительную определенность матрицы.
    """
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    p, diag, ex, ey, rhs = _solve_pressure()
    n = N
    M = (np.diag(diag) + np.diag(ex[:n - 1], 1) + np.diag(ex[:n - 1], -1)
         + np.diag(ey[:n - Nx], Nx) + np.diag(ey[:n - Nx], -Nx))
    x_ref = splu(csc_matrix(M)).solve(rhs)

    err = np.linalg.norm(p.T.ravel() - x_ref) / np.linalg.norm(x_ref)
    assert err < 1e-10, f'ленточный решатель разошелся с SuperLU: относительная ошибка {err:.2e}'


if __name__ == '__main__':
    test_pressure_orientation()
    test_pressure_maximum_principle()
    test_band_solver_matches_superlu()
    print('OK')
