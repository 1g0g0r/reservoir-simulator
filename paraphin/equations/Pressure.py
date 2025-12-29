"""Решение уравнения давления: сборка матрицы (МКО и решение СЛАУ)."""
import numpy as np
from numba import njit, prange
from scipy.sparse.linalg._dsolve.linsolve import _superlu
from sparse_numba.sparse_superlu.superlu_numba_interface import superlu_solve_csc
from sparse_numba.sparse_umfpack.umfpack_numba_interface import umfpack_solve_csc

from paraphin import N, NN
from paraphin.constants import Nx, Ny, hx, hy, dt, volume, h
from paraphin.utils import mid_Ko_Kw, apply_bc, get_bound, calc_well_mult


@njit
def calc_pressure(Wo, Wo_0, m, m_0, k, S, S_0, mu_o, mu_w, wells, rows_indices, cols_ptr, sort_mask, data, rhs, boundary_condition):
    """Сборка матрицы и решение СЛАУ уравнения давления (МКО)

    Parameters
    ----------
    Wo: numpy.ndarray((Nx, Ny)
        Массовая доля масляного компонента в нефти, [-]
    Wo_0: numpy.ndarray(Nx, Ny)
        Массовая доля масляного компонента в нефти на прошлом временном слое, [-]
    m: numpy.ndarray(Nx, Ny)
        Пористость, [-]
    m_0: numpy.ndarray(Nx, Ny)
        Пористость на прошлом временном слое, [-]
    k: numpy.ndarray(Nx, Ny)
        Проницаемость, [м^2]
    S: numpy.ndarray(Nx, Ny)
        Водонасыщенность, [-]
    S_0: numpy.ndarray(Nx, Ny)
        Водонасыщенность на прошлом временном слое, [-]
    mu_o: numpy.ndarray(Nx, Ny)
        Вязкость нефти, [Па*с]
    mu_w: numpy.ndarray(Nx, Ny)
        Вязкость воды, [Па*с]
    wells: numpy.ndarray(n_wells)
        Массив скважин
    rows_indices: ti.ndarray(NN)
        Массив строк разреженной матрицы
    cols_ptr: ti.ndarray(NN)
        Массив столбцов разреженной матрицы
    sort_mask: ti.ndarray(NN)
        Массив перестановки элементов матрицы из стандартной расположения в csc формат
    boundary_condition: ti.field(4, 3, 2)
        Граничные условия: Граница -> Поле -> Тип, Значение
    """
    _fill_matrix_and_rhs(Wo, Wo_0, m, m_0, k, S, S_0, mu_o, mu_w, data, rhs, boundary_condition)
    _adding_wells(wells, S, k, mu_o, mu_w, data, rhs)

    # TODO попробовать вызывать сразу компилированный модуль без обертки
    solution, _ = umfpack_solve_csc(data[sort_mask], rows_indices, cols_ptr, rhs)

    return solution.reshape((Ny, Nx)).T


@njit #(nogil=True, parallel=True)
def _fill_matrix_and_rhs(Wo, Wo_0, m, m_0, k, S, S_0, mu_o, mu_w, data, rhs, boundary_conditions):
    """Заполнение массивов матрицы и правой части уравнения давления."""
    num = 0
    # TODO перейти к одномерному массиву, который создается в препроцессинге
    for i in range(Nx):
        for j in range(Ny):
            idx = i + j * Nx
            p_sum = 0.0

            # rhs filling
            rhs[idx] = 0.0  # ((m[i, j] - m_0[i, j]) + m_0[i, j] * S_0[i, j] * (Wo[i, j] - Wo_0[i, j]) / Wo[i, j]) / dt * volume

            # matrix filling
            arr = [[i + 1, j, hx, hy*h], [i - 1, j, hx, hy*h], [i, j + 1, hy, hx*h], [i, j - 1, hy, hx*h]]
            for qq in range(4):
                i1, j1, hij, areaij = arr[qq]
                i1, j1 = int(i1), int(j1)
                if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                    val = mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j], k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * areaij / hij
                    data[num] = val
                    p_sum -= val
                    num += 1
                else:
                    bound = get_bound(i1, j1)
                    i1, j1, hij = i, j, hij * 0.5
                    S_ij = apply_bc(boundary_conditions, bound, 1, S, i, j, hij)
                    val = mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                    k[i1, j1], S_ij, mu_o[i1, j1], mu_w[i1, j1]) * areaij / hij
                    if boundary_conditions[bound, 0, 0] == 1: # Дирихле
                        rhs[idx] -= boundary_conditions[bound, 0, 1] * val
                        p_sum -= val
                    else:  # Нейман
                        rhs[idx] -= boundary_conditions[bound, 0, 1] * val

            data[num] = p_sum
            num += 1


@njit
def _adding_wells(wells, S, k, mu_o, mu_w, data, rhs):
    """Учет скважин в уравнении давления."""
    for i in range(len(wells)):
        well = wells[i]
        temp_data = calc_well_mult(well, S, k, mu_o, mu_w)  # well.q[2] / well.dp
        data[well.idx_mat] -= temp_data
        rhs[well.idx_rhs] -= temp_data * well.p
