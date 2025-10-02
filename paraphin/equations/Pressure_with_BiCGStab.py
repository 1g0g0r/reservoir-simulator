"""Решение уравнения давления: сборка матрицы (МКО и решение СЛАУ)."""
import numpy as np
import taichi as ti
from scipy.sparse.linalg._dsolve.linsolve import _superlu
from taichi._kernels import ndarray_to_ext_arr, ext_arr_to_tensor

from paraphin import N, NN
from paraphin.constants import data_type, Nx, Ny, hx, hy, dt, volume, h
from paraphin.utils import mid_Ko_Kw
from paraphin.well import calc_well_mult
# from pypardiso import spsolve


def calc_pressure(p, Wo, Wo_0, m, m_0, k, S, S_0, mu_o, mu_w, wells, solver, rhs, matrix):
    """Сборка матрицы и решение СЛАУ уравнения давления (МКО)

    Parameters
    ----------
    p: taichi.field(Nx, Ny)
        Давление, [Па]
    Wo: taichi.field(Nx, Ny)
        Массовая доля масляного компонента в нефти, [-]
    Wo_0: taichi.field(Nx, Ny)
        Массовая доля масляного компонента в нефти на прошлом временном слое, [-]
    m: taichi.field(Nx, Ny)
        Пористость, [-]
    m_0: taichi.field(Nx, Ny)
        Пористость на прошлом временном слое, [-]
    k: taichi.field(Nx, Ny)
        Проницаемость, [м^2]
    S: taichi.field(Nx, Ny)
        Водонасыщенность, [-]
    S_0: taichi.field(Nx, Ny)
        Водонасыщенность на прошлом временном слое, [-]
    mu_o: taichi.field(Nx, Ny)
        Вязкость нефти, [Па*с]
    mu_w: taichi.field(Nx, Ny)
        Вязкость воды, [Па*с]
    wells: taichi.field(n_wells)
        Массив скважин
    solver: BICGSolver
        Класс решения пяти диагональныхматриц стабилизированным методом бисопряженных градиентов
    rhs: taichi.field(Nx*Ny)
        Вектор правой части уравнения давления
    matrix: taichi.field(5, Nx*Ny)
        Массив пятидиагональной матрицы уравнения давления
        0: i - Nx
        1: i - 1
        2: i
        3: i + 1
        4: i + Nx
    """
    _build_matrix_and_rhs(wells, Wo, Wo_0, m, m_0, k, S, S_0, mu_o, mu_w, matrix, rhs)
    solver.solve(matrix, rhs)


@ti.kernel
def _build_matrix_and_rhs(wells: ti.template(), Wo: ti.template(), Wo_0: ti.template(), m: ti.template(), m_0: ti.template(),
                          k: ti.template(), S: ti.template(), S_0: ti.template(), mu_o: ti.template(), mu_w: ti.template(),
                          matrix: ti.template(), rhs: ti.template()):
    _fill_matrix_and_rhs(Wo, Wo_0, m, m_0, k, S, S_0, mu_o, mu_w, matrix, rhs)
    _adding_wells(wells, S, k, mu_o, mu_w, matrix, rhs)


@ti.func
def _fill_matrix_and_rhs(Wo: ti.template(), Wo_0: ti.template(), m: ti.template(), m_0: ti.template(), k: ti.template(),
                         S: ti.template(), S_0: ti.template(), mu_o: ti.template(), mu_w: ti.template(),
                         matrix: ti.template(), rhs: ti.template()):
    """Сборка матрицы уравнения давления"""
    for i in ti.ndrange(Nx):
        for j in ti.ndrange(Ny):
            idx = j + i * Nx
            p_sum = 0.0
            arr = [[0, i, j - 1, hy, hx*h], [1, i - 1, j, hx, hy*h], [3, i + 1, j, hx, hy*h], [4, i, j + 1, hy, hx*h]]
            for qq in ti.static(ti.ndrange(4)):
                idx_mat, i1, j1, hij, areaij = arr[qq]
                if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                    val = mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                    k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * areaij / hij
                    matrix[idx_mat, idx] = val
                    p_sum -= val

            matrix[2, idx] = p_sum

            rhs[idx] = 0.0  # ((m[i, j] - m_0[i, j]) + m_0[i, j] * S_0[i, j] * (Wo[i, j] - Wo_0[i, j]) / Wo[i, j]) / dt * volume


@ti.func
def _adding_wells(wells: ti.template(), S: ti.template(), k: ti.template(), mu_o: ti.template(),
                  mu_w: ti.template(), matrix: ti.template(), rhs: ti.template()):
    """Добавление скважин в уравнение давления"""
    for i in wells:
        well = wells[i]
        temp_data = calc_well_mult(well, S, k, mu_o, mu_w)  # well.q[2] / well.dp
        matrix[2, well.idx] -= temp_data
        rhs[well.idx] -= temp_data * well.p
