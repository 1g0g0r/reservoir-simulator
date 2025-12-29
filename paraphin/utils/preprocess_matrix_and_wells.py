"""Предобработка матрицы уравнения давления и инициализация данных скважин."""
import numpy as np
from numba import njit, prange

from paraphin import NN
from paraphin.constants import Nx, Ny, hx, hy, h


def preprocess_matrix_and_wells(wells, wells_buffer):
    """Препроцессинг профиля матрицы уравнения давления и обработка массива скважин."""
    rows_indices = np.zeros(dtype=np.int32, shape=NN)
    cols_indices = np.zeros(dtype=np.int32, shape=NN)
    _get_rows_cols(row_indices=rows_indices, col_indices=cols_indices)
    diagonal = rows_indices == cols_indices

    # Добавление скважин
    if len(wells_buffer) > 0:
        for i in range(len(wells)):
            wells[i] = wells_buffer[i]['well']
            wells[i].idx_rhs = wells[i].i + wells[i].j * Nx
            wells[i].idx_mat = np.where(np.logical_and(rows_indices == wells[i].idx_rhs, diagonal))[0][0]

    sorted_indices = np.lexsort((rows_indices, cols_indices))
    cols_sorted = cols_indices[sorted_indices]
    rows_sorted = rows_indices[sorted_indices].astype(np.intc, copy=False)
    _, cols_ptr = np.unique(cols_sorted, return_index=True)
    cols_ptr = np.append(cols_ptr, len(cols_sorted)).astype(np.intc, copy=False)

    return sorted_indices, rows_sorted, cols_ptr, wells


@njit #(parallel=True)
def _get_rows_cols(row_indices, col_indices):
    """Сборка матрицы уравнения давления"""
    # TODO убрать использование num. Сохранять индексы
    num = 0
    for i in range(Nx):
        for j in range(Ny):
            idx = i + j * Nx

            arr = [[i + 1, j, hx, hy*h], [i - 1, j, hx, hy*h], [i, j + 1, hy, hx*h], [i, j - 1, hy, hx*h]]
            for qq in range(4):
                i1, j1, hij, areaij = arr[qq]
                if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                    row_indices[num] = idx
                    col_indices[num] = idx + (i1-i) + Nx * (j1-j)
                    num += 1

            row_indices[num] = idx
            col_indices[num] = idx
            num += 1
