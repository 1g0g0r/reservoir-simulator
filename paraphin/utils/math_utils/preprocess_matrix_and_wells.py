"""Предобработка матрицы уравнения давления и инициализация данных скважин."""
import numpy as np
import taichi as ti

from paraphin import NN
from paraphin.constants import Nx, Ny, hx, hy, h
from paraphin.well import upd_q_and_eta


def preprocess_matrix_and_wells(wells, wells_buffer, p, S, k, mu_o, mu_w):
    """Препроцессинг профиля матрицы уравнения давления и обработка массива скважин."""
    rows_indices = ti.field(ti.i32, shape=NN)
    cols_indices = ti.field(ti.i32, shape=NN)
    _get_rows_cols(row_indices=rows_indices, col_indices=cols_indices)

    rows_indices_np = rows_indices.to_numpy()
    cols_indices_np = cols_indices.to_numpy()
    diagonal = rows_indices_np == cols_indices_np

    # Добавление скважин
    if len(wells_buffer) > 0:
        for i in range(wells.shape[0]):
            wells[i] = wells_buffer[i]['well']
            wells[i].idx_rhs = wells[i].i + wells[i].j * Nx
            wells[i].idx_mat = np.where(np.logical_and(rows_indices_np == wells[i].idx_rhs, diagonal))[0][0]

        _update_wells_data(wells, p, S, k, mu_o, mu_w)

    sorted_indices = np.lexsort((rows_indices_np, cols_indices_np))
    cols_sorted = cols_indices_np[sorted_indices]
    rows_sorted = rows_indices_np[sorted_indices].astype(np.intc, copy=False)
    _, cols_ptr = np.unique(cols_sorted, return_index=True)
    cols_ptr = np.append(cols_ptr, len(cols_sorted)).astype(np.intc, copy=False)

    return sorted_indices, rows_sorted, cols_ptr, wells


@ti.kernel
def _get_rows_cols(row_indices: ti.template(), col_indices: ti.template()):
    """Сборка матрицы уравнения давления"""
    num = 0
    for i in ti.ndrange(Nx):
        for j in ti.ndrange(Ny):
            idx = i + j * Nx

            arr = [[i + 1, j, hx, hy*h], [i - 1, j, hx, hy*h], [i, j + 1, hy, hx*h], [i, j - 1, hy, hx*h]]
            for qq in ti.static(ti.ndrange(4)):
                i1, j1, hij, areaij = arr[qq]
                if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                    row_indices[num] = idx
                    col_indices[num] = idx + (i1-i) + Nx * (j1-j)
                    num += 1

            row_indices[num] = idx
            col_indices[num] = idx
            num += 1


@ti.kernel
def _update_wells_data(wells: ti.template(), p: ti.template(), S: ti.template(), k: ti.template(), mu_o: ti.template(), mu_w: ti.template()):
    """Обновление дебетов и обводненности скважин."""
    ti.loop_config(serialize=True)
    for i in ti.ndrange(wells.shape[0]):
        wells[i] = upd_q_and_eta(wells[i], p, S, k, mu_o, mu_w)
