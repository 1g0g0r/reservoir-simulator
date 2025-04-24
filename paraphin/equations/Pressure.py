import taichi as ti
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
# from pypardiso import spsolve

from paraphin.constants import data_type, Nx, Ny, hx, hy, dt, volume, h
from paraphin.utils import mid_Ko_Kw
from paraphin.well import calc_q_mult

N = Nx * Ny  # размер матрицы
NN = (Nx - 2) * (Ny - 2) * 5 + (Nx-2) * 8 + (Ny-2) * 8 + 12  # количество ненулевых элементов в матрице давления
data = ti.field(data_type, shape=NN)
rhs = ti.field(data_type, shape=N)


def calc_pressure(p, Wo, m, m_0, k, S, mu_o, mu_w, wells, row_indices_np, col_indices_np) -> None:
    """
    Сборка матрицы и решение СЛАУ уравнения давления (МКО)

    Parameters
    ----------
    p: taichi.field(Nx, Ny)
        Давление, [Па]
    Wo: taichi.field(Nx, Ny)
        Объемная доля масляного компонента в нефти, [-]
    m: taichi.field(Nx, Ny)
        Пористость, [-]
    m_0: taichi.field(Nx, Ny)
        Пористость на прошлом временном слое, [-]
    k: taichi.field(Nx, Ny)
        Проницаемость, [м^2]
    S: taichi.field(Nx, Ny)
        Водонасыщенность, [-]
    mu_o: taichi.field(Nx, Ny)
        Вязкость нефти, [Па*с]
    mu_w: taichi.field(Nx, Ny)
        Вязкость воды, [Па*с]
    """
    _fill_matrix_and_rhs(Wo, m, m_0, k, S, mu_o, mu_w, data, rhs)
    _adding_wells(wells, Wo, S, k, mu_o, mu_w, data, rhs)

    A_csr = csr_matrix((data.to_numpy(), (row_indices_np, col_indices_np)), shape=(N, N))
    solution = spsolve(A_csr, rhs.to_numpy())

    p.from_numpy(solution.reshape((Nx, Ny)))


@ti.kernel
def _adding_wells(wells: ti.template(), Wo: ti.template(), S: ti.template(), k: ti.template(),
                    mu_o: ti.template(), mu_w: ti.template(), data: ti.template(), rhs: ti.template()):
    """Добавление скважин в уравнение давления"""
    for i in ti.ndrange(wells.shape[0]):
        temp_data = Wo[wells[i].i, wells[i].j] * calc_q_mult(wells[i], S, k, mu_o, mu_w)
        data[wells[i].idx_mat] -= temp_data
        rhs[wells[i].idx_rhs] -= temp_data * wells[i].p


@ti.kernel
def _fill_matrix_and_rhs(Wo: ti.template(), m: ti.template(), m_0: ti.template(), k: ti.template(), S: ti.template(),
                         mu_o: ti.template(), mu_w: ti.template(), data: ti.template(), rhs: ti.template()):
    """Сборка матрицы уравнения давления"""
    num = 0
    for j in ti.ndrange(Ny):
        for i in ti.ndrange(Nx):
            idx = i + j * Nx
            p_sum = 0.0
            # matrix
            arr = [[i + 1, j, hx, hy*h], [i - 1, j, hx, hy*h], [i, j + 1, hy, hx*h], [i, j - 1, hy, hx*h]]
            for qq in ti.static(ti.ndrange(4)):
                i1, j1, hij, areaij = arr[qq]
                if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                    val = Wo[i, j] * mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                               k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * areaij / hij
                    data[num] = val
                    p_sum -= val
                    num += 1
            data[num] = p_sum
            num += 1

            # rhs
            rhs[idx] = Wo[i, j] * (m[i, j] - m_0[i, j]) / dt * volume

    """
    В этом же цикле обновлять поля данных. 
    Использовать поля Wo, Wo_0, m, m_0, k, S с нового временного слоя (префикс new_) 
    """


def preprocess_matrix_and_wells(wells, wells_buffer):
    row_indices = ti.field(ti.i32, shape=NN)
    col_indices = ti.field(ti.i32, shape=NN)
    _get_rows_cols(row_indices=row_indices, col_indices=col_indices)

    row_indices_np = row_indices.to_numpy()
    col_indices_np = col_indices.to_numpy()
    diagonal = row_indices_np == col_indices_np

    # Добавили скважины
    for i in range(wells.shape[0]):
        wells[i] = wells_buffer[i]['well']
        wells[i].idx_rhs = wells[i].i + wells[i].j * Nx
        wells[i].idx_mat = np.where(np.logical_and(row_indices_np == wells[i].idx_rhs, diagonal))[0][0]

    return row_indices_np, col_indices_np, wells


@ti.kernel
def _get_rows_cols(row_indices: ti.template(), col_indices: ti.template()):
    """Сборка матрицы уравнения давления"""
    num = 0
    for j in ti.ndrange(Ny):
        for i in ti.ndrange(Nx):
            idx = i + j * Nx
            # matrix
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
