import numpy as np
import taichi as ti
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
# from scipy.sparse.linalg import spsolve
# from pypardiso import spsolve
from taichi._kernels import ndarray_to_ext_arr, ext_arr_to_tensor

from paraphin import N, NN
from paraphin.constants import data_type, Nx, Ny, hx, hy, dt, volume, h, np_dtype
from paraphin.utils import mid_Ko_Kw

rhs = ti.ndarray(data_type, shape=N)
rhs_np = np.zeros(N, dtype=np_dtype)
data = ti.ndarray(data_type, shape=NN)
data_np = np.zeros(NN, dtype=np_dtype)


def calc_pressure(p, Wo, m, m_0, k, S, mu_o, mu_w, wells, rows_indices, cols_indices) -> None:
    """Сборка матрицы и решение СЛАУ уравнения давления (МКО)

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
    wells: taichi.field(n_wells)
        Массив скважин
    mask_csc_sort: np.ndarray(NN)
        Маска сортировки элементов в разреженном формате CSC

    """
    _fill_matrix_and_rhs(Wo, m, m_0, k, S, mu_o, mu_w, data, rhs)
    _adding_wells(wells, Wo, data, rhs)

    ndarray_to_ext_arr(data, data_np)
    ndarray_to_ext_arr(rhs, rhs_np)

    A_csc = csc_matrix((data_np, (rows_indices, cols_indices)), shape=(N, N))
    sp = splu(A_csc)
    solution = sp.solve(rhs_np)

    ext_arr_to_tensor(solution.reshape((Nx, Ny)), p)


@ti.kernel
def _adding_wells(wells: ti.template(), Wo: ti.template(), data: ti.types.ndarray(), rhs: ti.types.ndarray()):
    """Добавление скважин в уравнение давления"""
    ti.loop_config(serialize=True)
    for i in ti.ndrange(wells.shape[0]):
        temp_data = Wo[wells[i].i, wells[i].j] * wells[i].q[2] / wells[i].dp
        data[wells[i].idx_mat] -= temp_data
        rhs[wells[i].idx_rhs] -= temp_data * wells[i].p


@ti.kernel
def _fill_matrix_and_rhs(Wo: ti.template(), m: ti.template(), m_0: ti.template(), k: ti.template(), S: ti.template(),
                         mu_o: ti.template(), mu_w: ti.template(), data: ti.types.ndarray(), rhs: ti.types.ndarray()):
    """Сборка матрицы уравнения давления"""
    num = 0
    for j in ti.ndrange(Ny):
        for i in ti.ndrange(Nx):
            idx = i + j * Nx
            p_sum = 0.0

            if i != 0:  # i - 1, j, hx, hy*h
                val = Wo[i, j] * mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                           k[i-1, j], S[i-1, j], mu_o[i-1, j], mu_w[i-1, j]) * hy * h / hx
                data[num] = val
                p_sum -= val
                num += 1
            if i != Nx - 1:  # i + 1, j, hx, hy*h
                val = Wo[i, j] * mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                           k[i + 1, j], S[i + 1, j], mu_o[i + 1, j], mu_w[i + 1, j]) * hy * h / hx
                data[num] = val
                p_sum -= val
                num += 1
            if j != 0:  # i, j - 1, hy, hx*h
                val = Wo[i, j] * mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                           k[i, j - 1], S[i, j - 1], mu_o[i, j - 1], mu_w[i, j - 1]) * hx * h / hy
                data[num] = val
                p_sum -= val
                num += 1
            if j != Ny - 1:  # i, j + 1, hy, hx*h
                val = Wo[i, j] * mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                           k[i, j + 1], S[i, j + 1], mu_o[i, j + 1], mu_w[i, j + 1]) * hx * h / hy
                data[num] = val
                p_sum -= val
                num += 1

            data[num] = p_sum
            num += 1

            # rhs
            rhs[idx] = Wo[i, j] * (m[i, j] - m_0[i, j]) / dt * volume
            # matrix
            # arr = [[i + 1, j, hx, hy*h], [i - 1, j, hx, hy*h], [i, j + 1, hy, hx*h], [i, j - 1, hy, hx*h]]
            # for qq in ti.static(ti.ndrange(4)):
            #     i1, j1, hij, areaij = arr[qq]
            #     if (0 <= i1 < Nx) and (0 <= j1 < Ny):
            #         val = Wo[i, j] * mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
            #                                    k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * areaij / hij
            #         data[num] = val
            #         p_sum -= val
            #         num += 1

    """
    В этом же цикле обновлять поля данных. 
    Использовать поля Wo, Wo_0, m, m_0, k, S с нового временного слоя (префикс new_) 
    """
