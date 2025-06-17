"""Решение уравнения давления: сборка матрицы (МКО и решение СЛАУ)."""
import taichi as ti
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from paraphin import N, NN
from paraphin.constants import data_type, Nx, Ny, hx, hy, dt, volume, h
from paraphin.utils import mid_Ko_Kw

# from scipy.sparse.linalg import splu
# from pypardiso import spsolve

rhs = ti.ndarray(data_type, shape=N)
data = ti.ndarray(data_type, shape=NN)


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
    """
    _fill_matrix_and_rhs(m, m_0, k, S, mu_o, mu_w, data, rhs)
    _adding_wells(wells, data, rhs, k)

    A_csc = csc_matrix((data.to_numpy(), (rows_indices, cols_indices)), shape=(N, N))
    solution = spsolve(A_csc, rhs.to_numpy())  # splu(A_csc).solve(rhs.to_numpy())
    p.from_numpy(solution.reshape((Ny, Nx)).T)


@ti.kernel
def _fill_matrix_and_rhs(m: ti.template(), m_0: ti.template(), k: ti.template(), S: ti.template(), mu_o: ti.template(),
                         mu_w: ti.template(), data: ti.types.ndarray(), rhs: ti.types.ndarray()):
    """Сборка матрицы уравнения давления"""
    num = 0
    for i in ti.ndrange(Nx):
        for j in ti.ndrange(Ny):
            idx = i + j * Nx
            p_sum = 0.0
            arr = [[i + 1, j, hx, hy*h], [i - 1, j, hx, hy*h], [i, j + 1, hy, hx*h], [i, j - 1, hy, hx*h]]
            for qq in ti.static(ti.ndrange(4)):
                i1, j1, hij, areaij = arr[qq]
                if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                    val = mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                    k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * areaij / hij
                    data[num] = val
                    p_sum -= val
                    num += 1

            data[num] = p_sum
            num += 1

            # rhs
            rhs[idx] = (m[i, j] - m_0[i, j]) / dt * volume
    """
    В этом же цикле обновлять поля данных. 
    Использовать поля Wo, Wo_0, m, m_0, k, S с нового временного слоя (префикс new_) 
    """


@ti.kernel
def _adding_wells(wells: ti.template(), data: ti.types.ndarray(), rhs: ti.types.ndarray(), k: ti.template()):
    """Добавление скважин в уравнение давления"""
    ti.loop_config(serialize=True)
    for i in ti.ndrange(wells.shape[0]):
        temp_data = wells[i].q[2] / wells[i].dp_k * k[wells[i].i, wells[i].j]
        data[wells[i].idx_mat] -= temp_data
        rhs[wells[i].idx_rhs] -= temp_data * wells[i].p


# if i != 0:  # i - 1, j, hx, hy*h
#     val = Wo[i, j] * mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
#                                k[i-1, j], S[i-1, j], mu_o[i-1, j], mu_w[i-1, j]) * hy * h / hx
#     data[num] = val
#     p_sum -= val
#     num += 1
# if i != Nx - 1:  # i + 1, j, hx, hy*h
#     val = Wo[i, j] * mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
#                                k[i + 1, j], S[i + 1, j], mu_o[i + 1, j], mu_w[i + 1, j]) * hy * h / hx
#     data[num] = val
#     p_sum -= val
#     num += 1
# if j != 0:  # i, j - 1, hy, hx*h
#     val = Wo[i, j] * mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
#                                k[i, j - 1], S[i, j - 1], mu_o[i, j - 1], mu_w[i, j - 1]) * hx * h / hy
#     data[num] = val
#     p_sum -= val
#     num += 1
# if j != Ny - 1:  # i, j + 1, hy, hx*h
#     val = Wo[i, j] * mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
#                                k[i, j + 1], S[i, j + 1], mu_o[i, j + 1], mu_w[i, j + 1]) * hx * h / hy
#     data[num] = val
#     p_sum -= val
#     num += 1
