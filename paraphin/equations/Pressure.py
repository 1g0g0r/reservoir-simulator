import taichi as ti
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
# from pypardiso import spsolve

from paraphin.constants import data_type, Nx, Ny, hx, hy, dt, volume, bar_to_pa, h, DEBUGGING
from paraphin.utils import show_plot, mid

N = Nx * Ny  # размер матрицы
NN = (Nx - 2) * (Ny - 2) * 5 + (Nx-2) * 8 + (Ny-2) * 8 + 12  # количество ненулевых элементов
data = ti.field(data_type, shape=NN)
row_indices = ti.field(ti.i32, shape=NN)
col_indices = ti.field(ti.i32, shape=NN)
rhs = ti.field(data_type, shape=N)


def calc_pressure(p, Wo, Wo_0, m, m_0, k, S, mu_o, mu_w, wells) -> None:
    """
    Сборка матрицы и решение СЛАУ уравнения давления (МКО)

    Parameters
    ----------
    p: taichi.field(Nx, Ny)
        Давление, [Па]
    Wo: taichi.field(Nx, Ny)
        Объемная доля масляного компонента в нефти, [-]
    Wo_0: taichi.field(Nx, Ny)
        Объемная доля масляного компонента в нефти на прошлом временном слое, [-]
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
    @ti.kernel
    def fill_matrix_and_rhs():
        num = 0
        for i, j in ti.ndrange(Nx, Ny):
            idx = i + j * Nx
            p_sum = 0.0
            # matrix
            arr = [[i + 1, j, hx, hy*h], [i - 1, j, hx, hy*h], [i, j + 1, hy, hx*h], [i, j - 1, hy, hx*h]]
            for qq in ti.static(ti.ndrange(4)):
                i1, j1, hij, areaij = arr[qq]
                if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                    val = Wo[i, j] * mid(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                         k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * areaij / hij
                    row_indices[num] = idx
                    col_indices[num] = idx + (i1-i) + Nx * (j1-j)
                    data[num] = val
                    p_sum -= val
                    num += 1

            row_indices[num] = idx
            col_indices[num] = idx
            data[num] = p_sum
            num += 1

            # rhs
            rhs[idx] = (Wo[i, j] * (m[i, j] - m_0[i, j]) + (1 - S[i, j]) * m[i, j] * (Wo[i, j] - Wo_0[i, j])) / dt * volume

    fill_matrix_and_rhs()

    row_indices_np = row_indices.to_numpy()
    col_indices_np = col_indices.to_numpy()
    data_np = data.to_numpy()
    diagonal = row_indices_np == col_indices_np

    # Добавили скважины
    for well in wells:
        well.calc_q(well.p + 1.0, S, k, mu_o, mu_w)
        temp_data = Wo[well.i, well.j] * well.q[2]
        data_idx = np.logical_and(row_indices_np == well.idx, diagonal)
        data_np[data_idx] -= temp_data
        rhs[well.idx] -= temp_data * well.p

    A_csr = csr_matrix((data_np, (row_indices_np, col_indices_np)), shape=(N, N))
    solution = spsolve(A_csr, rhs.to_numpy())

    p.from_numpy(solution.reshape((Nx, Ny)))
    if DEBUGGING:
        show_plot(solution / bar_to_pa, 'Pressure')
