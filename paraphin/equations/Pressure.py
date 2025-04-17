import taichi as ti
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from paraphin.constants import data_type, Nx, Ny, hx, hy, dt, volume, Po, Pw, bar_to_pa, h, DEBUGGING
from paraphin.constants import conductivity_well as c_well
from paraphin.utils import show_plot, pf_o, pf_w, mid

N = Nx * Ny  # размер матрицы
NN = (Nx - 2) * (Ny - 2) * 5 + (Nx-2) * 8 + (Ny-2) * 8 + 12  # количество ненулевых элементов
data = ti.field(data_type, shape=NN)
row_indices = ti.field(ti.i32, shape=NN)
col_indices = ti.field(ti.i32, shape=NN)
rhs = ti.field(data_type, shape=N)


def calc_pressure(p, Wo, Wo_0, m, m_0, k, S, mu_o, mu_w) -> None:
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
        for i in ti.ndrange(Nx):
            for j in ti.ndrange(Ny):
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

        # Добавили скважины в точки (0,0) (Nx-1, Ny-1)
        data[2]    -= c_well * Wo[0, 0] * k[0, 0] / mu_w[0, 0]
        rhs[0]     -= c_well * Wo[0, 0] * k[0, 0] / mu_w[0, 0] * Pw
        data[NN-1] -= c_well * Wo[Nx-1, Ny-1] * k[Nx-1, Ny-1] * (pf_o(S[Nx-1, Ny-1]) / mu_o[Nx-1, Ny-1] + pf_w(S[Nx-1, Ny-1]) / mu_w[Nx-1, Ny-1])
        rhs[N-1]   -= c_well * Wo[Nx-1, Ny-1] * k[Nx-1, Ny-1] * (pf_o(S[Nx-1, Ny-1]) / mu_o[Nx-1, Ny-1] + pf_w(S[Nx-1, Ny-1]) / mu_w[Nx-1, Ny-1]) * Po

    fill_matrix_and_rhs()
    A_csr = csr_matrix((data.to_numpy(), (row_indices.to_numpy(), col_indices.to_numpy())), shape=(N, N))

    solution = spsolve(A_csr, rhs.to_numpy())  # lgmres(A_csr, rhs.to_numpy(), rtol=1e-8)[0]
    # ml = pyamg.ruge_stuben_solver(A_csr)
    # x = ml.solve(rhs.to_numpy(), tol=1e-8)

    p.from_numpy(solution.reshape((Nx, Ny)))
    if DEBUGGING:
        show_plot(solution / bar_to_pa, 'Pressure')
