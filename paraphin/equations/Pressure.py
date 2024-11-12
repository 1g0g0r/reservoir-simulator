from numpy import log, sqrt, pi
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, bicgstab
from taichi import i32, field, ndrange, kernel, static

from paraphin.constants import default_type, Nx, Ny, area, hx, hy, dt, volume, Po, Pw, rw
from paraphin.utils import mid, show_plot

# Операции с константными величинами (вычисляются один раз только при импорте модуля)
well_mult = 2.0 * pi / log(rw / (0.14 * sqrt(hx*hx + hy*hy))) * volume
N = Nx * Ny  # размер матрицы
NN = (Nx - 2) * (Ny - 2) * 5 + (Nx-2) * 8 + (Ny-2) * 8 + 12  # количество ненулевых элементов


def calc_pressure(p, Wo, Wo_0, m, m_0, k, S, mu_o, mu_w) -> (field(dtype=default_type, shape=(Nx, Ny)), bool):
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
        Водоносыщенность, [-]
    mu_o: taichi.field(Nx, Ny)
        Вязкость нефти, [Па*с]
    mu_w: taichi.field(Nx, Ny)
        Вязкость воды, [Па*с]

    Returns
    -------
    p: taichi.field(Nx, Ny)
        Давление на новом временном слое, [Па]
    mat_singularity: bool
        Matrix singularity flag, [-]
    """
    data = field(default_type, shape=NN)
    row_indices = field(i32, shape=NN)
    col_indices = field(i32, shape=NN)
    b = field(default_type, shape=N)
    mat_singularity = False

    @kernel
    def fill_matrix_and_rhs():
        num = 0
        for i in ndrange(Nx):
            for j in ndrange(Ny):
                idx = i + j * Nx
                p_sum = 0.0
                # matrix
                arr = [[i + 1, j, hx], [i - 1, j, hx], [i, j + 1, hy], [i, j - 1, hy]]
                for qq in static(ndrange(4)):
                    i1, j1, hij = arr[qq]
                    if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                        temp = Wo[i1, j1] * mid(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                                k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * area / hij
                        row_indices[num] = idx
                        col_indices[num] = idx + (i1-i) + Nx * (j1-j)
                        data[num] = -temp
                        p_sum += temp
                        num += 1

                row_indices[num] = idx
                col_indices[num] = idx
                data[num] = p_sum
                num += 1

                # rhs
                b[idx] = (Wo[i, j] * (m[i, j] - m_0[i, j]) + (1 - S[i, j]) *
                          m[i, j] * (Wo[i, j] - Wo_0[i, j])) * volume / dt

        # Добавили скважины в точки (0,0) (nx-1, ny-1)
        b[0] += well_mult * k[0, 0] * Pw / mu_o[0, 0]
        data[0] += well_mult * k[0, 0] / mu_o[0, 0]
        b[N-1] -= well_mult * k[Nx-1, Ny-1] * Po / mu_o[Nx-1, Ny-1] * Wo[Nx-1, Ny-1]
        data[NN-1] -= well_mult * k[Nx-1, Ny-1] / mu_o[Nx-1, Ny-1] * Wo[Nx-1, Ny-1]

    fill_matrix_and_rhs()
    A_csr = csr_matrix((data.to_numpy(), (row_indices.to_numpy(), col_indices.to_numpy())), shape=(N, N))
    try:
        x = spsolve(A_csr, b.to_numpy())
    except Exception as e:
        x, _ = bicgstab(A_csr, b.to_numpy())
        mat_singularity = True

    # import pyamg
    # ml = pyamg.ruge_stuben_solver(A_csr)  # construct the multigrid hierarchy
    # xx = ml.solve(b.to_numpy(), tol=1e-10)

    # show_plot(x, 'plotly')
    p.from_numpy(x.reshape((Nx, Ny)))

    return p, mat_singularity
