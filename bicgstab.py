import taichi as ti
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from time import perf_counter

from paraphin.utils import show_plot
from paraphin.constants import data_type

# Инициализация Taichi
ti.init(arch=ti.cpu, default_fp=ti.f64)

Nx, Ny = 6, 6
N = Nx * Ny


@ti.data_oriented
class BICGSolver:
    def __init__(self, sol: ti.template, eps=1e-5, debug=True, max_iter = 150):
        self.eps = eps
        self.debug = debug
        self.max_iter = ti.min(max_iter, Nx * Ny)

        self.betta = ti.field(dtype=data_type, shape=())
        self.rho   = ti.field(dtype=data_type, shape=())
        self.rho_0 = ti.field(dtype=data_type, shape=())
        self.alpha = ti.field(dtype=data_type, shape=())
        self.omega = ti.field(dtype=data_type, shape=())
        self.err   = ti.field(dtype=data_type, shape=())

        self.x = sol
        self.r_0 = ti.field(dtype=data_type, shape=N)
        self.r = ti.field(dtype=data_type, shape=N)
        self.v = ti.field(dtype=data_type, shape=N)
        self.p = ti.field(dtype=data_type, shape=N)
        self.t = ti.field(dtype=data_type, shape=N)
        self.s = ti.field(dtype=data_type, shape=N)


    def init(self, mat: ti.template(), rhs: ti.template()):
        """Инициализация решения"""
        self.rho_0[None], self.alpha[None], self.omega[None] = 1.0, 1.0, 1.0
        self._calc_err(self.r_0, mat, self.x, rhs)
        self._fill_init_arrays()

    @ti.kernel
    def _dot_product(self, vec_1: ti.template(), vec_2: ti.template()) -> data_type:
        """Скалярное произведение векторов."""
        ret = 0.0
        for i in vec_1:
            ret += vec_1[i] * vec_2[i]
        return ret

    @ti.kernel
    def _mat_mult(self, result: ti.template(), mat: ti.template(), vector: ti.template()):
        """Умножение пятидиагональной матрицы на вектор.
        0: i - Nx
        1: i - 1
        2: i
        3: i + 1
        4: i + Nx
        """
        for i in vector:
            result[i] = (mat[0, i] * vector[i - Nx] + mat[1, i] * vector[i - 1] +
                         mat[2, i] * vector[i] + mat[3, i] * vector[i + 1] +
                         mat[4, i] * vector[i + Nx])

    @ti.kernel
    def _calc_err(self, array: ti.template(), mat: ti.template(), vector: ti.template(), rhs: ti.template()):
        """Умножение матрицы на вектор решения и вычитание правой части.
        0: i - Nx
        1: i - 1
        2: i
        3: i + 1
        4: i + Nx
        """
        for i in self.r:
            # i -> ii + Nx*jj
            jj = i // Nx
            ii = i - Nx * jj
            array[i] = rhs[i] - (mat[0, i] * get(vector, ii, jj - 1) + mat[1, i] * get(vector, ii - 1, jj) +
                                 mat[2, i] * get(vector, ii, jj) + mat[3, i] * get(vector, ii + 1, jj) +
                                 mat[4, i] * get(vector, ii, jj + 1))

    @ti.kernel
    def _fill_init_arrays(self):
        """Заполнение полей при инициализации решателя."""
        for i in self.r:
            self.r[i] = self.r_0[i]
            self.v[i] = 0.0
            self.p[i] = 0.0

    @ti.kernel
    def _calc_p(self):
        """Умножение матрицы на вектор."""
        for i in self.r:
            self.p[i] = self.r[i] + self.betta * (self.p[i] - self.omega[None] * self.v[i])

    @ti.kernel
    def _calc_s(self):
        """Умножение матрицы на вектор."""
        for i in self.r:
            self.s[i] = self.r[i] - self.alpha[None] * self.v[i]

    @ti.kernel
    def _calc_omega(self):
        """Умножение матрицы на вектор."""
        t_s, t_t = 0.0, 0.0
        for i in self.r:
            t_s += self.t[i] * self.s[i]
            t_t += self.t[i] * self.t[i]
        # self.omega[None] = self._dot_product(self.t, self.s) / self._dot_product(self.t, self.t)
        self.omega[None] = t_s / t_t

    @ti.kernel
    def _calc_x_and_r(self):
        """Умножение матрицы на вектор."""
        r_r = 0.0
        for i in self.r:
            # i -> ii + Nx*jj
            jj = i // Nx
            ii = i - Nx * jj  # TODO разобраться с индексами
            self.x[jj, ii] += self.omega[None] * self.s[i] + self.alpha[None] * self.p[i]
            new_r = self.s[i] - self.omega[None] * self.t[i]
            self.r[i] = new_r
            r_r += new_r * new_r
        self.err[None] = ti.sqrt(r_r)

    def _iter_BiCGStab(self, mat: ti.template()):
        """Итерация стабилизированного метода бисопряженных градиентов."""
        self.rho[None] = self._dot_product(self.r_0, self.r)    # 1
        self.betta = self.rho[None] * self.alpha[None] / self.rho_0[None] / self.omega[None]  # 2
        self._calc_p()                                          # 3
        self._mat_mult(self.v, mat, self.p)                     # 4
        self.alpha[None] = self.rho[None] / self._dot_product(self.r_0, self.v) # 5
        self._calc_s()                                          # 6
        self._mat_mult(self.t, mat, self.s)                     # 7
        self._calc_omega()                                      # 8
        self._calc_x_and_r()                                    # 9-10
        self.rho_0[None] = self.rho[None]

        # self._calc_err(self.s, mat, self.x, rhs)
        # self.err[None] = self._dot_product(self.s, self.s)

    def solve(self, mat, rhs):
        self.init(mat, rhs)
        for i in range(self.max_iter):
            self._iter_BiCGStab(mat)
            if self.debug:
                print('>>> Iter =', i, ' Residual =', self.err[None])
            if self.err[None] < self.eps and i > 0:
                print('Iters:', i)
                break

@ti.func
def get(_arr, _i, _j=None):
    """Получение элемента массива с обработкой выхода за границы."""
    ret = 0.0
    if 0 < _i < Nx - 1 and 0 < _j < Ny - 1:
            ret = _arr[_i, _j]

    return ret

# Инициализация коэффициентов (оператор Лапласа)
@ti.kernel
def init_coef(_coef: ti.template()):
    for i, j in ti.ndrange(Nx, Ny):
        idx = i + j * Nx
        arr = [[i, j - 1], [i - 1, j], [i, j], [i + 1, j], [i, j + 1]]
        for qq in ti.static(ti.ndrange(5)):
            i1, j1 = arr[qq]
            if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                _coef[qq, idx] = 1.0 if qq[0] != 2 else -4.0


# Инициализация правой части (точечный источник в центре)
@ti.kernel
def init_rhs(_rhs: ti.template()):
    for i in _rhs:
        jj = i // Nx
        ii = i - Nx * jj
        if ii == Nx // 4 and jj == Ny * 2 // 3:
            _rhs[i] = 10.0
        else:
            _rhs[i] = 0.0


# Инициализация
coef = ti.field(ti.f64, shape=(5, N))
rhs = ti.field(ti.f64, shape=N)

init_coef(coef)
init_rhs(rhs)

ti.field(ti.f64, shape=(Nx, Ny))
# Создание и запуск решателя
x_ti = ti.field(ti.f64, shape=(Nx, Ny))
tt = perf_counter()
solver = BICGSolver(sol=x_ti, eps=1e-6, max_iter=10, debug=True)
print(perf_counter() - tt, 'init solver')
print()

tt = perf_counter()
solver.solve(coef, rhs)
print(perf_counter() - tt, 'ti solve')

A_np = diags(coef.to_numpy(), [-Nx, -1, 0, 1, Nx], shape=(N, N), format='csr')
# Решение системы
tt = perf_counter()
x = spsolve(A_np, rhs.to_numpy())
print(perf_counter() - tt, 'scipy solve')

print(np.max(x_ti.to_numpy() - x.reshape(Nx, Ny)), np.min(x_ti.to_numpy() - x.reshape(Nx, Ny)))
show_plot(x_ti.to_numpy() - x.reshape(Nx, Ny), "Разность решений SciPy и Taichi", True)
