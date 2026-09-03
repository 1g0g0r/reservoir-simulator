"""Прямой ленточный решатель СЛАУ уравнения давления.

Неизвестная нумеруется `idx = i + j*Nx`, шаблон пятиточечный, поэтому матрица уравнения давления -
ленточная с полушириной `Nx` и ровно тремя ненулевыми диагоналями: 0 (центр), 1 (сосед по x),
Nx (сосед по y). Хранить ее в CSC и звать SuperLU незачем: разложение Холецкого по ленте на порядок дешевле.

Собирается сразу положительно определенная форма `M = -A` (диагональ положительна, внедиагональные
элементы отрицательны, правая часть `c = -rhs`). M симметрична (гармоническое среднее подвижностей
симметрично по ячейкам), диагонально доминантна и неприводима, скважины и Дирихле дают строгое
доминирование хотя бы в одной строке - значит M положительно определена и выбор главного элемента
не нужен.

Значения матрицы меняются на ~0.2% за шаг, поэтому фактор Холецкого пересобирается раз в
`p_refactor_period` шагов, а промежуточные шаги добираются PCG с этим (устаревшим) фактором в роли
предобуславливателя - в среднем 1.8 итерации. Если PCG не уложился в `p_pcg_maxit`, фактор
пересобирается и решение получается точно, так что за результат отвечает всегда прямой метод.
"""
import numpy as np
from numba import njit, prange

from paraphin.constants import Nx, Ny, p_pcg_rtol, p_pcg_maxit, p_refactor_period

_N = Nx * Ny   # размер системы
_KD = Nx       # полуширина ленты


@njit(cache=True)
def band_pack(diag, ex, ey, w) -> None:
    """Раскладка трех диагоналей в ленточное хранение `w[j, d] = M[j, j+d]`.

    Обнуляется весь буфер: разложение заполняет ленту целиком, а не только исходные три диагонали.
    """
    w[:, :] = 0.0
    for j in range(_N):
        w[j, 0] = diag[j]
        w[j, 1] = ex[j]
        if j + _KD < _N:
            w[j, _KD] = ey[j]


@njit(cache=True)
def band_chol(w) -> int:
    """Разложение Холецкого по ленте на месте: M = R^T R, R верхняя треугольная.

    Возвращает 0 при успехе или номер строки (1-based), на которой ведущий элемент оказался
    неположительным - это значит, что матрица вырождена (обычно нулевая суммарная подвижность
    в ячейке при полном блокировании пор).
    """
    for j in range(_N):
        v = w[j, 0]
        if v <= 0.0:
            return j + 1
        v = np.sqrt(v)
        w[j, 0] = v
        mm = _KD if j + _KD < _N else _N - 1 - j
        for dd in range(1, mm + 1):
            w[j, dd] /= v

        for d1 in range(1, mm + 1):
            f = w[j, d1]
            if f != 0.0:
                row = j + d1
                for d2 in range(d1, mm + 1):
                    w[row, d2 - d1] -= f * w[j, d2]

    return 0


@njit(cache=True)
def band_solve(w, rhs, x) -> None:
    """Решение M x = rhs по готовому фактору: прямой и обратный ход."""
    for j in range(_N):
        acc = rhs[j]
        lo = _KD if j >= _KD else j
        for dd in range(1, lo + 1):
            acc -= w[j - dd, dd] * x[j - dd]
        x[j] = acc / w[j, 0]

    for j in range(_N - 1, -1, -1):
        acc = x[j]
        mm = _KD if j + _KD < _N else _N - 1 - j
        for dd in range(1, mm + 1):
            acc -= w[j, dd] * x[j + dd]
        x[j] = acc / w[j, 0]


@njit(parallel=True, cache=True)
def stencil_mv(diag, ex, ey, v, out) -> None:
    """Умножение на матрицу прямо по трем диагоналям, без сборки разреженного формата."""
    for j in prange(_N):
        acc = diag[j] * v[j]
        if j + 1 < _N:
            acc += ex[j] * v[j + 1]
        if j > 0:
            acc += ex[j - 1] * v[j - 1]
        if j + _KD < _N:
            acc += ey[j] * v[j + _KD]
        if j >= _KD:
            acc += ey[j - _KD] * v[j - _KD]
        out[j] = acc


@njit(cache=True)
def pcg_band(diag, ex, ey, w, rhs, x, r, z, p, q) -> int:
    """PCG с готовым фактором Холецкого в роли предобуславливателя.

    Стартует с `x` - решения предыдущего шага. Возвращает число итераций или -1, если не сошелся.
    """
    stencil_mv(diag, ex, ey, x, q)
    nb = 0.0
    for j in range(_N):
        r[j] = rhs[j] - q[j]
        nb += rhs[j] * rhs[j]
    nb = np.sqrt(nb)

    band_solve(w, r, z)
    rz = 0.0
    for j in range(_N):
        p[j] = z[j]
        rz += r[j] * z[j]

    for it in range(p_pcg_maxit):
        stencil_mv(diag, ex, ey, p, q)
        pq = 0.0
        for j in range(_N):
            pq += p[j] * q[j]
        alpha = rz / pq

        rn = 0.0
        for j in range(_N):
            x[j] += alpha * p[j]
            r[j] -= alpha * q[j]
            rn += r[j] * r[j]
        if np.sqrt(rn) <= p_pcg_rtol * nb:
            return it + 1

        band_solve(w, r, z)
        rz_new = 0.0
        for j in range(_N):
            rz_new += r[j] * z[j]
        beta = rz_new / rz
        rz = rz_new
        for j in range(_N):
            p[j] = beta * p[j] + z[j]

    return -1


@njit(cache=True)
def solve_band_system(diag, ex, ey, rhs, w, x, r, z, p, q, age) -> int:
    """Решение системы с ленивой факторизацией. Возвращает новый возраст фактора.

    `age` - сколько шагов прошло с последней пересборки фактора. При `age >= p_refactor_period`
    и на первом шаге (где age задан заведомо большим) считается точно прямым методом.
    """
    if 0 < age < p_refactor_period:
        if pcg_band(diag, ex, ey, w, rhs, x, r, z, p, q) >= 0:
            return age + 1

    band_pack(diag, ex, ey, w)
    info = band_chol(w)
    if info != 0:
        raise ValueError('Матрица давления вырождена: неположительный ведущий элемент. '
                         'Скорее всего в какой-то ячейке обнулилась суммарная подвижность.')
    band_solve(w, rhs, x)

    return 1
