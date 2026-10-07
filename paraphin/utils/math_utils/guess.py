"""Начальное приближение PCG уравнения давления: проекция на прошлые решения (Fischer, 1998).

Число итераций PCG держит не гладкость решения во времени, а погрешность прошлых решений: каждое верно лишь до
`p_pcg_rtol`, и экстраполяция по ним эту погрешность усиливает (у квадратичной веса 3, -3, 1 - в ~4 раза). С точными
прошлыми решениями многосеточному PCG хватало 0.2 итерации на шаг, с решениями расчета - 1.7, и ослабление
`p_pcg_rtol` не помогло бы: погрешность масштабируется вместе с допуском.

Поэтому приближение ищется проекцией: x0 = x1 + sum_a c_a*d_a, где x1 - прошлое решение, d_a = x_a - x_{a+1} -
разности `p_guess_m` + 1 прошлых решений, а c минимизирует ошибку в энергетической норме: (D^T M D) c = D^T (b - M x1).
Любая полиномиальная экстраполяция по этим решениям (в том числе квадратичная) лежит в том же подпространстве,
поэтому проекция не хуже ее, а составляющие погрешности прошлых решений гасит: итераций 0.8 вместо 1.7 на 100x100
(docs/PERFORMANCE_FINDINGS.md, «Решатель давления»). Цена - проход по матрице и `p_guess_m` + 1 векторам
(Грам) и проход на сборку x0.

Прошлые решения - кольцо `p_hist` (`p_guess_m` строк), номер новейшего лежит в состоянии решателя (`ST_HEAD`),
поэтому за шаг копируется один вектор, а не сдвигается вся история. Пока история не накопилась, разности нулевые и
проекция их отбрасывает: старт от прошлого решения.
"""
import numpy as np
from numba import njit, prange

from paraphin.constants import Nx, Ny, p_guess_m
from paraphin.utils.math_utils.mg_solver import ST_HEAD

_M = p_guess_m
_K = _M * (_M + 1) // 2 + _M  # верхний треугольник Грама и правая часть
_PX = Nx + 2                  # векторы - с рамкой (`layout`)


@njit(parallel=True, fastmath=True, cache=True)
def _gram(diag, ex, ey, rhs, x, hist, slots, part):
    """Построчные суммы: part[j] - верхний треугольник D^T M D, затем D^T (b - M x1) по строке j сетки.

    M d считается по самим разностям, а не как M x_a - M x_{a+1}: при M x ~ b разность произведений теряет ~9 знаков,
    а Грам обусловлен на 1e10-1e11 (разности соседних решений почти коллинеарны) - приближение было в 5 раз хуже."""
    for j in prange(Ny):
        base = (j + 1) * _PX + 1
        dd = np.empty(_M)
        md = np.empty(_M)
        acc = np.zeros(_K)
        for i in range(Nx):
            f = base + i
            d0, ce, cw, cn, cs = diag[f], ex[f], ex[f - 1], ey[f], ey[f - _PX]
            vf, ve, vw, vn, vs = x[f], x[f + 1], x[f - 1], x[f + _PX], x[f - _PX]
            r1 = rhs[f] - (d0 * vf + ce * ve + cw * vw + cn * vn + cs * vs)
            for a in range(_M):
                h = hist[slots[a]]
                hf, he, hw, hn, hs = h[f], h[f + 1], h[f - 1], h[f + _PX], h[f - _PX]
                dd[a] = vf - hf
                md[a] = d0 * (vf - hf) + ce * (ve - he) + cw * (vw - hw) + cn * (vn - hn) + cs * (vs - hs)
                vf, ve, vw, vn, vs = hf, he, hw, hn, hs
            k = 0
            for a in range(_M):
                for b in range(a, _M):
                    acc[k] += dd[a] * md[b]
                    k += 1
            for a in range(_M):
                acc[k + a] += dd[a] * r1
        part[j, :] = acc


@njit(cache=True)
def _coefficients(part, c):
    """Коэффициенты c из Грама с масштабированием по диагонали; зависимые разности (пока история не накопилась)
    выпадают: Холецкий с отбрасыванием ведущих элементов меньше 1e-14 после масштабирования - уровень округления.
    На расчете ведущие элементы ~1e-9."""
    tot = np.zeros(_K)
    for j in range(part.shape[0]):
        for k in range(_K):
            tot[k] += part[j, k]
    G = np.zeros((_M, _M))
    k = 0
    for a in range(_M):
        for b in range(a, _M):
            G[a, b] = tot[k]
            G[b, a] = tot[k]
            k += 1
    s = np.zeros(_M)
    for a in range(_M):
        if G[a, a] > 0.0:
            s[a] = 1.0 / np.sqrt(G[a, a])
    L = np.zeros((_M, _M))
    keep = np.zeros(_M, np.bool_)
    for a in range(_M):
        if s[a] == 0.0:
            continue
        v = G[a, a] * s[a] * s[a]
        for b in range(a):
            v -= L[a, b] * L[a, b]
        if v <= 1e-14:
            continue
        keep[a] = True
        L[a, a] = np.sqrt(v)
        for i in range(a + 1, _M):
            w = G[i, a] * s[i] * s[a]
            for b in range(a):
                w -= L[i, b] * L[a, b]
            L[i, a] = w / L[a, a]
    y = np.zeros(_M)
    for a in range(_M):
        if keep[a]:
            v = tot[k + a] * s[a]
            for b in range(a):
                v -= L[a, b] * y[b]
            y[a] = v / L[a, a]
    for a in range(_M - 1, -1, -1):
        c[a] = 0.0
        if keep[a]:
            v = y[a]
            for b in range(a + 1, _M):
                v -= L[b, a] * c[b]
            c[a] = v / L[a, a]
    for a in range(_M):
        c[a] *= s[a]


@njit(parallel=True, cache=True)
def _apply(x, hist, slots, c):
    """x0 = x1 + sum c_a*d_a; прошлое решение x1 ложится в кольцо на место самого старого."""
    old = slots[_M - 1]
    for j in prange(Ny):
        base = (j + 1) * _PX + 1
        acc = np.empty(Nx)
        for i in range(Nx):
            acc[i] = x[base + i]
        for a in range(_M):
            v = x if a == 0 else hist[slots[a - 1]]
            h = hist[slots[a]]
            ca = c[a]
            for i in range(Nx):
                f = base + i
                acc[i] += ca * (v[f] - h[f])
        for i in range(Nx):
            f = base + i
            hist[old, f] = x[f]
            x[f] = acc[i]


@njit(cache=True)
def project_guess(diag, ex, ey, rhs, x, hist, state) -> None:
    """Проекция; векторы - с рамкой (`layout`). x на входе - прошлое решение, на выходе - начальное
    приближение; кольцо `hist` (`p_guess_m` строк) сдвигается на шаг."""
    head = int(state[ST_HEAD])
    slots = np.empty(_M, np.int64)
    for a in range(_M):
        slots[a] = (head + a) % _M
    part = np.empty((Ny, _K))
    _gram(diag, ex, ey, rhs, x, hist, slots, part)
    c = np.zeros(_M)
    _coefficients(part, c)
    _apply(x, hist, slots, c)
    state[ST_HEAD] = slots[_M - 1]
