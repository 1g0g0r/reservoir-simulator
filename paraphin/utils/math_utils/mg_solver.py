"""Решатель уравнения давления: PCG с многосеточным предобуславливателем (V-цикл).

Матрица - положительно определенная форма `M = -A` (`equations/Pressure.py`): диагональ положительна, связи с соседями
отрицательны, правая часть с обратным знаком. M симметрична (гармоническое среднее подвижностей симметрично по
ячейкам), диагонально доминантна и неприводима, скважины и Дирихле дают строгое доминирование - значит, положительно
определена. Пятиточечный шаблон хранится тремя диагоналями: центр, связь с соседом по x, связь с соседом по y.

Прежде были еще ленточный Холецкий (до 50x50) и двусторонняя факторизация (50-100): с проекцией начального
приближения и работой без копий многосеточный быстрее их на всех сетках 50-150 (docs/PERFORMANCE_FINDINGS.md,
«Решатель давления»), и они удалены. Цена V-цикла линейна по N, Холецкого - ~Nx^3*Ny.

Грубые уровни - агрегация блоков 2x2 с кусочно-постоянным продолжением P. Оператор Галеркина P^T M P при
такой P - снова пятиточечный шаблон: диагональ блока - сумма диагоналей и удвоенных внутренних связей, связь
блоков - сумма связей через их общую грань. Поэтому все уровни хранятся так же, как сама матрица, - тремя
диагоналями, в плоских массивах подряд. Пересборка операторов - O(N).

Каждый уровень хранится с рамкой фиктивных ячеек: ячейка (i, j) уровня l лежит в `_OFF[l] + (j+1)*(nx+2) + i+1`.
У фиктивных ячеек связи и решение нулевые, поэтому во внутренних циклах нет проверок границ - иначе четыре
ветвления на ячейку срывают векторизацию (V-цикл был ~100 тактов на ячейку при ~40 флопах).

Сглаживание - красно-черный Гаусс-Зейдель (красные - (i + j) четное): до спуска красные, затем черные, после
подъема в обратном порядке, так что V-цикл - симметричный оператор и годится предобуславливателем для CG.
Продолжение без сглаживания (unsmoothed aggregation) недооценивает грубую поправку, ее умножают на `p_mg_scale`
(~1.7). Самый грубый уровень (<= 64 ячеек) решается ленточным Холецким. На сетке в одну-две ячейки по одной из
сторон (одномерный керн) огрублять нечего: уровень один, и он же решается Холецким точно (полуширина ленты - Nx). Проходов по памяти на уровень - 2.5 вместо 7:
  - решение уровня не обнуляется: первый красный полушаг при нулевом приближении - просто x = b/D;
  - после черного полушага невязка черных ячеек равна нулю точно (они только что решены), поэтому невязка
    считается только в двух красных ячейках каждого блока 2x2 и сразу суммируется в грубый уровень (P^T r);
  - продолжения отдельным проходом нет: черный полушаг после подъема берет красных соседей с поправкой грубого
    уровня «на лету», а красным ячейкам поправка не нужна - следующий красный полушаг их перезапишет, не глядя на
    старое значение. Математически это тот же V-цикл: x += P*xc, затем черный и красный полушаги.

Каждое ядро уровня есть в двух версиях - параллельной и последовательной: запуск параллельного цикла стоит
десятки микросекунд, и на мелких уровнях это дороже самой работы. Параллельная берется на уровнях не меньше
`p_mg_par_min` ячеек.

Копий нет совсем. Мелкий уровень буфера - это сама матрица: `calc_pressure` собирает ее прямо в `buf[_D/_EX/_EY]`
(векторы давления - с той же рамкой, `layout.PX`, `layout.P0`), а невязка и
предобусловленная невязка PCG - строки `buf[_B]` и `buf[_X]` мелкого уровня: V-цикл берет правую часть оттуда же и
туда же кладет результат. Векторные операции PCG слиты в три параллельных прохода на итерацию: q = M z + beta*q
вместе с p = z + beta*p и p.q (M p не нужен: M (z + beta*p) = M z + beta*q), обновление x, r с |r|^2 и r.z.

Грубые уровни устаревают медленно (матрица меняется на ~0.2 % за шаг), поэтому пересобираются не каждый шаг, а по
правилу цены (`solve_mg_system`): средняя цена шага с пересборки (`p_mg_setup_ratio` + сумма итераций)/возраст
сначала падает, а когда итераций очередного шага больше нее - уровни пора пересобрать. Цена пересборки в итерациях -
`p_mg_setup_ratio`. Мелкий уровень всегда свежий - это сама матрица; V-цикл с устаревшими грубыми уровнями
остается симметричным и годится предобуславливателем.
"""
import numpy as np
from numba import njit, prange

from paraphin.constants import Nx, Ny, p_pcg_rtol, p_mg_maxit, p_mg_scale, p_mg_par_min, p_mg_setup_ratio
_PX = Nx + 2  # шаг строки мелкого уровня

# Состояние решателя между шагами - массив `Solver.p_state`
ST_AGE, ST_COST, ST_REFACTOR = 0, 1, 2  # возраст уровней в шагах, сумма итераций с пересборки, флаг пересборки
ST_HEAD = 3                             # новейшее решение в кольце прошлых решений (`guess.py`)
P_STATE = 4


def _levels(nx, ny, n_coarse=64):
    nxs, nys = [nx], [ny]
    while nxs[-1] * nys[-1] > n_coarse and nxs[-1] > 2 and nys[-1] > 2:
        nxs.append((nxs[-1] + 1) // 2)
        nys.append((nys[-1] + 1) // 2)
    off = np.zeros(len(nxs) + 1, np.int64)
    for lev in range(len(nxs)):
        off[lev + 1] = off[lev] + (nxs[lev] + 2) * (nys[lev] + 2)  # с рамкой фиктивных ячеек
    return np.array(nxs, np.int64), np.array(nys, np.int64), off


_NXS, _NYS, _OFF = _levels(Nx, Ny)
_NL = _NXS.size
MG_TOTAL = int(_OFF[-1])                      # ячеек на всех уровнях с рамками: длина рабочих массивов `mg_buf`
MG_COARSE = int(_NXS[-1] * _NYS[-1])          # ячеек самого грубого уровня (без рамки)
MG_COARSE_KD = int(_NXS[-1])                  # полуширина его ленты
MG_ROWS = 5                                   # строки `mg_buf`: D, EX, EY, правая часть, решение
_D, _EX, _EY, _B, _X = range(MG_ROWS)


def _coarsen(D, EX, EY, lev):
    """Оператор Галеркина уровня lev+1 по уровню lev (агрегация 2x2); пишет только внутренние ячейки."""
    nx, ny, o, px = _NXS[lev], _NYS[lev], _OFF[lev], _NXS[lev] + 2
    nxc, nyc, oc, pxc = _NXS[lev + 1], _NYS[lev + 1], _OFF[lev + 1], _NXS[lev + 1] + 2
    for J in prange(nyc):
        for I in range(nxc):
            d, exc, eyc = 0.0, 0.0, 0.0
            for dj in range(2):
                j = 2 * J + dj
                if j >= ny:
                    continue
                for di in range(2):
                    i = 2 * I + di
                    if i >= nx:
                        continue
                    f = o + (j + 1) * px + i + 1
                    d += D[f]
                    if di == 0:
                        d += 2.0 * EX[f]  # связь внутри блока (у последнего столбца она нулевая)
                    else:
                        exc += EX[f]      # через правую грань блока
                    if dj == 0:
                        d += 2.0 * EY[f]
                    else:
                        eyc += EY[f]
            c = oc + (J + 1) * pxc + I + 1
            D[c] = d
            EX[c] = exc
            EY[c] = eyc


def _red_init(D, lev, b, x):
    """Первый красный полушаг при нулевом приближении: x = b/D в красных ячейках."""
    nx, ny, o, px = _NXS[lev], _NYS[lev], _OFF[lev], _NXS[lev] + 2
    for j in prange(ny):
        base = o + (j + 1) * px + 1
        for i in range(j % 2, nx, 2):
            x[base + i] = b[base + i] / D[base + i]


def _sweep(D, EX, EY, lev, b, x, color):
    """Полушаг Гаусса-Зейделя по ячейкам цвета color (0 - красные, (i + j) четное), без ветвлений."""
    nx, ny, o, px = _NXS[lev], _NYS[lev], _OFF[lev], _NXS[lev] + 2
    for j in prange(ny):
        base = o + (j + 1) * px + 1
        for i in range((color + j) % 2, nx, 2):
            f = base + i
            x[f] = (b[f] - EX[f] * x[f + 1] - EX[f - 1] * x[f - 1] - EY[f] * x[f + px] - EY[f - px] * x[f - px]) / D[f]


def _red_residual_restrict(D, EX, EY, lev, b, x):
    """Правая часть уровня lev+1 = P^T r: в блоке 2x2 две красные ячейки (0,0) и (1,1), невязка черных - ноль.
    Красная ячейка за краем нечетной сетки - фиктивная: у нее b, x и связи нулевые, и вклад нулевой. Тем же проходом -
    первый красный полушаг уровня lev+1 (x = b/D): на одну параллельную область и один проход меньше."""
    o, px = _OFF[lev], _NXS[lev] + 2
    nxc, nyc, oc, pxc = _NXS[lev + 1], _NYS[lev + 1], _OFF[lev + 1], _NXS[lev + 1] + 2
    for J in prange(nyc):
        for I in range(nxc):
            f0 = o + (2 * J + 1) * px + 2 * I + 1
            f1 = f0 + px + 1
            r0 = b[f0] - D[f0] * x[f0] - EX[f0] * x[f0 + 1] - EX[f0 - 1] * x[f0 - 1] - EY[f0] * x[f0 + px] \
                - EY[f0 - px] * x[f0 - px]
            r1 = b[f1] - D[f1] * x[f1] - EX[f1] * x[f1 + 1] - EX[f1 - 1] * x[f1 - 1] - EY[f1] * x[f1 + px] \
                - EY[f1 - px] * x[f1 - px]
            c = oc + (J + 1) * pxc + I + 1
            b[c] = r0 + r1
            if (I + J) % 2 == 0:
                x[c] = b[c] / D[c]


def _black_corrected(D, EX, EY, lev, b, x):
    """Черный полушаг после подъема: красные соседи берутся с поправкой грубого уровня, x_r + p_mg_scale*x_c.
    Соседи за краем - фиктивные (x = 0, связь нулевая); их блок на грубом уровне может быть настоящим, но связь
    все равно нулевая."""
    nx, ny, o, px = _NXS[lev], _NYS[lev], _OFF[lev], _NXS[lev] + 2
    oc, pxc = _OFF[lev + 1], _NXS[lev + 1] + 2
    for j in prange(ny):
        base = o + (j + 1) * px + 1
        cj = oc + (j // 2 + 1) * pxc + 1         # строка грубого уровня самой ячейки
        cn = oc + ((j + 1) // 2 + 1) * pxc + 1   # соседа сверху
        cs = oc + ((j - 1) // 2 + 1) * pxc + 1   # соседа снизу (j = 0: -1 // 2 = -1 - рамка грубого уровня)
        for i in range((1 + j) % 2, nx, 2):
            f = base + i
            xe = x[f + 1] + p_mg_scale * x[cj + (i + 1) // 2]
            xw = x[f - 1] + p_mg_scale * x[cj + (i - 1) // 2]
            xn = x[f + px] + p_mg_scale * x[cn + i // 2]
            xs = x[f - px] + p_mg_scale * x[cs + i // 2]
            x[f] = (b[f] - EX[f] * xe - EX[f - 1] * xw - EY[f] * xn - EY[f - px] * xs) / D[f]


# одно тело - две версии: prange в njit без parallel - обычный range
_coarsen_p, _coarsen_s = njit(parallel=True, cache=True)(_coarsen), njit(cache=True)(_coarsen)
_init_p, _init_s = njit(parallel=True, cache=True)(_red_init), njit(cache=True)(_red_init)
_sweep_p, _sweep_s = njit(parallel=True, cache=True)(_sweep), njit(cache=True)(_sweep)
_rr_p, _rr_s = njit(parallel=True, cache=True)(_red_residual_restrict), njit(cache=True)(_red_residual_restrict)
_bc_p, _bc_s = njit(parallel=True, cache=True)(_black_corrected), njit(cache=True)(_black_corrected)


@njit(cache=True)
def _par(lev):
    return _NXS[lev] * _NYS[lev] >= p_mg_par_min


@njit(cache=True)
def _coarse_pack_chol(D, EX, EY, wc):
    """Ленточный Холецкий самого грубого уровня (без рамки, полуширина ленты - его nx)."""
    lev = _NL - 1
    nx, ny, o, px = _NXS[lev], _NYS[lev], _OFF[lev], _NXS[lev] + 2
    n, kd = MG_COARSE, MG_COARSE_KD
    wc[:, :] = 0.0
    for j in range(ny):
        for i in range(nx):
            f, k = o + (j + 1) * px + i + 1, i + j * nx
            wc[k, 0] = D[f]
            wc[k, 1] = EX[f]
            if k + kd < n:
                wc[k, kd] = EY[f]
    for j in range(n):
        v = wc[j, 0]
        if v <= 0.0:
            return j + 1
        v = np.sqrt(v)
        wc[j, 0] = v
        mm = kd if j + kd < n else n - 1 - j
        for dd in range(1, mm + 1):
            wc[j, dd] /= v
        for d1 in range(1, mm + 1):
            f = wc[j, d1]
            if f != 0.0:
                row = j + d1
                for d2 in range(d1, mm + 1):
                    wc[row, d2 - d1] -= f * wc[j, d2]
    return 0


@njit(cache=True)
def _coarse_solve(wc, b, x):
    lev = _NL - 1
    nx, ny, o, px = _NXS[lev], _NYS[lev], _OFF[lev], _NXS[lev] + 2
    n, kd = MG_COARSE, MG_COARSE_KD
    y = np.empty(n)
    for j in range(ny):
        for i in range(nx):
            y[i + j * nx] = b[o + (j + 1) * px + i + 1]
    for j in range(n):
        v = y[j] / wc[j, 0]
        y[j] = v
        mm = kd if j + kd < n else n - 1 - j
        for dd in range(1, mm + 1):
            y[j + dd] -= wc[j, dd] * v
    for j in range(n - 1, -1, -1):
        acc = y[j]
        mm = kd if j + kd < n else n - 1 - j
        for dd in range(1, mm + 1):
            acc -= wc[j, dd] * y[j + dd]
        y[j] = acc / wc[j, 0]
    for j in range(ny):
        for i in range(nx):
            x[o + (j + 1) * px + i + 1] = y[i + j * nx]


@njit(cache=True)
def mg_setup(buf, wc) -> int:
    """Операторы грубых уровней по мелкому (он же матрица) и фактор самого грубого. 0 - успех.

    Рамки фиктивных ячеек не пишет никто, поэтому они остаются нулями, какими `buf` создан (`Solver.__init__`):
    связи и решение нулевые, на диагональ рамки никто не делит. Обнулять их каждый шаг не нужно."""
    D, EX, EY = buf[_D], buf[_EX], buf[_EY]
    for lev in range(_NL - 1):
        if _par(lev):
            _coarsen_p(D, EX, EY, lev)
        else:
            _coarsen_s(D, EX, EY, lev)
    return _coarse_pack_chol(D, EX, EY, wc)


@njit(cache=True)
def mg_vcycle(buf, wc) -> None:
    """buf[_X] = V(buf[_B]) на мелком уровне."""
    D, EX, EY, b, x = buf[_D], buf[_EX], buf[_EY], buf[_B], buf[_X]
    if _par(0):  # первый красный полушаг мелкого уровня; на грубых его делает сжатие
        _init_p(D, 0, b, x)
    else:
        _init_s(D, 0, b, x)
    for lev in range(_NL - 1):
        if _par(lev):
            _sweep_p(D, EX, EY, lev, b, x, 1)
            _rr_p(D, EX, EY, lev, b, x)
        else:
            _sweep_s(D, EX, EY, lev, b, x, 1)
            _rr_s(D, EX, EY, lev, b, x)
    _coarse_solve(wc, b, x)
    for lev in range(_NL - 2, -1, -1):
        if _par(lev):
            _bc_p(D, EX, EY, lev, b, x)
            _sweep_p(D, EX, EY, lev, b, x, 0)
        else:
            _bc_s(D, EX, EY, lev, b, x)
            _sweep_s(D, EX, EY, lev, b, x, 0)


# Векторные операции PCG на мелком уровне: только внутренние ячейки, рамка остается нулевой
@njit(parallel=True, cache=True)
def _residual(D, EX, EY, rhs, x, r):
    """r = rhs - M x; возвращает |rhs|^2 и |r|^2."""
    bb, rr = 0.0, 0.0
    for j in prange(Ny):
        base = (j + 1) * _PX + 1
        for i in range(Nx):
            f = base + i
            v = rhs[f] - (D[f] * x[f] + EX[f] * x[f + 1] + EX[f - 1] * x[f - 1] + EY[f] * x[f + _PX]
                          + EY[f - _PX] * x[f - _PX])
            r[f] = v
            bb += rhs[f] * rhs[f]
            rr += v * v
    return bb, rr


@njit(parallel=True, cache=True)
def _direction(D, EX, EY, z, p, q, beta):
    """p = z + beta*p, q = M z + beta*q (= M p); возвращает p.q."""
    pq = 0.0
    for j in prange(Ny):
        base = (j + 1) * _PX + 1
        for i in range(Nx):
            f = base + i
            mz = D[f] * z[f] + EX[f] * z[f + 1] + EX[f - 1] * z[f - 1] + EY[f] * z[f + _PX] + EY[f - _PX] * z[f - _PX]
            pv = z[f] + beta * p[f]
            qv = mz + beta * q[f]
            p[f] = pv
            q[f] = qv
            pq += pv * qv
    return pq


@njit(parallel=True, cache=True)
def _update(x, r, p, q, alpha):
    """x += alpha*p, r -= alpha*q; возвращает |r|^2."""
    rr = 0.0
    for j in prange(Ny):
        base = (j + 1) * _PX + 1
        for i in range(Nx):
            f = base + i
            x[f] += alpha * p[f]
            v = r[f] - alpha * q[f]
            r[f] = v
            rr += v * v
    return rr


@njit(parallel=True, cache=True)
def _dot(a, b):
    s = 0.0
    for j in prange(Ny):
        base = (j + 1) * _PX + 1
        for i in range(Nx):
            s += a[base + i] * b[base + i]
    return s


@njit(cache=True)
def pcg_mg(rhs, x, p, q, buf, wc) -> int:
    """PCG с V-циклом. Матрица - мелкий уровень `buf`, невязка - `buf[_B]`, предобусловленная - `buf[_X]`. Стартует
    с `x`; возвращает число итераций или -1, если не сошелся за `p_mg_maxit`."""
    D, EX, EY, r, z = buf[_D], buf[_EX], buf[_EY], buf[_B], buf[_X]
    bb, rn = _residual(D, EX, EY, rhs, x, r)
    nb = np.sqrt(bb)
    if np.sqrt(rn) <= p_pcg_rtol * nb:
        return 0
    mg_vcycle(buf, wc)
    rz = _dot(r, z)
    beta = 0.0
    for it in range(p_mg_maxit):
        alpha = rz / _direction(D, EX, EY, z, p, q, beta)
        rn = _update(x, r, p, q, alpha)
        if np.sqrt(rn) <= p_pcg_rtol * nb:
            return it + 1
        mg_vcycle(buf, wc)
        rz_new = _dot(r, z)
        beta = rz_new / rz
        rz = rz_new
    return -1


@njit(cache=True)
def _setup_or_raise(buf, wc, state) -> None:
    if mg_setup(buf, wc) != 0:
        raise ValueError('Матрица давления вырождена: неположительный ведущий элемент на грубом уровне. '
                         'Скорее всего в какой-то ячейке обнулилась суммарная подвижность.')
    state[ST_AGE] = 0.0
    state[ST_COST] = 0.0
    state[ST_REFACTOR] = 0.0


@njit(cache=True)
def solve_mg_system(rhs, x, p, q, buf, wc, state) -> int:
    """Решение M x = rhs многосеточным PCG; матрица - в мелком уровне `buf`, x на входе - начальное приближение.
    Векторы - с рамкой (`layout.PX`). `state` - `Solver.p_state`: грубые уровни пересобираются по правилу цены,
    цена пересборки - `p_mg_setup_ratio` итераций. Возвращает число итераций."""
    fresh = state[ST_AGE] == 0.0 or state[ST_REFACTOR] != 0.0
    if fresh:
        _setup_or_raise(buf, wc, state)
    it = pcg_mg(rhs, x, p, q, buf, wc)
    if it < 0 and not fresh:  # не помог предобуславливатель с устаревшими уровнями - пересборка
        _setup_or_raise(buf, wc, state)
        it = pcg_mg(rhs, x, p, q, buf, wc)
    if it < 0:
        raise ValueError('Многосеточный PCG не сошелся за p_mg_maxit итераций.')
    cost = float(it)
    state[ST_AGE] += 1.0
    state[ST_COST] += cost
    state[ST_REFACTOR] = 1.0 if cost * state[ST_AGE] > p_mg_setup_ratio + state[ST_COST] else 0.0
    return it


def mg_pad(v, out) -> None:
    """Вектор подряд (idx = i + j*Nx) - в раскладку мелкого уровня с рамкой (для тестов)."""
    out[:_OFF[1]].reshape(Ny + 2, _PX)[1:-1, 1:-1] = np.asarray(v).reshape(Ny, Nx)


def mg_unpad(v):
    """Обратно к раскладке подряд (копия)."""
    return v[:_OFF[1]].reshape(Ny + 2, _PX)[1:-1, 1:-1].ravel()
