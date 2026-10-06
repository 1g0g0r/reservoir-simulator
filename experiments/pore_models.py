"""Геометрические модели порового пространства - потомки пучка капилляров - и их связь k(m) при отложении.

    python experiments/pore_models.py          # секунды - минута: все модели против пар (m/m0, k/k0) He et al. (2020)
    python experiments/pore_models.py --plot   # только рисунок

Семейства моделей (обзор `resources/всякие статьи/модели ПС/`, docs/кинетика_осаждения.md, разд. 13.13), все с
одинаковым слоем отложения толщины h на стенках (сужение):

  1. пучок капилляров (Kosugi, как `paraphin/geometry.py`): k ~ int r^4 fi, m ~ int r^2 fi;
  2. фрактальный пучок (Yu & Cheng 2002; Liu et al. 2018): число каналов N(>r) = (r_max/r)^D_f, плотность
     fi ~ r^-(D_f+1) на [r_min, r_max];
  3. пучок с периодическими сужениями (Li et al. 2021): проводимость - по горлам радиуса gamma*r0 - h, объем - по
     порам r0 - h; это сеть при z -> inf;
  4. сеть пор и горл, эффективная среда (Kirkpatrick 1973) - то, что считает пакет при `pore_network`;
  5. стохастическая сеть (Fatt 1956; Nemati et al. 2020): кубическая решетка z = 6 со случайными радиусами из
     fi_0, давление решается прямо (разреженная СЛАУ) - проверка приближения эффективной среды;
  6. степенной закон (m/m0)^n - замыкание модели глубинной фильтрации (для сравнения).
Динамическая сеть (Nikooey et al. 2015) - это модели 4-5 с радиусами, меняющимися по отложению: в пакете так и
считается (fi и h переносятся ядром `Deposition.py`, проводимость - по сети).
"""
import json
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from common import mode_from_argv, save_results, load_results, FIGURES
import he2020 as HE

SIGMA_R = 0.4
H_GRID = np.linspace(0.0, 2.5, 300)


def kosugi(r):
    return np.exp(-0.5 * (np.log(r) / SIGMA_R) ** 2) / r


def bundle_curve(r, fi, gam=1.0):
    """Пучок (gam = 1) или пучок с сужениями (gam < 1): объем по порам, проводимость по горлам (параллельно)."""
    out = []
    for h in H_GRID:
        rt = np.maximum(gam * r - h, 0.0)
        out.append((np.sum(np.maximum(r - h, 0.0) ** 2 * fi) / np.sum(r ** 2 * fi),
                    np.sum(rt ** 4 * fi) / np.sum((gam * r) ** 4 * fi)))
        if out[-1][1] <= 0.0:
            break
    return np.array(out)


def fractal(d_f, ratio=0.01):
    """Фрактальный пучок: fi ~ r^-(D_f+1) на [ratio*r_max, r_max] (в единицах медианы Kosugi, r_max = 3.3)."""
    r = np.linspace(3.3 * ratio, 3.3, 4000)
    return r, r ** (-(d_f + 1.0))


def ema_curve(r, fi, gam, z):
    from paraphin.equations.Kinetics_math import ema_conductance
    w0 = fi / fi.sum()
    g0 = ema_conductance((gam * r) ** 4, w0, 0.0, z)
    out = []
    for h in H_GRID:
        rt = gam * r - h
        op = rt > 0.0
        if not op.any():
            break
        gm = ema_conductance(np.where(op, rt, 0.0) ** 4, np.where(op, w0, 0.0), float(w0[~op].sum()), z)
        out.append((np.sum(np.maximum(r - h, 0.0) ** 2 * fi) / np.sum(r ** 2 * fi), gm / g0))
    return np.array(out)


def lattice_curve(gam, n=14, seeds=3):
    """Стохастическая сеть: кубическая решетка n^3 узлов (z = 6), радиусы пор связей - выборка из Kosugi, горло
    gam*r0 - h; перепад вдоль x, проводимость сети - расход при единичном перепаде. Среднее по реализациям."""
    r_grid = np.linspace(1e-3, 5.0, 4000)
    cdf = np.cumsum(kosugi(r_grid))
    cdf /= cdf[-1]
    idx = np.arange(n ** 3).reshape(n, n, n)
    bonds = []
    for ax in range(3):
        a = idx.take(range(n - 1), axis=ax).ravel()
        b = idx.take(range(1, n), axis=ax).ravel()
        bonds.append(np.column_stack([a, b]))
    bonds = np.vstack(bonds)
    x_of = (np.arange(n ** 3) // (n * n))
    curves = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        r0 = np.interp(rng.random(len(bonds)), cdf, r_grid)

        def flow(g):
            # закрытые горла - проводимость 1e-12 от наибольшей: изолированные узлы не делают матрицу вырожденной,
            # а расход через них - на 12 порядков ниже
            g = np.maximum(g, 1e-12 * max(float(g.max()), 1e-300))
            a, b = bonds[:, 0], bonds[:, 1]
            diag = np.bincount(a, g, n ** 3) + np.bincount(b, g, n ** 3)
            inlet, outlet = x_of == 0, x_of == n - 1
            fixed = inlet | outlet
            free = ~fixed
            p_fix = np.where(inlet, 1.0, 0.0)
            rows = np.concatenate([a, b, np.arange(n ** 3)])
            cols = np.concatenate([b, a, np.arange(n ** 3)])
            vals = np.concatenate([-g, -g, diag + 1e-30])
            mat = coo_matrix((vals, (rows, cols)), shape=(n ** 3, n ** 3)).tocsr()
            rhs = -(mat[:, fixed] @ p_fix[fixed])
            p = p_fix.copy()
            p[free] = spsolve(mat[free][:, free].tocsc(), rhs[free])
            ga = g[(x_of[a] == 0) & (x_of[b] == 1)]
            pb = p[b[(x_of[a] == 0) & (x_of[b] == 1)]]
            return float(np.sum(ga * (1.0 - pb)))

        q0 = flow((gam * r0) ** 4)
        out = []
        for h in H_GRID[::6]:
            rt = np.maximum(gam * r0 - h, 0.0)
            q = flow(rt ** 4) if rt.any() else 0.0
            out.append((np.sum(np.maximum(r0 - h, 0.0) ** 2) / np.sum(r0 ** 2), q / q0))
            if q <= 1e-6 * q0:
                break
        curves.append(np.array(out))
    m = np.linspace(1.0, 0.0, 101)
    k = np.mean([np.interp(-m, -c[:, 0], c[:, 1], right=0.0) for c in curves], axis=0)
    keep = m >= min(c[:, 0].min() for c in curves)
    return np.column_stack([m[keep], k[keep]])


def fit_gamma(make, pts, metric):
    res = minimize_scalar(lambda g: metric(make(g), pts), bounds=(0.2, 0.95), method='bounded', options=dict(xatol=2e-3))
    return float(res.x), float(res.fun)


def run(mode: str = 'full') -> dict:
    if mode == 'plot':
        out = load_results('pore_models')
        plot(out)
        return out
    data = HE.pairs()
    pts = [(m, k) for rows in data.values() for m, k, _ in rows]
    lin = lambda c, p: HE.lin_error(c, p)
    lg = lambda c, p: HE.log_error(c, p)[0]
    r = np.linspace(1e-3, 5.0, 4000)
    fi = kosugi(r)
    curves, rows = {}, []

    def add(name, curve, n_par, note):
        curves[name] = curve
        rows.append(dict(name=name, lin=lin(curve, pts), lg=lg(curve, pts), n_par=n_par, note=note))
        print(f'{name}: СКО k/k0 {rows[-1]["lin"]:.3f}, lg {rows[-1]["lg"]:.3f} ({note})', flush=True)

    add('1. пучок капилляров', bundle_curve(r, fi), 0, 'без подбора')
    res = minimize_scalar(lambda d: lin(bundle_curve(*fractal(d)), pts), bounds=(0.5, 2.5), method='bounded')
    add(f'2. фрактальный пучок, D_f = {res.x:.2f}', bundle_curve(*fractal(res.x)), 1, 'подбор D_f')
    g_c, _ = fit_gamma(lambda g: bundle_curve(r, fi, g), pts, lin)
    add(f'3. пучок с сужениями, горло {g_c:.2f}', bundle_curve(r, fi, g_c), 1, 'подбор горла')
    add('4. сеть, эффективная среда, z = 6, горло 0.4', ema_curve(r, fi, 0.4, 6.0), 0, 'по умолчанию пакета')
    g_n, _ = fit_gamma(lambda g: ema_curve(r, fi, g, 6.0), pts, lin)
    add(f'4. сеть, эффективная среда, z = 6, горло {g_n:.2f}', ema_curve(r, fi, g_n, 6.0), 1, 'подбор горла')
    lat = lattice_curve(g_n)
    add(f'5. стохастическая решетка z = 6, горло {g_n:.2f}', lat, 0, 'проверка эффективной среды')
    lm, lk = np.log(np.array(pts)).T
    n_fit = float(np.sum(lm * lk) / np.sum(lm * lm))
    mg = np.linspace(0.3, 1.0, 141)
    add(f'6. степенной закон, n = {n_fit:.1f}', np.column_stack([mg, mg ** n_fit]), 1, 'подбор n')
    # точность эффективной среды против решетки на одной кривой
    ema_on = ema_curve(r, fi, g_n, 6.0)
    m_common = np.linspace(max(lat[:, 0].min(), ema_on[:, 0].min()), 1.0, 50)
    k_lat = np.interp(-m_common, -lat[:, 0], lat[:, 1])
    k_ema = np.interp(m_common, ema_on[::-1, 0], ema_on[::-1, 1])
    ema_vs_lattice = float(np.sqrt(np.mean((k_ema - k_lat) ** 2)))
    print(f'эффективная среда против решетки: СКО k/k0 {ema_vs_lattice:.3f}', flush=True)
    out = dict(pairs={w: [list(p) for p in rr] for w, rr in data.items()}, rows=rows,
               curves={k: v.tolist() for k, v in curves.items()}, ema_vs_lattice=ema_vs_lattice)
    save_results('pore_models', out)
    plot(out)
    return out


def summary(out) -> list:
    return [(r['name'], r['lin'], r['lg'], r['n_par'], r['note']) for r in out['rows']]


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for (well, rows), marker in zip(out['pairs'].items(), ('^', 'D', 's', 'v')):
        p = np.array(rows)
        ax.semilogy(p[:, 0], p[:, 1], marker, color='k', mfc='white', ms=4.5, label=f'He et al., {well}')
    styles = (':', '--', '-.', (0, (5, 1, 1, 1)), '-', (0, (1, 1)), (0, (6, 2)))
    for (name, curve), style in zip(out['curves'].items(), styles):
        c = np.array(curve)
        o = np.argsort(c[:, 0])
        ax.semilogy(c[o, 0], np.maximum(c[o, 1], 1e-3), ls=style, lw=1.6, label=name)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(5e-3, 1.3)
    ax.set_xlabel('m / m₀')
    ax.set_ylabel('k / k₀')
    ax.legend(fontsize=6, loc='lower right', handlelength=3.0)
    fig.tight_layout()
    fig.savefig(FIGURES / 'pore_models.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run(mode_from_argv())
