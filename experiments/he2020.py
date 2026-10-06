"""Связь потери пористости и проницаемости при отложении парафина (He et al., 2020): проверка без подбора.

    python experiments/he2020.py          # секунды: аналитика по пучку fi_0 -> results/he2020.json + рисунок
                                          # (C0 гель-отложения берется из results/sandyga2020.json)
    python experiments/he2020.py --plot   # только рисунок

Керны четырех скважин месторождения Чанчуньлин охлаждались от 50 до 2-5 C с прокачкой парафинистой нефти; при
каждой температуре измерены пористость (гелиевый порозиметр) и эффективная проницаемость по нефти
(`data/he2020.json`). Кинетики в статье нет, есть пары (m/m0, k/k0) - то, что задает связь проницаемости с
отложением. С ними сравниваются (`docs/кинетика_осаждения.md`, разд. 13.9-13.10):
  - Козени-Карман и степенной закон (m/m0)^n - замыкания модели глубинной фильтрации (`perm_model`); показатель
    n подбирается по точкам и сравнивается с диапазоном 8-19 из обзора (разд. 4.2);
  - пучок капилляров, сужение всех каналов слоем кристаллов (C0 = 1, прежняя модель);
  - сеть пор и горл (`pore_network`, разд. 13.13): поры fi(r) соединены горлами радиуса gamma*r, сеть с
    координационным числом z, проводимость - по эффективной среде (Kirkpatrick 1973, `ema_conductance` пакета).
    Слой отложения закрывает горла раньше, чем заполняет поры, а закрытые горла сеть обходит до порога
    протекания 2/z. Априорные z = 6, gamma = 0.4 (по умолчанию пакета) - прогноз; подбор - для сравнения;
  - пучок капилляров с гель-отложением (`deposit_aging`): сужение идет по объему геля, а пористость теряет только
    кристаллы, m/m0 = 1 - C0*(1 - m_c/m0). Порозиметр видит полную пористость: нефть, захваченную гелем, при
    подготовке образца вымывают. C0 - та же доля парафина в свежем геле, что подобрана по опыту Sandyga et al.
    (`results/sandyga2020.json`), здесь - прогноз; для сравнения показан и подбор C0 по точкам He.
"""
import json
import sys

import numpy as np
from scipy.optimize import minimize_scalar

from common import load, mode_from_argv, save_results, load_results, RESULTS, FIGURES

DATA = load('he2020')
M0_HE = 0.27        # пористость кернов до опыта (26.5-27.5 %)
C0_DEFAULT = 0.1    # доля парафина в свежем геле, если подбора по Sandyga еще нет (`kinetics_params.DEFAULTS['AGE_C0']`)
SIGMA_R = 0.4       # ширина fi_0 модели (paraphin/geometry.py); связь k(m) при сужении от r_m не зависит


def pairs():
    """{скважина: [(m/m0, k/k0, T)]} при общих температурах; опорная точка - самая горячая из общих."""
    por, perm = DATA['porosity_percent'], DATA['permeability_mD']
    out = {}
    for well in por:
        common = sorted({int(t) for t in por[well]} & {int(t) for t in perm[well]}, reverse=True)
        m0, k0 = por[well][str(common[0])], perm[well][str(common[0])]
        out[well] = [(por[well][str(t)] / m0, perm[well][str(t)] / k0, t) for t in common[1:]]
    return out


def narrowing(c0: float):
    """(m/m0, k/k0) при сужении всех каналов пучка fi_0 слоем одной толщины; слой - гель с долей кристаллов c0."""
    r = np.linspace(1e-3, 5.0, 20000)  # в единицах r_m
    fi = np.exp(-0.5 * (np.log(r) / SIGMA_R) ** 2) / r
    m_tot, k_tot = np.sum(r ** 2 * fi), np.sum(r ** 4 * fi)
    out = []
    for x in np.linspace(0.0, 3.0, 400):
        rr = np.maximum(r - x, 0.0)
        m_c = np.sum(rr ** 2 * fi) / m_tot
        out.append((1.0 - c0 * (1.0 - m_c), np.sum(rr ** 4 * fi) / k_tot))
    return np.array(out)


NET_PRIOR = (6.0, 0.4)  # z, gamma - по умолчанию пакета (`kinetics_params`: NET_Z, NET_GAMMA = gamma)


def network(z: float, gam: float):
    """(m/m0, k/k0) сети пор и горл при сужении слоем толщины h: поры r - h (объем), горла gamma*r - h
    (проводимость ~ r_t^4), закрытые горла - проводимость 0; эффективная среда с координационным числом z."""
    from paraphin.equations.Kinetics_math import ema_conductance
    r = np.linspace(1e-3, 5.0, 4000)
    fi = np.exp(-0.5 * (np.log(r) / SIGMA_R) ** 2) / r
    w0 = fi / fi.sum()
    g0 = ema_conductance((gam * r) ** 4, w0, 0.0, z)
    out = []
    for h in np.linspace(0.0, 2.5, 400):
        rt = gam * r - h
        open_ = rt > 0.0
        if not open_.any():
            break
        g = np.where(open_, rt, 0.0) ** 4
        w = np.where(open_, w0, 0.0)
        gm = ema_conductance(g, w, float(w0[~open_].sum()), z)
        out.append((np.sum(np.maximum(r - h, 0.0) ** 2 * fi) / np.sum(r ** 2 * fi), gm / g0))
    return np.array(out)


def kozeny_carman(m_rel, m0=M0_HE):
    return m_rel ** 3 * ((1.0 - m0) / (1.0 - m_rel * m0)) ** 2


def log_error(curve, points):
    """СКО lg(k/k0) кривой k(m) от точек, до которых кривая доходит, и число точек, до которых не доходит
    (гель с долей кристаллов C0 не опускает полную пористость ниже 1 - C0: сравнивать там не с чем)."""
    order = np.argsort(curve[:, 0])
    m_c, k_c = curve[order, 0], np.maximum(curve[order, 1], 1e-6)
    pts = np.array(points)
    reach = pts[:, 0] >= m_c[0]
    if not reach.any():
        return float('nan'), int(len(pts))
    model = np.interp(pts[reach, 0], m_c, k_c)
    return float(np.sqrt(np.mean((np.log10(model) - np.log10(pts[reach, 1])) ** 2))), int((~reach).sum())


def lin_error(curve, points):
    """СКО k/k0 (в линейной шкале - та же метрика, что у кривых k/k0(PV) остальных опытов) по досягаемым точкам."""
    order = np.argsort(curve[:, 0])
    pts = np.array(points)
    reach = pts[:, 0] >= curve[order[0], 0]
    if not reach.any():
        return float('nan')
    model = np.interp(pts[reach, 0], curve[order, 0], curve[order, 1])
    return float(np.sqrt(np.mean((model - pts[reach, 1]) ** 2)))


def c0_from_sandyga():
    path = RESULTS / 'sandyga2020.json'
    if path.is_file():
        best = json.loads(path.read_text(encoding='utf-8')).get('best')
        if best and 'c0' in best:
            return float(best['c0']), 'подбор по Sandyga et al.'
    return C0_DEFAULT, 'kinetics_params AGE_C0 (подбора по Sandyga нет)'


def run(mode: str = 'full') -> dict:
    """Режимы 'full' и 'quick' здесь одинаковы: прогонов нет, все считается за секунды."""
    if mode == 'plot':
        out = load_results('he2020')
        plot(out)
        return out
    data = pairs()
    pts = [(m, k) for rows in data.values() for m, k, _ in rows]
    lm, lk = np.log(np.array(pts)).T
    n_fit = float(np.sum(lm * lk) / np.sum(lm * lm))  # lg(k/k0) = n*lg(m/m0), МНК через начало координат
    m_grid = np.linspace(0.3, 1.0, 141)
    c0, c0_source = c0_from_sandyga()
    # C0 по точкам He - только среди тех, при которых кривая доходит до всех точек (C0 > 1 - min m/m0)
    c0_min = 1.0 - min(m for m, _ in pts) + 1e-3
    fit = minimize_scalar(lambda c: log_error(narrowing(c), pts)[0], bounds=(c0_min, 1.0), method='bounded')
    # z и gamma вдоль оврага взаимозаменяемы (при z -> inf сеть вырождается в пучок с узкими горлами), поэтому
    # z - в диапазоне песчаников 3-8, при каждом z подбирается gamma
    # Подбор gamma - по СКО k/k0 (метрика цели 0.05, общая с кривыми k/k0(PV)); по СКО lg оптимум другой
    # (gamma выше на 0.1): одной степенью (m/m0)^n обе метрики удовлетворяются лучше - это ограничение формы
    net_table = []
    for z in (3.0, 4.0, 5.0, 6.0, 8.0):
        res = minimize_scalar(lambda gm: lin_error(network(z, gm), pts), bounds=(0.2, 0.9), method='bounded',
                              options=dict(xatol=1e-3))
        res_lg = minimize_scalar(lambda gm: log_error(network(z, gm), pts)[0], bounds=(0.2, 0.9), method='bounded',
                                 options=dict(xatol=1e-3))
        net_table.append((z, float(res.x), float(res.fun), float(res_lg.x), float(res_lg.fun)))
        print(f'  сеть: z = {z:g}: по k/k0 горло {res.x:.3f}, СКО {res.fun:.3f}; по lg горло {res_lg.x:.3f}, '
              f'СКО lg {res_lg.fun:.3f}', flush=True)
    z_fit, gam_fit = min(net_table, key=lambda e: e[2])[:2]
    z_lg, gam_lg = min(net_table, key=lambda e: e[4])[0], min(net_table, key=lambda e: e[4])[3]
    curves = {
        'Козени-Карман': np.column_stack([m_grid, kozeny_carman(m_grid)]),
        f'(m/m0)^n, n = {n_fit:.1f} (подбор)': np.column_stack([m_grid, m_grid ** n_fit]),
        'пучок: сужение слоем кристаллов (C0 = 1)': narrowing(1.0),
        f'пучок: гель-отложение, C0 = {c0:.2f} ({c0_source})': narrowing(c0),
        f'пучок: гель-отложение, C0 = {fit.x:.2f} (подбор по He)': narrowing(fit.x),
        f'сеть пор и горл: z = {NET_PRIOR[0]:g}, горло {NET_PRIOR[1]:g} (априори)': network(*NET_PRIOR),
        f'сеть пор и горл: z = {z_fit:g}, горло {gam_fit:.2f} (подбор по k/k0)': network(z_fit, gam_fit),
        f'сеть пор и горл: z = {z_lg:g}, горло {gam_lg:.2f} (подбор по lg)': network(z_lg, gam_lg),
    }
    errors, missed, lin = {}, {}, {}
    for name, curve in curves.items():
        errors[name], missed[name] = log_error(curve, pts)
        lin[name] = lin_error(curve, pts)
        print(f'{name}: СКО lg(k/k0) {errors[name]:.3f}, СКО k/k0 {lin[name]:.3f}'
              + (f', не достает до {missed[name]} из {len(pts)} точек' if missed[name] else ''), flush=True)
    print(f'показатель степенного закона по He: n = {n_fit:.1f} (обзор, разд. 4.2: 8-19)', flush=True)
    out = dict(pairs={w: [list(p) for p in rows] for w, rows in data.items()}, n_fit=n_fit, c0=c0,
               c0_source=c0_source, c0_fit=float(fit.x), errors=errors, missed=missed, lin=lin, net_prior=list(NET_PRIOR),
               net_fit=[z_fit, gam_fit], net_fit_lg=[z_lg, gam_lg], net_table=net_table,
               curves={name: curve.tolist() for name, curve in curves.items()})
    save_results('he2020', out)
    plot(out)
    return out


def summary(out) -> list:
    """Строки сводной таблицы: (зависимость k(m), СКО lg(k/k0), СКО k/k0, точек вне досягаемости, всего)."""
    n_pts = sum(len(rows) for rows in out['pairs'].values())
    return [(name, e, out.get('lin', {}).get(name, float('nan')), out['missed'][name], n_pts)
            for name, e in out['errors'].items()]


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for (well, rows), marker in zip(out['pairs'].items(), ('^', 'D', 's', 'v')):
        p = np.array(rows)
        ax.semilogy(p[:, 0], p[:, 1], marker, color='k', mfc='white', ms=4.5, label=f'He et al., {well}')
    styles = ('-.', ':', '--', '-', (0, (5, 1, 1, 1)), (0, (3, 1)), (0, (1, 1)), (0, (6, 2)))
    for (name, curve), style in zip(out['curves'].items(), styles):
        c = np.array(curve)
        order = np.argsort(c[:, 0])
        ax.semilogy(c[order, 0], np.maximum(c[order, 1], 1e-3), ls=style,
                    label=f'{name}: {out["errors"][name]:.2f}')
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(5e-3, 1.3)
    ax.set_xlabel('m / m₀')
    ax.set_ylabel('k / k₀')
    ax.legend(fontsize=6.3, loc='lower right')
    fig.tight_layout()
    fig.savefig(FIGURES / 'he2020.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run(mode_from_argv())
