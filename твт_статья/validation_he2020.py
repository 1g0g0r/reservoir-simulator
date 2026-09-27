"""Сравнение с опытами He et al. (2020): как связаны потеря пористости и потеря проницаемости при осаждении парафина.

He Y., Pu W., Chen X., Liu R., Chen F., Liu B., et al. Cold damage from wax deposition in a shallow,
low-temperature, and high-wax reservoir in Changchunling Oilfield // Sci. Rep. 2020. V. 10. P. 14223.
doi:10.1038/s41598-020-71065-z (resources/литература/парафиновое/he2020_changchunling_cold_damage.pdf).

Керны четырех скважин месторождения Чанчуньлин охлаждались от 50 до 5 C с прокачкой парафинистой нефти;
при каждой температуре измерены пористость после отложения (гелиевый порозиметр, рис. 9) и эффективная
проницаемость по нефти (рис. 10). Кинетики (PV, время) в статье нет, зато есть пары (m/m0, k/k0) - то, что
в модели задает пучок капилляров: m ~ int r^2 fi, k ~ int r^4 fi. С ними сравниваются:
  - модель, чистое блокирование каналов r < r_b: прежняя - канал выбывает целиком, нынешняя - теряется
    только объем пробки, а канал остается тупиковой пористостью (механизм q_p2);
  - модель, чистое сужение: все каналы сужаются на одну толщину осадка (механизм u_r);
  - конечные состояния (m/m0, k/k0) ячеек из расчетов керна (Sutton & Roberts, Li, Sandyga), если они посчитаны:
    они лежат при m/m0 < 0.25 и сравниваются с концом опыта Sandyga, а не с точками He (m/m0 > 0.55);
  - Козени-Кармана, которую для той же задачи берут Sandyga et al. (2020, ур. 17).
Комментарии и выводы - VALIDATION.md.

    python твт_статья/validation_he2020.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core_flood as cf  # noqa: E402

OUT = cf.ROOT / 'outputs' / 'data' / 'validation_he2020.json'
FIG = cf.ROOT / 'outputs' / 'figures' / 'validation' / 'he2020.png'
DATA = cf.ROOT / 'outputs' / 'data'

# --- Данные He et al. (2020) ---------------------------------------------------------------------------
# Оцифровка рис. 9 и 10 по сетке с растра 300 dpi: пористость ~0.3%, проницаемость ~3 мД, T - по узлам
# маркеров. DSC WAT нефтей 25-26 C (табл. 2), температура застывания 14.6-18.2 C (разд. «Wax deposition tests»).
POROSITY = {  # {скважина: {T, C: пористость, %}}
    'C110': {50: 26.5, 40: 26.3, 35: 25.9, 26: 25.4, 15: 24.2, 10: 18.3, 7: 17.8},
    'C109': {40: 26.2, 30: 25.9, 25: 25.9, 20: 25.2, 15: 22.5, 10: 17.5, 2: 16.6},
    'C107-3-1': {50: 27.1, 40: 26.9, 35: 26.8, 30: 26.5, 16: 25.4, 10: 15.5, 2: 14.8},
    'C107-1-6': {50: 27.5, 40: 27.1, 35: 26.8, 26: 25.7, 15: 21.1, 8: 15.3, 2: 13.2},
}
PERMEABILITY = {  # {скважина: {T, C: проницаемость, мД}}
    'C110': {50: 229, 40: 229, 35: 206, 26: 187, 15: 131, 10: 29, 5: 25},
    'C109': {50: 101, 40: 99, 30: 94, 25: 90, 20: 80, 15: 45, 10: 16, 5: 13},
    'C107-3-1': {50: 132, 40: 125, 30: 111, 16: 80, 10: 5},
    'C107-1-6': {50: 191, 40: 170, 35: 157, 26: 117, 15: 33, 8: 6, 5: 4},
}
# Для сравнения - конечная точка опыта Sandyga et al. (2020): пористость 9.0 -> 2.1% (томография), градиент
# давления вырос в 44 раза при постоянном расходе, из них ~1.25 раза - вязкость (E = 25 кДж/моль, 40 -> 32.8 C).
SANDYGA_POINT = (2.1 / 9.0, 1.25 / (36.14 / 0.82))
SIGMA_R = 0.4  # ширина fi_0 модели; без пробок зависимости k(m) от r_m не зависят (подобие по r/r_m)
R_M = 12e-6    # медианный радиус fi_0 модели - нужен только объему пробки относительно канала


def pairs():
    """Пары (m/m0, k/k0, T) при общих температурах; опорная точка - самая горячая из общих."""
    out = {}
    for well in POROSITY:
        common = sorted(set(POROSITY[well]) & set(PERMEABILITY[well]), reverse=True)
        m0, k0 = POROSITY[well][common[0]], PERMEABILITY[well][common[0]]
        out[well] = [(POROSITY[well][t] / m0, PERMEABILITY[well][t] / k0, t) for t in common[1:]]
    return out


def bundle_relations():
    """k/k0 от m/m0 для пучка капилляров с fi_0 модели при крайних механизмах.

    Блокирование каналов r < r_b: прежняя модель выводила канал целиком (m теряет r^2 канала), нынешняя -
    только пробку, один кристалл d_p на канал (D^3/(6*L_k) в единицах r^2, `plug_cv`), а объем канала
    остается тупиковой пористостью. d_p, L_k - калибровка по Sutton & Roberts, r_m = 12 мкм.
    Сужение всех каналов на одну толщину осадка - в обеих моделях одинаково.
    """
    calibration = json.loads((DATA / 'core_flood.json').read_text(encoding='utf-8'))['best']['1']
    plug = calibration['d_p'] ** 3 / (6.0 * calibration['lk']) / R_M ** 2  # в единицах r_m^2
    r = np.linspace(1e-3, 5.0, 20000)  # в единицах r_m
    fi = np.exp(-0.5 * (np.log(r) / SIGMA_R) ** 2) / r
    m_tot, k_tot = np.sum(r ** 2 * fi), np.sum(r ** 4 * fi)
    old_blocking, blocking, narrowing = [], [], []
    for x in np.linspace(0.0, 3.0, 400):
        keep = r >= x
        k_rel = np.sum((r ** 4 * fi)[keep]) / k_tot
        old_blocking.append((np.sum((r ** 2 * fi)[keep]) / m_tot, k_rel))
        blocking.append((1.0 - np.sum((np.minimum(r ** 2, plug) * fi)[~keep]) / m_tot, k_rel))
        rr = np.maximum(r - x, 0.0)
        narrowing.append((np.sum(rr ** 2 * fi) / m_tot, np.sum(rr ** 4 * fi) / k_tot))
    return np.array(old_blocking), np.array(blocking), np.array(narrowing)


def kozeny_carman(m_rel, m0):
    m = m_rel * m0
    return m_rel ** 3 * ((1.0 - m0) / (1.0 - m)) ** 2


def simulated():
    """Пары (m/m0, k/k0) по ячейкам из уже посчитанных прогонов керна."""
    sources = {}
    path = DATA / 'core_flood.json'
    if path.exists():
        best = json.loads(path.read_text(encoding='utf-8'))['best']['1']
        sources['модель, Sutton & Roberts опыт 1'] = (best['m_profile'], best['k_profile'])
    path = DATA / 'validation_li2024.json'
    if path.exists():
        li = json.loads(path.read_text(encoding='utf-8'))
        run = li['runs'].get('перенос с Berea, поры Berea')
        if run:
            sources['модель, Li et al. 25 C'] = (run['m_profile'], run['k_profile'])
    path = DATA / 'validation_sandyga2020.json'
    if path.exists():
        run = json.loads(path.read_text(encoding='utf-8'))['runs'].get('WAT 33.8 C (керн)')
        if run:
            sources['модель, Sandyga et al. (WAT 33.8 C)'] = (run['m_profile'], run['k_profile'])
    return {name: np.array(list(zip(*prof))) for name, prof in sources.items()}


def log_error(curve, points):
    """СКО lg(k/k0) кривой k(m) от точек; кривая интерполируется по m/m0. Если кривая не доходит до
    самых малых m/m0 точек (блокирование с пробками: m почти не меняется), возвращает nan -
    подстановка крайнего значения выдала бы за согласие то, чего модель не описывает."""
    order = np.argsort(curve[:, 0])
    m_c, k_c = curve[order, 0], np.maximum(curve[order, 1], 1e-6)
    pts = np.array(points)
    if m_c[0] > pts[:, 0].min():
        return float('nan')
    model = np.interp(pts[:, 0], m_c, k_c)
    return float(np.sqrt(np.mean((np.log10(model) - np.log10(pts[:, 1])) ** 2)))


def at_m(curve, m):
    """k/k0 кривой при заданном m/m0 или nan, если кривая туда не доходит."""
    order = np.argsort(curve[:, 0])
    if not curve[order[0], 0] <= m <= curve[order[-1], 0]:
        return float('nan')
    return float(np.interp(m, curve[order, 0], curve[order, 1]))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    data = pairs()
    old_blocking, blocking, narrowing = bundle_relations()
    m_grid = np.linspace(0.2, 1.0, 81)
    kc = np.column_stack([m_grid, kozeny_carman(m_grid, 0.27)])
    sims = simulated()

    all_pts = [(m, k) for rows in data.values() for m, k, _ in rows]
    cold_pts = [(m, k) for rows in data.values() for m, k, t in rows if t <= 16]
    warm_pts = [(m, k, t) for rows in data.values() for m, k, t in rows if t > 20]

    print('| Скважина | T, C | m/m0 | k/k0 | Козени-Карман | блокирование (весь канал) | сужение |')
    for well, rows in data.items():
        for m, k, t in rows:
            print(f'| {well} | {t} | {m:.3f} | {k:.3f} | {kozeny_carman(m, 0.27):.3f} | {at_m(old_blocking, m):.3f} | '
                  f'{at_m(narrowing, m):.3f} |')

    curves = {'Козени-Карман': kc, 'прежняя модель: блокирование (весь канал)': old_blocking,
              'модель: блокирование (пробка)': blocking, 'модель: сужение': narrowing}
    errors = {}
    print('\n| Зависимость k(m) | СКО lg(k/k0), все точки | точки T <= 16 C | k/k0 при m/m0 = 0.7 | '
          'наименьшее m/m0 |')
    for name, curve in curves.items():
        errors[name] = dict(all=log_error(curve, all_pts), cold=log_error(curve, cold_pts))
        print(f'| {name} | {errors[name]["all"]:.2f} | {errors[name]["cold"]:.2f} | {at_m(curve, 0.7):.3f} | '
              f'{curve[:, 0].min():.3f} |')
    m_s, k_s = SANDYGA_POINT
    print(f'\nSandyga et al.: m/m0 = {m_s:.3f}, k/k0 ~ {k_s:.3f}; Козени-Карман {kozeny_carman(m_s, 0.09):.3f}, '
          f'блокирование (весь канал) {at_m(old_blocking, m_s):.3f}, сужение {at_m(narrowing, m_s):.3f}')
    # Прогоны дают только конечное состояние ячеек: облако точек, а не кривая - только диапазоны
    print('| Прогон (конечное состояние ячеек) | m/m0 | k/k0 |')
    for name, curve in sims.items():
        print(f'| {name} | {curve[:, 0].min():.3f}-{curve[:, 0].max():.3f} | {curve[:, 1].min():.3f}-{curve[:, 1].max():.3f} |')
    print('Выше 20 C (до DSC WAT 25-26 C и около него): ' +
          ', '.join(f'{t} C: m {m:.2f}, k {k:.2f}' for m, k, t in warm_pts))

    OUT.write_text(json.dumps(dict(
        pairs={w: [list(p) for p in rows] for w, rows in data.items()},
        relations={name: curve.tolist() for name, curve in curves.items()},
        simulated={name: curve.tolist() for name, curve in sims.items()},
        errors=errors, sandyga=SANDYGA_POINT,
    ), ensure_ascii=False), encoding='utf-8')
    print(f'Записано: {OUT}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for (well, rows), marker in zip(data.items(), ('^', 'D', 's', 'x')):
        pts = np.array(rows)
        ax.semilogy(pts[:, 0], pts[:, 1], marker, color='k', mfc='none', label=f'He et al., {well}')
    ax.semilogy(*SANDYGA_POINT, 'k*', ms=10, label='Sandyga et al., конец опыта')
    for (name, curve), style in zip(curves.items(), ('k-.', 'k:', 'k-', 'k--')):
        order = np.argsort(curve[:, 0])
        ax.semilogy(curve[order, 0], np.maximum(curve[order, 1], 1e-3), style, label=name)
    for (name, curve), marker in zip(sims.items(), ('.', '+', '1')):
        ax.semilogy(curve[:, 0], np.maximum(curve[:, 1], 1e-3), marker, color='0.45', ms=4,
                    label=name + ' (ячейки в конце)')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(1e-3, 1.2)
    ax.set_xlabel('m/m₀')
    ax.set_ylabel('k/k₀')
    ax.legend(fontsize=6.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f'Рисунок: {FIG}')


if __name__ == '__main__':
    main()
