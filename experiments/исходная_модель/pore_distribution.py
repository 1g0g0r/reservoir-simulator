"""Начальная функция пор по размерам и коэффициент диффузии частиц: из-за них ли расчет не сходится с опытом.

Керн Sutton & Roberts (опыты 1 и 2, `core_flood.py`). Для каждой формы fi_0 (логнормаль Косуги по числу
каналов: медианный радиус r_m, ширина sigma_r) подбираются два параметра кинетики:
  - размер частиц d_p - через долю проводимости, которую блокирование снимает сразу (каналы r < d_p/(2*gamma)):
    20/30/40% (в базе 31%). В абсолютных микрометрах сетку не задать: у широких распределений проводимость несут
    крупные каналы, и оптимум d_p уходит за 20 мкм;
  - множитель к коэффициенту диффузии по Стоксу-Эйнштейну `diff_mult` (0.1-10). В сужение D_p входит как D_p^2/L_k,
    поэтому L_k держится 0.3 мм, а варьируется D_p.
Оба опыта считаются для всех сочетаний: так видно и прогноз опыта 2 по калибровке на опыте 1, и есть ли вообще
сочетание, описывающее оба опыта.

Сетка радиусов каждого варианта доходит до 99.5% проводимости (int r^4*fi_0) при шаге ~1.3 мкм, как в базе:
при r_max = 40 мкм распределения шире базового обрезаются (sigma_r = 0.6 теряет 65% проводимости), и прежние
прогоны `core_flood.py --diag` с sigma_r = 0.6, 0.8 и r_m = 16 мкм считались по обрезанным распределениям.

    python experiments/исходная_модель/pore_distribution.py            # статика + все прогоны (~150 шт., ~1.2 ч, правят constants.py)
    python experiments/исходная_модель/pore_distribution.py --static   # только моменты распределений
    python experiments/исходная_модель/pore_distribution.py --table    # таблица по сохраненному json

Результат - outputs/data/pore_distribution.json, комментарии - docs/WAX_PRECIPITATION_FINDINGS.md.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core_flood as cf  # noqa: E402

OUT = cf.ROOT / 'outputs' / 'data' / 'pore_distribution.json'
GAMMA = 0.4       # отношение радиуса горла к радиусу канала (constants.py)
LK = 3e-4         # длина канала, [м] - как в калибровке; скорость сужения варьируется через diff_mult
DR = 1.33e-6      # шаг сетки радиусов базы (40 мкм / 30)

# Формы fi_0: (r_m, sigma_r, r_max). r_max = None - до 99.5% проводимости.
VARIANTS = {
    'база: r_m 12 мкм, sigma 0.4, r_max 40 мкм': (12e-6, 0.4, 40e-6),
    'r_m 12 мкм, sigma 0.4 без обрезки': (12e-6, 0.4, None),
    'r_m 12 мкм, sigma 0.3': (12e-6, 0.3, None),
    'r_m 12 мкм, sigma 0.6': (12e-6, 0.6, None),
    'r_m 8 мкм, sigma 0.6': (8e-6, 0.6, None),
}
BLOCKED = (0.2, 0.3, 0.4)            # доля проводимости, снимаемая блокированием сразу
DIFF = (0.1, 0.3, 1.0, 3.0, 10.0)    # множитель к D_p по Стоксу-Эйнштейну


def kosugi(r, r_m, sigma):
    return np.exp(-0.5 * (np.log(r / r_m) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma * r)


def r_max_for(r_m, sigma, share=0.995):
    """Радиус, до которого набирается `share` интеграла r^4*fi_0 (медиана проводимости r_m*exp(4*sigma^2))."""
    from scipy.stats import norm
    return r_m * np.exp(4.0 * sigma ** 2 + norm.ppf(share) * sigma)


def grid(r_m, sigma, r_max):
    r_max = r_max or r_max_for(r_m, sigma)
    nr = int(round(r_max / DR)) + 1
    return r_max, nr


def moments(r_m, sigma, r_max):
    """Медианы по числу, объему и проводимости, доля проводимости за r_max, извилистость для опыта 1."""
    r = np.linspace(1e-9, max(r_max, r_max_for(r_m, sigma, 0.9999)), 200000)
    f = kosugi(r, r_m, sigma)
    inside = r <= r_max

    def median(w):
        return r[np.searchsorted(np.cumsum(w) / w.sum(), 0.5)]

    lost_k = (r ** 4 * f)[~inside].sum() / (r ** 4 * f).sum()
    fi = f * inside
    eta = np.sqrt(cf.POROSITY * np.sum(r ** 4 * fi) / (8.0 * cf.EXPERIMENTS[1]['k0'] * 9.869233e-13 * np.sum(r ** 2 * fi)))
    return dict(median_n=median(f), median_v=median(r ** 2 * f), median_k=median(r ** 4 * f), lost_k=lost_k, eta=eta)


def d_p_for(r_m, sigma, r_max, share):
    """Размер частиц, при котором блокирование сразу снимает долю `share` проводимости (каналы r < d_p/(2*gamma))."""
    r = np.linspace(1e-9, r_max, 200000)
    w = np.cumsum(r ** 4 * kosugi(r, r_m, sigma))
    w /= w[-1]
    r_pass = float(np.interp(share, w, r))
    return 2.0 * GAMMA * r_pass


def static():
    rows = {}
    print('| Вариант | r_max, мкм | Nr | медиана по числу / объему / проводимости, мкм | проводимость за r_max | '
          'извилистость | d_p при 20/30/40% | d_p = 15 мкм снимает |')
    for name, (r_m, sigma, r_max) in VARIANTS.items():
        r_max, nr = grid(r_m, sigma, r_max)
        mo = moments(r_m, sigma, r_max)
        d_ps = [d_p_for(r_m, sigma, r_max, s) for s in BLOCKED]
        r = np.linspace(1e-9, r_max, 200000)
        w = r ** 4 * kosugi(r, r_m, sigma)
        base_share = w[r < 15e-6 / (2 * GAMMA)].sum() / w.sum()
        print(f'| {name} | {r_max * 1e6:.0f} | {nr} | {mo["median_n"] * 1e6:.1f} / {mo["median_v"] * 1e6:.1f} / '
              f'{mo["median_k"] * 1e6:.1f} | {100 * mo["lost_k"]:.1f}% | {mo["eta"]:.2f} | '
              f'{" / ".join(f"{d * 1e6:.1f}" for d in d_ps)} | {100 * base_share:.0f}% |')
        rows[name] = dict(r_m=r_m, sigma=sigma, r_max=r_max, Nr=nr, d_p=d_ps, **mo)
    return rows


def runs(rows):
    data = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
    for name, row in rows.items():
        if name in data and len(data[name].get('runs', [])) == len(BLOCKED) * len(DIFF):
            continue  # уже посчитан - прерванную серию можно продолжить
        extra = {'r_m': repr(row['r_m']), 'sigma_r': repr(row['sigma']), 'r_max': repr(row['r_max']),
                 'Nr': str(row['Nr'])}
        result = []
        for share, d_p in zip(BLOCKED, row['d_p']):
            for diff in DIFF:
                res = {}
                for number in (1, 2):
                    r = cf.run_case(number, d_p, LK, extra=dict(extra, diff_mult=repr(diff)))
                    res[number] = r
                item = dict(f_block=share, d_p=d_p, diff=diff,
                            rms1=res[1]['rms'], rms2=res[2]['rms'],
                            plugged1=res[1]['plugged'], plugged2=res[2]['plugged'],
                            k1=[float(np.interp(x, res[1]['pv'], res[1]['k'])) for x in (0.25, 1.0, 3.0, 5.0)],
                            k2=[float(np.interp(x, res[2]['pv'], res[2]['k'])) for x in (0.25, 1.0, 3.0, 5.0)],
                            loss1=float(np.mean(res[1]['pore_loss']) / cf.POROSITY))
                result.append(item)
                print(f'{name} | {share:.0%} (d_p {d_p * 1e6:.1f} мкм) | D x{diff:g}: опыт 1 {item["rms1"]:.3f}, '
                      f'опыт 2 {item["rms2"]:.3f}', flush=True)
        data[name] = dict(row, runs=result)
        OUT.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')  # после каждого варианта
    return data


def table(data):
    print('\n| Вариант | калибровка по опыту 1: доля / D_p / СКО | прогноз опыта 2 | лучшее на оба: доля / D_p | '
          'СКО опыт 1 / опыт 2 |')
    for name, row in data.items():
        best1 = min(row['runs'], key=lambda x: x['rms1'])
        joint = min(row['runs'], key=lambda x: max(x['rms1'], x['rms2']))
        plug = f', закупорка {best1["plugged2"]:.1f} PV' if best1['plugged2'] else ''
        print(f'| {name} | {best1["f_block"]:.0%} ({best1["d_p"] * 1e6:.1f} мкм) / x{best1["diff"]:g} / {best1["rms1"]:.3f} | '
              f'{best1["rms2"]:.3f}{plug} | {joint["f_block"]:.0%} ({joint["d_p"] * 1e6:.1f} мкм) / x{joint["diff"]:g} | '
              f'{joint["rms1"]:.3f} / {joint["rms2"]:.3f} |')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if sys.argv[1:2] == ['--table']:
        table(json.loads(OUT.read_text(encoding='utf-8')))
        return
    rows = static()
    if sys.argv[1:2] == ['--static']:
        return
    table(runs(rows))


if __name__ == '__main__':
    main()
