"""Рисунки и числа статьи для ТВТ по готовым результатам: сравнения с опытами (`experiments/results/*.json`) и
полевые варианты (`results/thermal.json`, пишет `run_thermal.py`).

    python твт_статья_АСПО/make_article.py

Результат:
  figures/aspo_f*.tif - 600 dpi, 256 оттенков серого (правило 8.10), и .png для просмотра;
  results/numbers.json - все числа и таблицы текста: `article.md` ссылается на них как {{КЛЮЧ}}, подстановку
  делает `python docs/build_docx.py твт_статья_АСПО/article.md`;
  metrics.md - те же числа списком, для чтения.
Числа в тексте руками не набираются: после пересчета достаточно перезапустить этот скрипт и сборку docx.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT / 'experiments'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP))
FIGURES = HERE / 'figures'
RESULTS = HERE / 'results'

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.linewidth': 0.8, 'lines.linewidth': 1.5,
                     'legend.frameon': False, 'xtick.direction': 'in', 'ytick.direction': 'in'})
MARKERS = ('o', 's', '^', 'D', 'v', 'x')
STYLES = ('-', '--', ':', '-.', (0, (6, 1.5, 1, 1.5, 1, 1.5)), (0, (1, 1)))
PANEL = dict(fontsize=10, fontweight='normal')


def _load(name):
    path = EXP / 'results' / f'{name}.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def _save(fig, name: str) -> None:
    """TIF 600 dpi в 8-битных оттенках серого (matplotlib пишет RGB, PIL переводит в L) и PNG для просмотра."""
    from PIL import Image
    FIGURES.mkdir(exist_ok=True)
    tif = FIGURES / f'{name}.tif'
    fig.savefig(tif, dpi=600, format='tiff', pil_kwargs={'compression': 'tiff_lzw'}, bbox_inches='tight')
    with Image.open(tif) as img:
        gray = img.convert('L')
    gray.save(tif, compression='tiff_lzw', dpi=(600, 600))  # convert() теряет разрешение из метаданных
    fig.savefig(FIGURES / f'{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def _panel(ax, letter):
    ax.text(0.03, 0.95, f'({letter})', transform=ax.transAxes, va='top', **PANEL)


def _f(x, d=3):
    return '—' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.{d}f}'


def _sci(x, digits=1):
    """Число вида 3·10⁴ (показатель - верхними индексами Юникода: pandoc не переводит их в формулы)."""
    if x == 0 or not math.isfinite(x):
        return str(x)
    e = int(math.floor(math.log10(abs(x))))
    mant = x / 10 ** e
    if abs(round(mant, digits)) >= 10.0:  # 9.6 при нуле знаков - это 1·10^(e+1), а не 10·10^e
        mant, e = mant / 10.0, e + 1
    sup = str(e).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
    return f'{mant:.{digits}f}·10{sup}'


def _delta(x):
    """Разность КИН для таблицы: знак - типографский минус, меньше половины последнего знака - ноль."""
    return '0.000' if abs(x) < 5e-4 else f'{x:+.3f}'.replace('-', '−')


def _nz(x, d):
    """Число с d знаками без «-0.0»."""
    return f'{(round(x, d) or 0.0):.{d}f}'


def _table(head, rows):
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    return '\n'.join(lines + ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows])


# --- Опыты -------------------------------------------------------------------------------------------------

def fig_sutton_roberts(sr, num):
    import sutton_roberts as m
    g = lambda d, n: d[n] if n in d else d[str(n)]
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.7), sharey=True)
    for ax, n, letter in zip(axes, (1, 2), 'аб'):
        pv, k = np.array(m.EXPS[n]['k_pv']).T
        ax.plot(pv, k, 'o', mfc='white', mec='k', ms=4)
        own = g(sr['lsq']['rate_entrainment'][f'own{n}']['curves'], n)
        ax.plot(own['pv'], own['k'], '-', color='k', label='1')
        x, y = np.array(m.EXPS[n]['models']['Wang & Civan (2005)']).T
        ax.plot(x, y, '--', color='k', lw=1.2, label='2')
        leg = g(sr['legacy'], n)
        ax.plot(leg['pv'], leg['k'], ':', color='k', lw=1.3, label='3')
        ax.set_xlim(0, 5.1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('V')
        _panel(ax, letter)
        num[f'SR_OWN{n}'] = _f(g(sr['lsq']['rate_entrainment'][f'own{n}']['rms'], n))
        num[f'SR_WC{n}'] = _f(g(sr['others']['Wang & Civan (2005)'], n))
        num[f'SR_LEG{n}'] = _f(m.rms(leg, n))
        num[f'SR_NOISE{n}'] = _f(g(sr['noise'], n))
        num[f'SR_VISC{n}'] = _f(g(sr['visc']['rms'], n))
        if n == 1:  # начало кривых: ступенька блокирования против плавной закупорки Wang и Civan
            num['SR_EARLY_OWN'] = _f(float(np.interp(0.02, own['pv'], own['k'])), 2)
            num['SR_WC_01'], num['SR_WC_025'] = (_f(float(np.interp(v, x, y)), 2) for v in (0.1, 0.25))
            num['SR_EXP_PV1'], num['SR_EXP_K1'] = _f(pv[1], 2), _f(k[1], 2)
    axes[0].set_ylabel('k/k₀')
    axes[1].legend(loc='lower left', fontsize=8, handlelength=2.6)
    fig.tight_layout()
    _save(fig, 'aspo_f1')


def fig_li(li, num):
    import li2024 as m
    fig, ax = plt.subplots(figsize=(6.7, 2.7))
    best = m.stage_curves(li['cold']['result'])
    start = 0.0
    for t, pv_stage in m.STAGES:
        pts = np.array(m.DATA['k_pv'][str(int(t))])
        ax.plot(start + pts[:, 0], pts[:, 1], 'o', mfc='white', mec='k', ms=3.5)
        if t in best:
            x, y = best[t]
            ax.plot(start + x, y, '-', color='k')
        ax.axvline(start, color='0.6', lw=0.6)
        ax.text(start + 0.5 * pv_stage, 1.08, f'{t:.0f}°C', ha='center', fontsize=8)
        start += pv_stage
    ax.set_xlim(0, start)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel('V')
    ax.set_ylabel('k/k₀')
    fig.tight_layout()
    _save(fig, 'aspo_f2')
    rms = li['cold']['rms']
    for t in ('90', '65', '45', '25'):
        num[f'LI_{t}'] = _f(rms.get(f'{t}.0', rms.get(t)))
        num[f'LI_NOISE{t}'] = _f(li['noise'][t])
    leg = m.stage_rms(li['legacy'])
    num['LI_LEG90'] = _f(leg.get(90.0))
    kin = li['forms'][li['best_form']]['kin']
    num['LI_DH'] = f"{-kin['ADS_DH'] / 1e3:.0f}"
    num['LI_FILM'] = _f(math.sqrt(sum(li['forms']['film']['rms'][t] ** 2 for t in ('90.0', '65.0', '45.0')) / 3))
    c = li['cold']['kin']
    num['LI_DCR'] = f"{c['D_CRYST'] * 1e6:.1f}"
    num['LI_DIFF'] = f"{c['DIFF_MULT']:.2f}"
    num['LI_KCR'] = f"{c['K_CRYST']:.3f}"
    num['LI_TCR'] = f"{1.0 / c['K_CRYST']:.0f}"
    num['LI_HOLD'] = f"{m.STAGE_HOLD / 3600:.0f}"
    cum = li['cumulative']
    num['LI_CUM25'] = _f(cum['25.0'])
    # Сеть пор и горл с кинетикой, подобранной для пучка, и с кристаллами другого диаметра (`li2024.NET_D_CRYST`)
    net = li.get('network', [])
    if net:
        rms = lambda e, t: e['rms'].get(f'{t}.0', e['rms'].get(t))
        fit = next(e for e in net if e['fitted'])
        num['LI_NET_FIT45'] = _f(rms(fit, '45'))
        sweep = [e for e in net if not e['fitted']]
        ok = [e for e in sweep if rms(e, '45') is not None and rms(e, '45') <= 0.05]
        span = lambda vals, fmt: fmt.format(min(vals)) + '–' + fmt.format(max(vals))
        num['LI_NET_D'] = span([e['d_cryst'] * 1e6 for e in ok], '{:g}')
        num['LI_NET45'] = span([rms(e, '45') for e in ok], '{:.3f}')
        num['LI_NET_DALL'] = span([e['d_cryst'] * 1e6 for e in sweep], '{:g}')
        num['LI_NET25'] = span([rms(e, '25') for e in sweep if rms(e, '25') is not None], '{:.2f}')


def fig_sandyga(sd, num):
    import sandyga2020 as m
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.7))
    ax = axes[0]
    t_exp, g_exp = np.array(m.DATA['gradient']['points']).T
    ax.plot(t_exp, g_exp / g_exp[0], 'o', mfc='white', mec='k', ms=4)
    for e, style, label in ((sd['kinetics_best'], '-', '1'), (sd['legacy'], '--', '2')):
        t, g = m.curve(e['result'])
        ax.plot(t, g, ls=style, color='k', label=label)
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('T, °C')
    ax.set_ylabel('∇p/∇p₀')
    ax.legend(loc='upper left', fontsize=8, bbox_to_anchor=(0.0, 0.88), handlelength=2.6)
    _panel(ax, 'а')
    ax = axes[1]
    rates = sorted(sd['cooling'].values(), key=lambda e: e['rate'])
    for e, style in zip(rates, STYLES):
        t, g = (np.array(v) for v in e['curve'])
        ax.plot(t, g, ls=style, color='k', lw=1.3, label=f"{e['rate']:g}")
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('T, °C')
    ax.legend(title='°C/ч', fontsize=7.5, title_fontsize=7.5, loc='upper left', bbox_to_anchor=(0.0, 0.88),
              handlelength=2.6)
    _panel(ax, 'б')
    fig.tight_layout()
    _save(fig, 'aspo_f3')
    b = sd['best']
    num['SD_RMS'] = _f(b['rms_k'])
    num['SD_RMSLG'] = _f(b['rms_log'])
    num['SD_NOISE'] = _f(sd['noise_log'])
    num['SD_LEG'] = _f(sd['legacy']['rms_k'])
    num['SD_WAT'] = f"{b['wat']:.1f}"
    num['SD_WAT_AUTH'] = f"{sd['wat_authors']['wat']:.1f}"
    num['SD_MCOND'] = _f(b['m_conductive'], 2)
    num['SD_MEXP'] = _f(sd['porosity_exp'], 2)
    num['SD_FACTOR'] = _f(b['max_factor'], 2)
    num['SD_KCR'] = f"{b['k_cryst']:.1e}"
    num['SD_TCR'] = f"{1.0 / b['k_cryst'] / 60:.0f}"
    num['SD_KWALL'] = f"{b['k_wall']:.1e}"
    by = {f"{e['rate']:g}": e for e in rates}
    for key, e in by.items():
        tag = key.replace('.', '_')
        num[f'SD_T10_{tag}'] = _f(e['t10'], 1)
        num[f'SD_FIN_{tag}'] = f"{e['final']:.0f}"
        num[f'SD_DA_{tag}'] = f"{e['damkohler']:.1f}"
    reach = [e for e in rates if e['t10'] is not None]  # при быстром охлаждении рост в 10 раз не достигается
    lo, hi = reach[0], reach[-1]
    num['SD_DT10'] = _f(lo['t10'] - hi['t10'], 1)
    num['SD_DT10_LO'] = f"{lo['rate']:g}"
    num['SD_DT10_HI'] = f"{hi['rate']:g}"
    # WAT раствора от скорости охлаждения (Struchkov, Rogachev, 2017) - проверка k_cr без керна
    wr = sd['wat_rate']
    num['SD_RATE_SPAN'] = _f(wr['exp_span'], 1)
    num['SD_RATE_THR'], num['SD_RATE_RMS'] = _f(100 * wr['sandyga']['thr'], 2), _f(wr['sandyga']['rms'], 1)
    num['SD_RATE_LI_THR'], num['SD_RATE_LI_RMS'] = _f(100 * wr['li2024']['thr'], 0), _f(wr['li2024']['rms'], 1)
    num['SD_BULK20_RHEO'], num['SD_BULK20_CORE'] = _f(wr['bulk20_rheometer'], 1), _f(wr['bulk20_core'], 1)
    num['SD_BULK20_SHIFT'], num['SD_WAT_BULK'] = _f(wr['bulk20_shift'], 1), _f(m.SOL['WAT_bulk'], 1)


def fig_he(pm, he, num):
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for (well, rows), marker in zip(pm['pairs'].items(), MARKERS):
        p = np.array(rows)
        ax.semilogy(p[:, 0], p[:, 1], marker, color='k', mfc='white', ms=4)
    rows = {r['name']: r for r in pm['rows']}
    pick = [('1.', '-', '1'), ('4. сеть, эффективная среда, z = 6, горло 0.42', '--', '2'), ('3.', ':', '3')]
    for prefix, style, label in pick:
        name = next(n for n in pm['curves'] if n.startswith(prefix))
        c = np.array(pm['curves'][name])
        o = np.argsort(c[:, 0])
        ax.semilogy(c[o, 0], np.maximum(c[o, 1], 1e-3), ls=style, color='k', lw=1.3, label=label)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(5e-3, 1.3)
    ax.set_xlabel('m/m₀')
    ax.set_ylabel('k/k₀')
    ax.legend(loc='lower right', fontsize=8, handlelength=2.6)
    fig.tight_layout()
    _save(fig, 'aspo_f4')
    get = lambda prefix: next(r for n, r in rows.items() if n.startswith(prefix))
    num['HE_BUNDLE'] = _f(get('1.')['lin'])
    num['HE_NET'] = _f(get('4. сеть, эффективная среда, z = 6, горло 0.42')['lin'])
    num['HE_NET0'] = _f(get('4. сеть, эффективная среда, z = 6, горло 0.4 ')['lin']) if any(
        n.startswith('4. сеть, эффективная среда, z = 6, горло 0.4 ') for n in rows) else _f(
        rows['4. сеть, эффективная среда, z = 6, горло 0.4']['lin'])
    num['HE_CONSTR'] = _f(get('3.')['lin'])
    num['HE_POWER'] = _f(get('6.')['lin'])
    num['HE_POWER_N'] = get('6.')['name'].split('=')[-1].strip()
    num['HE_LATTICE'] = _f(pm['ema_vs_lattice'])
    # пары после осаждения асфальтенов (Lin 2021, Struchkov 2019): кривые моделей в их m/m0, без подбора
    curve = lambda prefix: np.array(pm['curves'][next(n for n in pm['curves'] if n.startswith(prefix))])
    at = lambda prefix, x: float(np.interp(x, *curve(prefix)[np.argsort(curve(prefix)[:, 0])].T))
    for p, tag in zip(_jload(EXP / 'data' / 'asphaltene_pairs.json')['pairs'], ('LIN', 'ST')):
        num[f'ASPH_{tag}_M'], num[f'ASPH_{tag}_K'] = _f(p['m_ratio'], 2), _f(p['k_ratio'], 3 if tag == 'LIN' else 2)
        num[f'ASPH_{tag}_BUNDLE'] = _f(at('1.', p['m_ratio']), 2)
        num[f'ASPH_{tag}_NET'] = _f(at('4. сеть, эффективная среда, z = 6, горло 0.42', p['m_ratio']), 3)
        num[f'ASPH_{tag}_LAT'] = _f(at('5.', p['m_ratio']), 3)
    num['HE_N'] = str(sum(len(v) for v in pm['pairs'].values()))


def table_experiments(num):
    rows = [
        ('Sutton, Roberts, опыт 1', num['SR_NOISE1'], num['SR_LEG1'], num['SR_OWN1'], num['SR_WC1']),
        ('Sutton, Roberts, опыт 2', num['SR_NOISE2'], num['SR_LEG2'], num['SR_OWN2'], num['SR_WC2']),
    ]
    for t in ('90', '65', '45', '25'):
        rows.append((f'Li и др., {t}°C', num[f'LI_NOISE{t}'], num['LI_LEG90'] if t == '90' else 'закупорка',
                     num[f'LI_{t}'], '—'))
    rows.append(('Sandyga и др., охлаждение', '—', num['SD_LEG'], num['SD_RMS'], '—'))
    rows.append(('He и др., k(m)', '—', num['HE_BUNDLE'], num['HE_NET'], num['HE_POWER']))
    num['T1'] = _table(['Опыт', 'Шумовой порог', 'Упрощенная модель', 'Настоящая модель', 'Другие модели'], rows)
    own = [float(r[3]) for r in rows]
    num['RMS_MIN'], num['RMS_MAX'] = f'{min(own):.3f}', f'{max(own):.3f}'
    legacy = [float(r[2]) for r in rows if r[2] not in ('закупорка', '—')]
    num['LEG_MIN'], num['LEG_MAX'] = f'{min(legacy):.2f}', f'{max(legacy):.2f}'


# --- Поле --------------------------------------------------------------------------------------------------

# Серая шкала без черного края: на темной заливке остаются читаемыми изолинии и их подписи
GREYS = matplotlib.colors.ListedColormap(plt.cm.Greys_r(np.linspace(0.3, 1.0, 256)))
MAP_LABELS = {'T': 'T, °C', 'S': 'S', 'k': 'k/k₀'}


def _map(ax, z, vmin, vmax, fmt, zoom=1.0):
    """Карта поля: заливка в сером (темнее - меньше), изолинии с подписями; возвращает заливку для цветовой шкалы.
    Треугольник - нагнетательная скважина, круг - добывающая."""
    from matplotlib.ticker import MaxNLocator
    from paraphin.constants import X_max, Y_max
    z = np.array(z).T
    ny, nx = z.shape
    # значения в центрах ячеек, продленные на границы, чтобы заливка доходила до краев
    x = np.concatenate(([0.0], (np.arange(nx) + 0.5) * X_max / nx, [X_max]))
    y = np.concatenate(([0.0], (np.arange(ny) + 0.5) * Y_max / ny, [Y_max]))
    z = np.pad(z, 1, mode='edge')
    fill = ax.contourf(x, y, np.clip(z, vmin, vmax), levels=np.linspace(vmin, vmax, 41), cmap=GREYS)
    levels = [c for c in MaxNLocator(6).tick_values(vmin, vmax) if vmin < c < vmax and z.min() < c < z.max()]
    if levels:
        cs = ax.contour(x, y, z, levels=levels, colors='k', linewidths=0.7)
        ax.clabel(cs, fmt=fmt, fontsize=7)
    ax.plot([0], [0], '^', color='k', ms=7, clip_on=False)
    ax.plot([X_max], [Y_max], 'o', color='k', ms=6, clip_on=False)
    ax.set_xlim(0, zoom * X_max)
    ax.set_ylim(0, zoom * Y_max)
    ax.set_aspect('equal')
    return fill


def _bar(fig, fill, ax, key):
    """Цветовая шкала карты с подписью величины."""
    bar = fig.colorbar(fill, ax=ax, shrink=0.85, pad=0.03, ticks=matplotlib.ticker.MaxNLocator(5))
    bar.set_label(MAP_LABELS[key], fontsize=8)
    bar.ax.tick_params(labelsize=7)


def _range(th, key, maps):
    """Пределы шкалы: температура - от закачки до пластовой, насыщенность и k/k0 - по всем картам."""
    if key == 'T':
        return 20.0, th['init_T']
    vals = np.concatenate([np.ravel(m) for m in maps])
    return float(vals.min()), float(max(vals.max(), vals.min() + 1e-3))


def _zoom(th, maps):
    """Доля стороны элемента, в которую с запасом укладываются остывшая и поврежденная зоны на конец расчета,
    с шагом 0.25: зона у нагнетательной скважины крупнее и видна целиком."""
    t, k = np.array(maps['T'][-1]), np.array(maps['k'][-1])
    hit = np.nonzero((t < th['init_T'] - 1.0) | (k < 0.98))
    extent = (max(hit[0].max(), hit[1].max()) + 1) / t.shape[0] if hit[0].size else 0.25
    return min(1.0, 0.25 * math.ceil(1.15 * extent / 0.25))


def fig_field_maps(th, num):
    """Журнальный рисунок: температура и проницаемость базового варианта на конец расчета, с цветовыми шкалами."""
    maps = th['maps']
    zoom = _zoom(th, maps)
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.9))
    for ax, key, fmt, letter in ((axes[0], 'T', '%.0f', 'а'), (axes[1], 'k', '%.2f', 'б')):
        fill = _map(ax, maps[key][-1], *_range(th, key, [maps[key][-1]]), fmt, zoom)
        _bar(fig, fill, ax, key)
        ax.set_xlabel('x, м')
        _panel(ax, letter)
    axes[0].set_ylabel('y, м')
    fig.tight_layout()
    _save(fig, 'aspo_f5')
    num['F_MAP_SIDE'] = f'{zoom * 200:.0f}'


def fig_field_evolution(th, num):
    """Полная версия: температура, водонасыщенность и k/k0 базового варианта в три момента по всему элементу;
    шкала одна на строку, чтобы моменты сравнивались по одной заливке."""
    maps = th['maps']
    fig, axes = plt.subplots(3, 3, figsize=(6.7, 6.6), sharex=True, sharey=True)
    for row, (key, fmt) in enumerate((('T', '%.0f'), ('S', '%.2f'), ('k', '%.2f'))):
        lo, hi = _range(th, key, maps[key])
        for col, z in enumerate(maps[key]):
            fill = _map(axes[row, col], z, lo, hi, fmt)
            _panel(axes[row, col], 'абвгдежзи'[3 * row + col])
            if row == 0:
                axes[row, col].set_title(f"t = {maps['years'][col]:g} г.", fontsize=9)
        _bar(fig, fill, list(axes[row]), key)
    for ax in axes[-1]:
        ax.set_xlabel('x, м')
    for ax in axes[:, 0]:
        ax.set_ylabel('y, м')
    _save(fig, 'aspo_f8')
    num['F_MAP_YEARS'] = ', '.join(f'{y:g}' for y in maps['years'])


# Журнальный рисунок - ключевые факторы, полная версия - температура закачки
FIELD_CURVES = (('tiso', '1'), ('base', '2'), ('hl0', '3'), ('nowax', '4'), ('ret', '5'))
TEMP_CURVES = (('tiso', '1'), ('t40', '2'), ('base', '3'), ('t5', '4'))


def _time_curves(th, curves, name):
    """Приемистость нагнетательной скважины (а) и КИН (б) во времени; маркеры - для различения кривых в сером."""
    v = th['variants']
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.8))
    for (key, label), style, marker in zip(curves, STYLES, MARKERS):
        if key not in v:
            continue
        s = v[key]['series']
        t = np.array(s['years'])
        every = max(1, len(t) // 7)
        for ax, y in ((axes[0], np.array(s['q_inj']) * 4.0), (axes[1], np.array(s['rf']))):
            ax.plot(t, y, ls=style, color='k', lw=1.2, marker=marker, ms=3.5, mfc='white', markevery=every,
                    label=label)
    axes[0].set_ylabel('Q, м³/сут')
    axes[1].set_ylabel('КИН')
    for ax, letter in zip(axes, 'аб'):
        ax.set_xlabel('t, годы')
        ax.set_ylim(bottom=0)
        _panel(ax, letter)
    axes[1].legend(loc='lower right', fontsize=8, handlelength=3.2)
    fig.tight_layout()
    _save(fig, name)


def fig_field_time(th, num):
    _time_curves(th, FIELD_CURVES, 'aspo_f6')
    _time_curves(th, TEMP_CURVES, 'aspo_f7')


def _opt(x, fmt, scale=1.0):
    return '—' if x is None else fmt.format(scale * x)


FIELD_ORDER = ('base', 'tiso', 't40', 't5', 'hl0', 'hl1', 'nowax', 'hl0_nowax', 'nogel', 'ret',
               'nolatent', 'equil', 'nopress', 'ltne')


def factorial(v, f) -> dict:
    """Полный факторный план теплообмен x парафин (табл. 3 прежней статьи): эффект одного фактора при двух уровнях
    другого и взаимодействие. f(имя варианта) - показатель."""
    hl_wax, hl_nowax = f('base') - f('hl0'), f('nowax') - f('hl0_nowax')
    return {'hl_wax': hl_wax, 'hl_nowax': hl_nowax, 'wax_hl': f('base') - f('nowax'),
            'wax_nohl': f('hl0') - f('hl0_nowax'), 'inter': hl_wax - hl_nowax}


def field_numbers(th, num):
    from paraphin.constants import (h, K_f, K_w, K_o, c_w, ro_w, c_o, ro_o, c_f, ro_f, init_m, init_S, latent_heat,
                                    P_ref_wax)
    import paraphin.oil_composition as oc
    t0 = th['init_T']
    v = th['variants']
    base = v['base']
    rows = []
    for name in FIELD_ORDER:
        if name not in v:
            continue
        e = v[name]
        d_rf = '—' if name == 'base' else _delta(e['rf'] - base['rf'])
        rows.append((e['label'], f"{e['rf']:.3f}", d_rf, f"{e['injected'] / 1e3:.1f}", _nz(e['skin_inj'], 1),
                     f"{e['wax_dep_t']:.0f}", f"{100 * e['below_wat_area']:.0f}", f"{100 * e['gel_area']:.0f}",
                     f"{e['T_mean']:.1f}"))
    num['T2'] = _table(['Вариант', 'КИН', 'ΔКИН', 'Закачано, тыс. м³', 'Скин-фактор нагнетательной скважины',
                        'Парафин в осадке, т', 'Ниже WAT, % площади', 'Гель, % площади',
                        'Средняя температура, °C'], rows)
    for name, e in v.items():
        tag = name.upper()
        num[f'F_{tag}_RF'] = f"{e['rf']:.3f}"
        num[f'F_{tag}_DRF'] = f"{e['rf'] - base['rf']:+.4f}".replace('-', '−')
        num[f'F_{tag}_DRF_ABS'] = f"{abs(e['rf'] - base['rf']):.3f}"
        num[f'F_{tag}_INJ'] = f"{e['injected'] / 1e3:.1f}"
        num[f'F_{tag}_SKIN'] = f"{e['skin_inj']:.1f}"
        num[f'F_{tag}_KINJ'] = f"{e['k_inj']:.3f}"
        num[f'F_{tag}_KMIN'] = f"{e['k_min']:.2f}"
        num[f'F_{tag}_WAX'] = f"{e['wax_dep_t']:.0f}"
        num[f'F_{tag}_RET'] = f"{e['retained_t']:.1f}"
        num[f'F_{tag}_TMEAN'] = f"{e['T_mean']:.1f}"
        num[f'F_{tag}_GEL'] = f"{100 * e['gel_area']:.0f}"
        num[f'F_{tag}_WCUT'] = f"{100 * e['water_cut']:.0f}"
        q = 4.0 * np.array(e['series']['q_inj'])  # вся скважина: в элементе симметрии - ее четверть
        num[f'F_{tag}_Q0'] = f'{q[0]:.0f}'
        num[f'F_{tag}_QMIN'] = f'{q.min():.0f}'
        num[f'F_{tag}_QEND'] = f'{q[-1]:.0f}'
        num[f'F_{tag}_COLD'] = f"{100 * e['cold_area']:.1f}"
        num[f'F_{tag}_BELOW'] = f"{100 * e['below_wat_area']:.1f}"
        num[f'F_{tag}_WAXMAX'] = f"{100 * e['wax_dep_max']:.1f}"
        num[f'F_{tag}_PMEAN'] = f"{e['p_mean']:.1f}"
        num[f'F_{tag}_KPROD'] = f"{e['k_prod']:.2f}"
        num[f'F_{tag}_ASPHPROD'] = f"{100 * e['asph_dep_prod']:.0f}"
        num[f'F_{tag}_ASPH'] = f"{e['asph_dep_t']:.0f}"
        num[f'F_{tag}_QDROP'] = _opt(e['years_q_drop'], '{:.0f}', 365.0)
        num[f'F_{tag}_PB'] = _opt(e['years_below_pb'], '{:.1f}')
        num[f'F_{tag}_WAT0'] = f"{e['wat0'][1]:.1f}"
    rf = lambda n: v[n]['rf']
    inj = lambda n: v[n]['injected'] / 1e3
    num['F_T0'] = f'{t0:g}'
    if 'tiso' in v:
        num['F_INJ_LOSS'] = f"{100 * (1 - base['injected'] / v['tiso']['injected']):.0f}"
        num['F_COOL_LOSS'] = f"{rf('tiso') - rf('base'):.3f}"
    if 'nowax' in v:
        num['F_WAX_EFF'] = f"{rf('nowax') - rf('base'):.3f}"
        num['F_WAX_EFF_PCT'] = f"{100 * (rf('nowax') - rf('base')) / rf('nowax'):.1f}"
        num['F_WAX_INJ_PCT'] = f"{100 * (1 - base['injected'] / v['nowax']['injected']):.0f}"
        if 'tiso' in v:
            num['F_WAX_SHARE'] = f"{100 * (rf('nowax') - rf('base')) / (rf('tiso') - rf('base')):.0f}"
        if 'nogel' in v:  # доля геля в эффекте парафина
            num['F_GEL_SHARE'] = f"{100 * (rf('nogel') - rf('base')) / (rf('nowax') - rf('base')):.0f}"
    if all(n in v for n in ('base', 'hl0', 'nowax', 'hl0_nowax')):
        rows3 = []
        for label, f, fmt in (('КИН', rf, '{:+.3f}'), ('Закачано, тыс. м³', inj, '{:+.1f}')):
            fx = factorial(v, f)
            rows3.append([label] + [fmt.format(fx[k]).replace('-', '−')
                                    for k in ('hl_wax', 'hl_nowax', 'wax_hl', 'wax_nohl', 'inter')])
        num['T3'] = _table(['Показатель', 'Эффект теплообмена при наличии парафина', 'Эффект теплообмена без парафина',
                            'Эффект парафина при учете теплообмена', 'Эффект парафина без учета теплообмена',
                            'Взаимодействие'], rows3)
        fx = factorial(v, rf)
        for key, value in fx.items():
            num[f'F_FX_{key.upper()}'] = f'{abs(value):.3f}'
        num['F_FX_INTER_SIGNED'] = f"{fx['inter']:+.3f}".replace('-', '−')
        num['F_FX_INTER_PCT'] = f"{100 * fx['inter'] / fx['hl_wax']:.0f}" if fx['hl_wax'] else '—'
    if 'hl0' in v:
        num['F_HL0_WAXRATIO'] = f"{v['hl0']['wax_dep_t'] / max(base['wax_dep_t'], 1e-9):.1f}"
    if 'hl1' in v:
        num['F_HL1_WAXPCT'] = f"{100 * (v['hl1']['wax_dep_t'] / max(base['wax_dep_t'], 1e-9) - 1):.0f}"
        if 'hl0' in v and rf('base') != rf('hl0'):
            num['F_HL1_SHARE'] = f"{100 * (rf('hl1') - rf('hl0')) / (rf('base') - rf('hl0')):.0f}"
    # Варианты, совпавшие с базовым: наибольшее относительное расхождение показателей
    for name in ('ltne', 'equil', 'nolatent'):
        if name in v:
            rel = max(abs(v[name][key] - base[key]) / max(abs(base[key]), 1e-30)
                      for key in ('rf', 'T_mean', 'wax_dep_t', 'injected', 'k_inj'))
            num[f'F_{name.upper()}_REL'] = _sci(rel, 0) if rel > 0 else '0'
    num['F_WAT0'] = f"{base['wat0'][0]:.1f}–{base['wat0'][1]:.1f}"
    num['F_PB_YEARS'] = _opt(base['years_below_pb'], '{:.1f}')
    num['F_QDROP_DAYS'] = _opt(base['years_q_drop'], '{:.0f}', 365.0)
    num['F_FRONT_W'] = f"{base['front_water_1y']:.0f}"
    num['F_FRONT_T'] = f"{base['front_cold_1y']:.0f}"
    num['F_FRONT_RATIO'] = f"{base['front_cold_1y'] / max(base['front_water_1y'], 1e-9):.2f}"
    num['F_WIDTH'] = f"{base['front_width_1y']:.0f}"
    num['F_GROUPS'] = ' / '.join(f'{100 * x:.0f}' for x in base['wax_dep_groups'])
    if v.get('ltne', {}).get('ltne_dT_max') is not None:
        num['F_LTNE_DT'] = f"{v['ltne']['ltne_dT_max']:.2g}"

    # Безразмерные числа
    s_w = 1.0 - init_S  # водонасыщенность обводненной зоны - порядок величины
    rc_eff = (1 - init_m) * ro_f * c_f + init_m * (s_w * ro_w * c_w + (1 - s_w) * ro_o * c_o)
    lam = (1 - init_m) * K_f + init_m * (s_w * K_w + (1 - s_w) * K_o)
    q_full = 4.0 * base['series']['q_inj'][len(base['series']['q_inj']) // 2] / 86400.0  # полная скважина, м^3/с
    pe = q_full * ro_w * c_w / (2 * math.pi * h * lam)
    num['F_PE'] = f'{pe:.0f}'
    # Скорость теплового фронта к скорости воды: доля теплоемкости воды в теплоемкости пласта
    num['F_VT_RATIO'] = f'{init_m * ro_w * c_w / rc_eff:.2f}'
    t_inj = 20.0
    w0 = oc.GR_W0.sum()
    solid = lambda t: w0 - oc.sle_split_np(oc.GR_W0, oc.GR_M, oc.GR_TM_K, oc.GR_DH, oc.GR_DV, t,
                                           dp=12e6 - P_ref_wax, n_g=oc.gas_moles(12e6)).sum()
    live = solid(t_inj) - solid(t0)  # выпадает при охлаждении живой нефти от пластовой до температуры закачки
    ste = rc_eff * (t0 - t_inj) / (init_m * (1 - init_S) * ro_o * live * latent_heat)
    num['F_STE'] = f'{ste:.0f}'
    num['F_WPREC'] = f'{100 * live:.1f}'
    num['F_WAX0'] = f'{100 * w0:.1f}'
    # Удержание: рост константы Ленгмюра при охлаждении до температуры закачки и начальное повреждение D(sigma_0)
    from paraphin.geometry import surf_0
    from paraphin.constants import R, ro_asph_dep
    from paraphin.layout import IA_D, I_R
    from paraphin.oil_composition import initial_components
    kin = th['kin']
    t_ref = kin.get('ADS_T_REF', 70.0) + 273.15
    k_l = lambda t: kin['ADS_K'] * math.exp(-kin['ADS_DH'] / R * (1.0 / (t + 273.15) - 1.0 / t_ref))
    num['F_KRATIO'] = _sci(k_l(t_inj) / k_l(t0), 0)
    wc = initial_components()
    g_max = kin['ADS_GMAX'] * surf_0
    kl0 = k_l(t0)
    g0 = sum(g_max * mult * kl0 * c / (1.0 + kl0 * c) for c, mult in ((wc[IA_D], 1.0), (wc[I_R], kin.get('ADS_RESIN', 0.5))))
    rel0 = kin['PERM_BETA'] * g0 / ro_asph_dep / (kin['PERM_SMAX'] * init_m)
    num['F_SIG0'] = f'{rel0:.2f}'
    num['F_D0'] = f"{max(1.0 - rel0, 0.0) ** kin['PERM_GAMMA']:.2f}"
    # Тепловое неравновесие: время релаксации температур флюида и зерна при Nu = 2 (нижняя граница Вакао-Кагеи)
    from paraphin.kinetics_params import DEFAULTS
    ltne_dg = DEFAULTS['LTNE_DG']
    s_mid = 0.5
    lam_f = s_mid * K_w + (1 - s_mid) * K_o
    h_v = 6.0 * (1 - init_m) / ltne_dg * 2.0 * lam_f / ltne_dg
    c_fl = init_m * (s_mid * ro_w * c_w + (1 - s_mid) * ro_o * c_o)
    c_s = (1 - init_m) * ro_f * c_f
    num['F_TAU_LTNE'] = f'{1.0 / (h_v * (1 / c_fl + 1 / c_s)):.2f}'
    num['F_DG'] = f'{ltne_dg * 1e3:.1f}'
    # Дамкелер кристаллизации в поле: k_cryst на время прохождения теплового фронта через частицу нефти
    v_t = base['front_cold_1y'] / (365.0 * 86400.0)
    t_pass = base['front_width_1y'] / max(v_t, 1e-12)
    da = th['kin']['K_CRYST'] * t_pass
    num['F_DA'] = _sci(da, 0)
    num['F_TPASS'] = f'{t_pass / 86400:.0f}'
    # Доля пересыщения, кристаллизующаяся на стенках, k_w/(k_w + k_cr) по (13): в поле k_w - значение пакета
    k_w = th['kin'].get('K_WALL', DEFAULTS['K_WALL'])
    num['F_WALL_SHARE'] = f"{100 * k_w / (k_w + th['kin']['K_CRYST']):.0f}"


# --- Термодинамика и реология, подбор параметров (только полная версия) -------------------------------------------

EOS_DIR = EXP / 'уравнение_состояния'
# Теплоты плавления и твердо-твердого перехода н-C24 (Broadhurst, J. Res. NBS 1962, 66A:241, табл. 3), ккал/моль
BROADHURST_C24 = (13.12, 7.48)
STYLE_N = {1: '-', 2: '--', 3: ':', 4: '-.', 5: STYLES[4], 6: (0, (10, 3))}  # 6 - длинный штрих: (1, 1) не отличить от ':'
UQ_LINE = dict(linestyle='-', color='0.55', lw=2.2)  # кривая 7 рис. 9 - твердый раствор UNIQUAC
PSI = 6894.757  # Па в psi
# Разложение эффектов упрощенной однокомпонентной моделью (статья о теплопотерях, `old_article.md`, табл. 3;
# расчеты `однокомпонентная_модель/run_cases.py`, `make_figures.py`): КИН, эффекты парафина и взаимодействие
OLD_FACTORIAL = {'wax_hl': -0.0004, 'wax_nohl': -0.0173, 'inter': 0.0170, 'total': 0.0211}


def _jload(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def _rng(vals, d=1):
    lo, hi = min(vals), max(vals)
    return _f(lo, d) if round(lo, d) == round(hi, d) else f'{_f(lo, d)}–{_f(hi, d)}'


def _e10(e):
    """10 в целой степени надстрочными цифрами: 10⁻³."""
    return '10' + str(int(round(e))).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))


def _signed(x, d=1):
    return f'{x:+.{d}f}'.replace('-', '−')


def _asph_peak(p, frac):
    p, frac = np.asarray(p) / 1e6, 100.0 * np.asarray(frac)
    i, on = int(np.argmax(frac)), p[frac > 0.1]
    return frac[i], p[i], (on.min(), on.max()) if on.size else (float('nan'),) * 2


def fig_thermo(eos, cal, ds, num):
    """Рис. 9: кривая выпадения нефти Жетыбая и две синтетические смеси н-алканов."""
    import core_flood as cf
    exp = np.array(cf.LI_PRECIPITATION)
    z, tli, sc = eos['zhetybai'], np.array(eos['t_li']), cal['scn']
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    ax = axes[0]
    ax.plot(exp[:, 0], exp[:, 1], 'o', mfc='white', mec='k', ms=3.5)
    for n, key in ((1, 'eff'), (2, 'pr'), (3, 'ideal')):
        ax.plot(tli, z['curves'][key], linestyle=STYLE_N[n], color='k', lw=1.2, label=str(n))
    ax.plot(sc['T'], sc['legacy'], linestyle=STYLE_N[4], color='k', lw=1.2, label='4')
    for n, key in ((5, 'ss'), (6, 'ms_c')):
        ax.plot(tli, z['curves'][key], linestyle=STYLE_N[n], color='k', lw=1.2, label=str(n))
    ax.plot(tli, z['curves']['uq'], label='7', **UQ_LINE)
    ax.set_xlim(-20, 75)
    ax.set_ylim(0, 31)
    ax.set_xlabel('T, °C')
    ax.set_ylabel('w, % масс.')
    ax.legend(loc='upper right', fontsize=7, handlelength=2.4)
    _panel(ax, 'а')
    for ax, name, letter in zip(axes[1:], ('Bim 0', 'Bim 13'), 'бв'):
        t, s = np.array(ds['mixtures'][name]['solid_wt']).T
        ax.plot(t, s, 'o', mfc='white', mec='k', ms=3.5)
        c = eos['synthetic'][name]['curves']
        # 1 - эффективные параметры нефти Жетыбая на смеси известного состава (перенос не работает); 4 - один
        # псевдокомпонент нефти, на смесь н-алканов не переносится вовсе
        for n, key in ((1, 'eff'), (2, 'pr'), (3, 'ideal'), (5, 'ss'), (6, 'ms_c')):
            ax.plot(eos['t_syn'], c[key], linestyle=STYLE_N[n], color='k', lw=1.2, label=str(n))
        ax.plot(eos['t_syn'], c['uq'], label='7', **UQ_LINE)
        ax.set_xlim(-10, 45)
        ax.set_ylim(0, 40)
        ax.set_xlabel('T, °C')
        _panel(ax, letter)
    axes[2].legend(loc='upper right', fontsize=7, handlelength=2.4)
    fig.tight_layout()
    _save(fig, 'aspo_f9')


def fig_pressure_asph(eos, cal, num):
    """Рис. 10: сдвиг WAT с давлением у дегазированной и живой нефти, колокол выпадения асфальтенов."""
    pr = eos['pressure']
    sl = cal['pressure']['slopes'].values()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    ax = axes[0]
    pd = np.array(eos['p_dead']) / 1e6
    ax.fill_between(pd, min(sl) * (pd - pd[0]), max(sl) * (pd - pd[0]), color='0.85', lw=0)
    for n, key in ((1, 'eff'), (2, 'pr')):
        ax.plot(pd, np.array(pr['dead'][key]) - pr['dead'][key][0], linestyle=STYLE_N[n], color='k', label=str(n))
    ax.set_ylabel('ΔWAT, °C')
    ax.legend(loc='lower right', fontsize=7, handlelength=2.4)
    _panel(ax, 'а')
    ax = axes[1]
    pl = np.array(eos['p_live']) / 1e6
    for n, key in ((1, 'eff'), (2, 'pr')):
        ax.plot(pl, np.array(pr['live'][key]) - pr['live'][key][0], linestyle=STYLE_N[n], color='k', marker='o',
                ms=2.5)
    ax.axvline(eos['P_bubble'] / 1e6, color='0.5', lw=0.6, ls='--')
    _panel(ax, 'б')
    ax = axes[2]
    pa = np.array(eos['p_asph']) / 1e6
    for n, key in ((1, 'hirschberg'), (2, 'nghiem'), (3, 'nghiem_kij0')):
        ax.plot(pa, 100.0 * np.array(eos['asph']['precipitated'][key]), linestyle=STYLE_N[n], color='k', lw=1.2,
                label=str(n))
    for x in (eos['P_bubble'], eos['P_onset_asph']):
        ax.axvline(x / 1e6, color='0.5', lw=0.6, ls='--')
    ax.set_ylabel('выпало, %')
    ax.set_ylim(0, 55)
    ax.legend(loc='upper right', fontsize=7, handlelength=2.4)
    _panel(ax, 'в')
    for ax in axes:
        ax.set_xlabel('P, МПа')
    fig.tight_layout()
    _save(fig, 'aspo_f10')


def fig_tabzar(eos, num):
    """Рис. 12: выпадение асфальтенов из живой нефти Tabzar et al. (2018) - модель Нгхайема, калибровка по точке начала
    осаждения и по кривой выпавших."""
    tz, data = eos['tabzar'], _jload(EXP / 'data' / 'tabzar2018_asphaltene.json')
    mpa = lambda psig: (np.asarray(psig, float) + 14.696) * PSI / 1e6  # noqa: E731
    p_exp, w_exp = np.array(data['precipitated']).T
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(mpa(p_exp), w_exp, 'o', mfc='white', mec='k', ms=3.5, zorder=3)
    for n, key in ((1, 'curve'), (2, 'vs'), (3, 'onset')):
        ax.plot(mpa(eos['p_tabzar_psig']), tz['variants'][key]['curve'], linestyle=STYLE_N[n], color='k', lw=1.2,
                label=str(n))
    ax.axvline(data['P_bubble_psia'] * PSI / 1e6, color='0.5', lw=0.6, ls='--')
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 2)
    ax.set_xlabel('P, МПа')
    ax.set_ylabel('w, % масс.')
    ax.legend(loc='upper right', fontsize=7, handlelength=2.4)
    fig.tight_layout()
    _save(fig, 'aspo_f12')


def fig_viscosity(cal, num):
    """Рис. 11: вязкость нефти Жетыбая при охлаждении (Li et al., 150 1/с) и модели."""
    rh = cal['rheology']
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.semilogy(rh['T'], rh['mu'], 'o', mfc='white', mec='k', ms=3.5)
    for n, key in ((1, 'bingham'), (2, 'kd'), (3, 'pr_full')):
        ax.semilogy(rh['T'], rh[key], linestyle=STYLE_N[n], color='k', lw=1.2, label=str(n))
    ax.set_ylim(5, 2000)
    ax.set_xlabel('T, °C')
    ax.set_ylabel('μ, мПа·с')
    ax.legend(loc='upper right', fontsize=7, handlelength=2.4)
    fig.tight_layout()
    _save(fig, 'aspo_f11')


def thermo_numbers(eos, cal, ds, val, params, th, num):
    """Числа раздела о термодинамике и реологии, разбора расхождений и подбора параметров."""
    import core_flood as cf
    import li2024
    import sandyga2020
    import sutton_roberts
    from paraphin import oil_composition as oc
    from paraphin.constants import MW, R, gamma, r_m, sigma_r, r_max, ro_p, ro_wax_liq
    exp = dict(cf.LI_PRECIPITATION)

    # A. Кривая выпадения: один псевдокомпонент по Вону без подбора, он же с эффективными параметрами, 4 группы, EOS
    w1, m1 = np.array([oc.WAX_TOTAL]), np.array([float(MW)])
    tm1, dh1 = oc.won_tm(m1), oc.won_dh(m1)
    prec1 = lambda t: 100 * (w1.sum() - oc.sle_split_np(w1, m1, tm1, dh1, np.zeros(1), t).sum())
    num['TH_WON1_WAT'] = _f(oc.wat_np(w1, m1, tm1, dh1, np.zeros(1)), 1)
    num['TH_WON1_35'], num['TH_WON1_25'] = _f(prec1(35.0), 1), _f(prec1(25.0), 1)
    num['TH_EXP_35'], num['TH_EXP_25'] = _f(exp[35.0], 1), _f(exp[25.0], 1)
    num['TH_WAT_EXP'] = _f(cf.LI_PRECIPITATION[0][0], 2)
    sc = cal['scn']
    num['TH_RMS_GROUPS'], num['TH_RMS_LEGACY'] = _f(100 * sc['rms_groups'], 2), _f(100 * sc['rms_legacy'], 2)
    num['TH_WAT_GROUPS'], num['TH_WAT_LEGACY'] = _f(sc['wat'], 1), _f(sc['wat_legacy'], 1)
    num['TH_SLOPE'] = _f(sc.get('slope', 0.07), 2)
    num['TH_DH'], num['TH_SHIFT'] = _f(sc['alpha'] / 1e3, 1), _f(sc['shift'], 1)
    z = eos['zhetybai']
    for key, tag in (('pr', 'PR'), ('ideal', 'IDEAL'), ('ss', 'SS')):
        num[f'EOS_LI_WAT_{tag}'] = _f(z['wat'][key], 1)
        num[f'EOS_LI_RMS_{tag}'] = _f(z['rms_fit'][key], 1)
    num['EOS_LI_DWAT'] = _f(z['wat']['pr'] - cf.LI_PRECIPITATION[0][0], 0)
    num['EOS_LI_PR_IDEAL'] = _f(z['wat']['pr'] - z['wat']['ideal'], 1)

    # Б. Синтетические смеси (da Silva et al., 2017)
    syn = eos['synthetic']
    dev = lambda key: [m['wdt'][key] - ds['mixtures'][n]['wdt_exp'] for n, m in syn.items()]
    auth = lambda model: [ds['mixtures'][n]['wdt_models'][model] - ds['mixtures'][n]['wdt_exp'] for n in syn]
    num['EOS_SYN_DPR'], num['EOS_SYN_DIDEAL'] = _signed(np.mean(dev('pr'))), _signed(np.mean(dev('ideal')))
    num['EOS_SYN_DSS_RNG'] = f"{_signed(min(dev('ss')))}…{_signed(max(dev('ss')))}"
    num['EOS_SYN_DAUTH_MS'] = _signed(np.mean(auth('PR+Multisolid')))
    num['EOS_SYN_DAUTH_SS'] = _signed(np.mean(auth('PR+IdealSolidSolution')))
    num['EOS_SYN_DAUTH_UQ'] = _signed(np.mean(auth('PR+UNIQUAC')))
    num['EOS_SYN_DPR_ABS'] = _f(abs(np.mean(dev('pr'))), 1)
    num['EOS_SYN_DIDEAL_ABS'] = _f(abs(np.mean(dev('ideal'))), 1)
    num['EOS_SYN_DAUTH_MS_ABS'] = _f(abs(np.mean(auth('PR+Multisolid'))), 1)
    num['EOS_SYN_SHIFT'] = _f(np.mean([m['wdt']['pr'] - m['wdt']['ideal'] for m in syn.values()]), 1)
    for key, tag in (('pr', 'PR'), ('ideal', 'IDEAL'), ('ss', 'SS')):
        num[f'EOS_SYN_RMS_{tag}'] = _rng([m['rms'][key] for m in syn.values()])
    num['EOS_SYN_AAD_UQ'] = _rng([ds['mixtures'][n]['aad_models']['PR+UNIQUAC'] for n in syn], 2)
    srng = lambda v: f'{_signed(min(v))}…{_signed(max(v))}'  # noqa: E731
    # Твердо-твердый переход и твердый раствор UNIQUAC по Coutinho (`paraphin/thermo`)
    for key, tag in (('ms_c', 'MSC'), ('ms_cp', 'MSCP'), ('uq', 'UQ'), ('uq_id', 'UQID')):
        num[f'EOS_SYN_D{tag}_RNG'] = srng(dev(key))
        num[f'EOS_SYN_RMS_{tag}'] = _rng([m['rms'][key] for m in syn.values()])
    num['EOS_SYN_DPR_RNG'] = srng(dev('pr'))
    # настоящая модель с эффективными параметрами нефти Жетыбая на смесях: проверка переноса параметров
    num['EOS_SYN_DEFF_RNG'], num['EOS_SYN_RMS_EFF'] = srng(dev('eff')), _rng([m['rms']['eff'] for m in syn.values()])
    num['EOS_SYN_DUQ_ABS'] = _f(max(abs(x) for x in dev('uq')), 1)
    num['EOS_SYN_DAUTH_UQ_RNG'] = srng(auth('PR+UNIQUAC'))
    num['EOS_SYN_AAD_UQ_OWN'] = _rng([m['aad']['uq'] for m in syn.values()], 2)
    for key, tag in (('ms_c', 'MSC'), ('uq', 'UQ')):
        num[f'EOS_LI_WAT_{tag}'] = _f(z['wat'][key], 1)
        num[f'EOS_LI_RMS_{tag}'] = _f(z['rms_fit'][key], 1)
    uf = z['uq_fit']
    num['EOS_LI_UQFIT_RMS'], num['EOS_LI_UQFIT_WAT'] = _f(uf['rms'], 1), _f(uf['wat'], 0)
    num['EOS_LI_UQFIT_S'], num['EOS_LI_UQFIT_LAST'] = _f(uf['scn_slope'], 3), str(uf['scn_last'])
    num['EOS_LI_WAT_HI'] = _f(max(z['wat'][k] for k in ('pr', 'ideal', 'ss', 'ms_c', 'uq'))
                              - cf.LI_PRECIPITATION[0][0], 0)
    m24 = 14.027 * 24 + 2.016
    num['BR_C24_WON'] = _f(oc.won_dh(np.array([m24]))[0] / 1e3, 0)
    f_kj, t_kj = (x * 4.184 for x in BROADHURST_C24)
    num['BR_C24_F'], num['BR_C24_T'], num['BR_C24_TOT'] = _f(f_kj, 1), _f(t_kj, 1), _f(f_kj + t_kj, 0)
    num['BR_C24_TR_PCT'] = _f(100 * t_kj / (f_kj + t_kj), 0)

    # В. Давление и газ
    pr, slopes = eos['pressure'], list(cal['pressure']['slopes'].values())
    num['PRESS_SANDYGA'] = f'{_f(min(slopes), 2)}–{_f(max(slopes), 2)}'
    num['PRESS_SL_EFF'] = _f(pr['slope_dead']['eff'], 3)
    num['PRESS_SL_PR'] = _f(pr['slope_dead']['pr'], 2)
    num['PRESS_SL_PR0'] = _f(pr['slope_dead']['pr_no_dv'], 3)
    i_b = eos['p_live'].index(eos['P_bubble'])
    for key, tag in (('eff', 'EFF'), ('pr', 'PR')):
        live = pr['live'][key]
        num[f'PRESS_DROP_{tag}'] = _f(live[0] - live[i_b], 1)
        num[f'PRESS_RISE_{tag}'] = _f((live[-1] - live[i_b]) / ((eos['p_live'][-1] - eos['P_bubble']) / 1e6), 2)
    num['PRESS_KIJ'] = _f(pr['kij_gas'], 3).replace('-', '−')
    num['PRESS_DV_EFF'] = _f(cal['pressure']['dv_frac'], 3)
    num['PRESS_DV_EOS'] = _f(1.0 - ro_wax_liq / ro_p, 2)

    # Г. Асфальтены: модели и лабораторные условия опыта Li
    a = eos['asph']['precipitated']
    for key, tag in (('hirschberg', 'H'), ('nghiem', 'N'), ('nghiem_kij0', 'N0')):
        peak, p_at, (lo, hi) = _asph_peak(eos['p_asph'], a[key])
        num[f'ASPH_{tag}_MAX'], num[f'ASPH_{tag}_P'] = _f(peak, 0), _f(p_at, 1)
        num[f'ASPH_{tag}_WIN'] = f'{_f(lo, 1)}–{_f(hi, 1)}'
    dead = max(max(v) for v in val['asphaltenes']['dead'].values())
    num['ASPH_LAB_DEAD'] = _f(100 * dead, 1)
    pl = val['asphaltenes']['plateau_exp']
    num['ASPH_LAB_PLATEAU'] = f"{_f(min(pl.values()), 2)}–{_f(max(pl.values()), 2)}"
    num['ASPH_DELTA_A'] = _f(val['asphaltenes']['delta_a'], 2)
    # Калибровка Нгхайема по кривой выпавших (Tabzar et al., 2018)
    tz, tzd = eos['tabzar'], _jload(EXP / 'data' / 'tabzar2018_asphaltene.json')
    tv = tz['variants']
    num['TZ_ASPH_WT'] = _f(tz['asph_wt'], 1)
    num['TZ_T'] = _f((tzd['T_F'] - 32.0) / 1.8, 0)
    num['TZ_PB'], num['TZ_PON'] = _f(tzd['P_bubble_psia'] * PSI / 1e6, 1), _f(tzd['P_onset_psia'] * PSI / 1e6, 1)
    num['TZ_PB_PR0'] = _f(tz['p_b_pr_kij0'] * PSI / 1e6, 1)
    num['TZ_EXP_RNG'] = _rng([w for _, w in tzd['precipitated']], 2)
    num['TZ_ONSET_RNG'] = _rng(tv['onset']['at_exp'], 0)
    num['TZ_KIJ_AUTH'], num['TZ_VS_AUTH'] = _f(tv['onset']['kij'], 1), _f(tv['onset']['V_s'], 1)
    num['TZ_ONSET_RMS'], num['TZ_VS04_RMS'] = _f(tv['onset']['rms'], 1), _f(tv['vs']['rms'], 2)
    num['TZ_VS04_LOW'] = _f(tv['vs']['at_exp'][0], 2)
    c = tv['curve']
    num['TZ_KIJ'], num['TZ_VS'], num['TZ_VBAR'] = _f(c['kij'], 3), _f(c['V_s'], 3), _f(c['v_bar'], 3)
    num['TZ_VS_DEV_PCT'], num['TZ_RMS'] = _f(100 * (c['V_s'] / c['v_bar'] - 1.0), 1), _f(c['rms'], 2)
    # PVT нефти модели по уравнению состояния против линейного R_s симулятора
    pv = eos['pvt']
    rel = np.abs(np.array(pv['Rs']) / np.array(pv['Rs_linear']) - 1.0)
    num['PVT_RS_DEV'], num['PVT_RS_DEV_P'] = _f(100 * rel.max(), 0), _f(eos['p_pvt'][int(rel.argmax())] / 1e6, 1)
    i6 = eos['p_pvt'].index(6e6)
    num['PVT_FREE6'], num['PVT_BO_B'] = _f(pv['free_gas'][i6], 2), _f(pv['Bo'][eos['p_pvt'].index(eos['P_bubble'])], 3)

    # Д. Вязкость и гель
    rh = cal['rheology']
    lnr = lambda key: float(np.sqrt(np.mean((np.log(rh[key]) - np.log(rh['mu'])) ** 2)))
    num['VISC_RMS'], num['VISC_RMS_KD'] = _f(lnr('bingham'), 3), _f(lnr('kd'), 2)
    num['VISC_KD_UNDER'] = _f(rh['mu'][0] / rh['kd'][0], 1)
    num['VISC_PR_OVER'] = _sci(rh['pr_full'][0] / rh['mu'][0], 0)
    num['VISC_T0'] = _f(rh['T'][0], 1)
    num['VISC_MU25'], num['VISC_E'] = _f(rh['mu25'], 1), _f(rh['E'] / 1e3, 1)
    num['VISC_D'], num['GEL_PHI'] = _f(rh['visc_D'], 2), _f(100 * rh['gel_phi'], 1)
    num['GEL_TAU'], num['GEL_N'] = _f(rh['gel_tau_ref'], 1), _f(rh['gel_n'], 2)
    runs = val['runs']
    for key, tag in (('sr1_legacy', 'SR_LEG'), ('sr1_gel_1.0', 'SR_1'), ('sr1_gel_0.1', 'SR_01'),
                     ('sr1_gel_0.03', 'SR_003'), ('sr2_legacy', 'SR2_LEG'), ('sr2_gel_0.03', 'SR2_003'),
                     ('li25_comp', 'LI_COMP'), ('li25_gel_1.0', 'LI_1')):
        num[f'GEL_{tag}'] = _f(runs[key]['rms'], 3)
    num['GEL_SD_STATIC'] = _f(1.0 / min(runs['sd_gelphi_0.02_1.0']['k']), 1)
    num['GEL_SD_LEG'] = _f(1.0 / min(runs['sd_legacy']['k']), 1)

    # Е. Причины расхождений в керновых опытах
    sr = _load('sutton_roberts')
    g = lambda d, n: d[n] if n in d else d[str(n)]
    joint = sr['lsq']['rate_entrainment']['joint']['rms']
    num['SR_JOINT'] = _rng([g(joint, 1), g(joint, 2)], 2)
    cross = sr['entrainment']['cross']
    num['SR_CROSS'] = f"{_f(min(cross.values()), 2)}–{_f(max(cross.values()), 2)}"
    x = sr['visc']['x']
    mult = (10 ** x[3], 10 ** x[4])
    num['SR_MULT1'], num['SR_MULT2'] = _f(mult[0], 1), _f(mult[1], 2)
    num['SR_MU1'], num['SR_MU2'] = _f(3.0 / mult[0], 0), _f(3.0 / mult[1], 0)
    sd = _load('sandyga2020')
    loss, loss_exp = sd['best']['pore_loss'], sd['best']['pore_loss_exp']
    num['SD_LOSS_NARROW'], num['SD_LOSS_WIDE'] = _rng(loss[:2], 2), _rng(loss[2:], 2)
    num['SD_LOSS_EXP'] = _rng(loss_exp, 2)

    # Ж. Пласт: доля блокируемых каналов при d_p полевого расчета, чувствительность вязкости к E_a
    r = np.linspace(1e-9, r_max, 4001)
    phi0 = np.exp(-np.log(r / r_m) ** 2 / (2 * sigma_r ** 2)) / r
    d_p = th['kin']['D_CRYST'] if th else 15e-6
    r_pass = d_p / (2 * gamma)
    below = r <= r_pass
    num['F_RPASS'] = _f(r_pass * 1e6, 1)
    vol = np.trapezoid(r[below] ** 2 * phi0[below], r[below]) / np.trapezoid(r ** 2 * phi0, r)
    cond = np.trapezoid(r[below] ** 4 * phi0[below], r[below]) / np.trapezoid(r ** 4 * phi0, r)
    num['F_VOL_BLOCK'], num['F_COND_BLOCK'] = _f(100 * vol, 0), _f(100 * cond, 0)
    # то же при диаметре кристалла общего набора Sutton и Roberts (распределение пор в керне - как в пласте)
    rp_sr = 10 ** params['sutton_roberts_lsq']['rate_entrainment']['joint'][0] / (2 * gamma)
    b_sr = r <= rp_sr
    num['SR_RPASS'] = _f(rp_sr * 1e6, 1)
    num['SR_VOL_BLOCK'] = _f(100 * np.trapezoid(r[b_sr] ** 2 * phi0[b_sr], r[b_sr]) / np.trapezoid(r ** 2 * phi0, r), 0)
    num['SR_COND_BLOCK'] = _f(100 * np.trapezoid(r[b_sr] ** 4 * phi0[b_sr], r[b_sr]) / np.trapezoid(r ** 4 * phi0, r), 0)
    t0 = (th['init_T'] if th else 55.0) + 273.15
    ratio = lambda e: math.exp(e / R * (1.0 / 293.15 - 1.0 / t0))
    num['F_MU_RATIO_25'], num['F_MU_RATIO_FIT'] = _f(ratio(25e3), 1), _f(ratio(rh['E']), 1)

    # Упрощенная модель: эффект парафина и доля взаимодействия в суммарном изменении КИН
    of = OLD_FACTORIAL
    num['OLD_FX_WAX_HL'] = _f(abs(of['wax_hl']), 4)
    num['OLD_FX_WAX_NOHL'] = _f(abs(of['wax_nohl']), 3)
    num['OLD_FX_INTER_PCT'] = _f(100 * of['inter'] / of['total'], 0)
    # Таблица 4: термодинамика и реология против независимых измерений
    rows = [
        ('Кривая выпадения, нефть Жетыбая [19]', 'СКО, % масс.', '—', num['TH_RMS_GROUPS'], num['EOS_LI_RMS_PR'],
         f"один псевдокомпонент: {num['TH_RMS_LEGACY']}; PR + UNIQUAC: {num['EOS_LI_RMS_UQ']}, с подобранным "
         f"распределением н-алканов: {num['EOS_LI_UQFIT_RMS']}"),
        ('То же', 'WAT, °C', num['TH_WAT_EXP'], num['TH_WAT_GROUPS'], num['EOS_LI_WAT_PR'],
         f"один псевдокомпонент: {num['TH_WAT_LEGACY']}; по Вону без подбора: {num['TH_WON1_WAT']}; "
         f"PR + UNIQUAC: {num['EOS_LI_WAT_UQ']}"),
        ('Синтетические смеси C18–C36 [30]', 'WDT − WDT опыта, °C', '—', num['EOS_SYN_DIDEAL'], num['EOS_SYN_DPR'],
         f"PR + переход по Coutinho: {num['EOS_SYN_DMSC_RNG']}; PR + UNIQUAC: {num['EOS_SYN_DUQ_RNG']}; "
         f"PR + multi-solid [30]: {num['EOS_SYN_DAUTH_MS']}; PR + UNIQUAC [30]: {num['EOS_SYN_DAUTH_UQ']}"),
        ('То же', 'отклонение доли твердого, % масс.', '—', num['EOS_SYN_RMS_IDEAL'], num['EOS_SYN_RMS_PR'],
         f"PR + переход по Coutinho: {num['EOS_SYN_RMS_MSC']}; PR + UNIQUAC: {num['EOS_SYN_RMS_UQ']} "
         f"(среднее абсолютное {num['EOS_SYN_AAD_UQ_OWN']}); PR + UNIQUAC [30]: {num['EOS_SYN_AAD_UQ']}"),
        ('Сдвиг WAT с давлением [46]', 'dWAT/dP, °C/МПа', num['PRESS_SANDYGA'], num['PRESS_SL_EFF'],
         num['PRESS_SL_PR'], f"PR без скачка объема: {num['PRESS_SL_PR0']}"),
        ('Вязкость при охлаждении [19]', 'СКО ln μ', '—', num['VISC_RMS'], '—',
         f"Кригер—Догерти: {num['VISC_RMS_KD']}"),
        ('Асфальтены у давления насыщения', 'выпало, % содержания', 'нет замеров', num['ASPH_H_MAX'],
         num['ASPH_N_MAX'], f"Nghiem, kij = 0: {num['ASPH_N0_MAX']} при {num['ASPH_N0_P']} МПа"),
        ('Асфальтены, живая нефть [34]', 'СКО выпавших, % масс.', '—', '—', num['TZ_RMS'],
         f"Нгхайем по точке начала осаждения: {num['TZ_ONSET_RMS']}; V_s по кривой при k_ij = "
         f"{num['TZ_KIJ_AUTH']}: {num['TZ_VS04_RMS']}"),
    ]
    num['T4'] = _table(['Измерение', 'Величина', 'Опыт', 'Настоящая модель', 'Уравнение состояния',
                        'Другие модели'], rows)

    # Таблица 5: что подбирается и по каким данным
    lsq = sutton_roberts.LSQ_FAMILIES['rate_entrainment']
    sj = params['sutton_roberts_lsq']['rate_entrainment']['joint']
    ad = params['li2024']['solid']['kin']
    cold = params['li2024_cold']['x']
    sdb = params['sandyga2020']['best']
    um = lambda lg, d=1: _f(10 ** lg * 1e6, d)
    rows = [
        ('Кривая выпадения и WAT нефти Жетыбая [19]', 'наклон распределения s, эффективная теплота ΔH_eff, сдвиг '
         'T_m; перебор разбиений на группы', 'метод наименьших квадратов', 'доля выпавшего при 10–45°C и WAT',
         f"s = {num['TH_SLOPE']}, ΔH_eff = {num['TH_DH']} кДж/моль, сдвиг {num['TH_SHIFT']} K"),
        ('Сдвиг WAT с давлением [46]', 'доля скачка объема Δv/v_L в (11)', 'по среднему наклону регрессий '
         'для 10–60% парафина', 'dWAT/dP', f"Δv/v_L = {num['PRESS_DV_EFF']}"),
        ('Вязкость при 24–100°C [19]', 'μ_ref, E_a (выше 40°C); D, φ_gel, τ_ref, n_g (ниже)',
         'наименьшие квадраты по ln μ, μ = μ_p + τ_y/γ̇', 'СКО ln μ',
         f"μ(25°C) = {num['VISC_MU25']} мПа·с, E_a = {num['VISC_E']} кДж/моль, D = {num['VISC_D']}, "
         f"φ_gel = {num['GEL_PHI']}%, τ_ref = {num['GEL_TAU']} Па, n_g = {num['GEL_N']}"),
        ('Давление начала осаждения асфальтенов', 'δ_a (Хиршберг) или f_s* (Нгхайем)',
         'насыщение в одной точке', '—', f"δ_a = {num['ASPH_DELTA_A']} МПа⁰·⁵"),
        ('Кривая выпавших асфальтенов [34]', 'k_ij асфальтены—легкие 0–0.5, V_s (Нгхайем); v_a (Хиршберг)',
         'перебор k_ij с шагом 0.005, V_s — метод наименьших квадратов; k_ij метан—C7+ по давлению насыщения',
         'СКО выпавших, % масс.', f"k_ij = {num['TZ_KIJ']}, V_s = {num['TZ_VS']} л/моль"),
        ('Sutton и Roberts [27]', f"d_p = {um(lsq['lo'][0], 0)}–{um(lsq['hi'][0], 0)} мкм, f_D = "
         f"{_e10(lsq['lo'][1])}–{_e10(lsq['hi'][1])}, порог выноса {lsq['lo'][2]:g}–{lsq['hi'][2]:g} Па и его "
         f"скорость {_e10(lsq['lo'][3])}–{_e10(lsq['hi'][3])} с⁻¹", 'метод наименьших квадратов по расчетам керна: '
         'общий набор, набор на опыт, перекрестный прогноз', 'СКО k/k₀',
         f"общий набор: d_p = {um(sj[0])} мкм, f_D = {_f(10 ** sj[1], 2)}, порог {_f(sj[2], 1)} Па, "
         f"скорость {_sci(10 ** sj[3], 1)} с⁻¹"),
        ('Li и др., 90–45°C [19]', f"Γ_max = {_e10(li2024.FIT_LO[0])}–{_e10(li2024.FIT_HI[0])} кг/м², "
         f"K_ref = {_e10(li2024.FIT_LO[1])}–{_e10(li2024.FIT_HI[1])}, −ΔH_ads = 0–{li2024.FIT_HI[2] * 10:g} кДж/моль, "
         f"k_ads = {_e10(li2024.FIT_LO[3])}–{_e10(li2024.FIT_HI[3])} с⁻¹", 'точно по трем плато, затем метод '
         'наименьших квадратов; две формы кинетики', 'СКО k/k₀ ступеней',
         f"Γ_max = {_sci(ad['ADS_GMAX'], 1)} кг/м², K_ref = {_f(ad['ADS_K'], 0)}, ΔH_ads = "
         f"−{_f(-ad['ADS_DH'] / 1e3, 0)} кДж/моль, k_ads = {_sci(ad['ADS_RATE'], 1)} с⁻¹"),
        ('Li и др., 25°C [19]', f"d_p = {um(li2024.COLD_LO[0], 0)}–{um(li2024.COLD_HI[0], 0)} мкм, f_D = "
         f"{_e10(li2024.COLD_LO[1])}–{_e10(li2024.COLD_HI[1])}, k_cr = {_e10(li2024.COLD_LO[2])}–"
         f"{_e10(li2024.COLD_HI[2])} с⁻¹; тиксотропное время геля "
         f"{', '.join(f'{t / 60:g}' for t in sorted(li2024.COLD_GEL_TIMES))} мин", 'метод наименьших квадратов',
         'СКО k/k₀', f"d_p = {um(cold[0])} мкм, f_D = {_f(10 ** cold[1], 2)}, 1/k_cr = {_f(10 ** -cold[2], 0)} с, "
         f"время геля {params['li2024_cold']['gel_time'] / 60:g} мин"),
        ('Sandyga и др. [2]', f"WAT пористой среды {', '.join(f'{w:g}' for w in sandyga2020.WATS)}°C; "
         f"k_w = {_e10(sandyga2020.FIT_LO[0])}–1 с⁻¹, C_0 = {_e10(sandyga2020.FIT_LO[1])}–1, "
         f"k_cr = {_e10(sandyga2020.FIT_LO[2])}–1 с⁻¹", 'сетка, затем метод наименьших квадратов',
         'СКО lg(∇p/∇p₀)', f"WAT {sdb[0]:g}°C, k_w = {_sci(sdb[1], 1)} с⁻¹, C_0 = {_f(sdb[2], 1)}, "
         f"1/k_cr = {_f(1 / sdb[3] / 60, 0)} мин"),
        ('He и др. [28]', 'отношение горла к поре (сеть), показатель n (степенная зависимость)',
         'одномерная минимизация', 'СКО k/k₀', f"горло/пора 0.40 при z = 8; n = {num.get('HE_POWER_N', '—')}"),
        ('Вариант на уравнении состояния', 'k_ij газ—нефть', 'по давлению насыщения при T_0', '—',
         f"k_ij = {num['PRESS_KIJ']}; T_m и ΔH — по Вону или Coutinho с переходом, без подбора"),
    ]
    num['T5'] = _table(['Данные', 'Что варьируется', 'Метод', 'Критерий', 'Результат'], rows)


def thermo_rheology(num, th):
    """Сопоставление термодинамики и реологии с независимыми измерениями (рис. 9-11, табл. 4) и подбор
    параметров (табл. 5) - только для полной версии статьи."""
    sys.path.insert(0, str(EXP / 'исходная_модель'))
    cal, eos = _load('calibration'), _jload(EOS_DIR / 'results.json')
    ds, val = _jload(EXP / 'data' / 'dasilva2017_sle.json'), _jload(EXP / 'состав' / 'validation.json')
    params = _jload(EXP / 'params.json')
    if not (cal and eos and ds and val and params):
        print('нет сравнений термодинамики: experiments/calibrate.py, experiments/уравнение_состояния/compare.py, '
              'experiments/состав/validate.py')
        return
    fig_thermo(eos, cal, ds, num)
    fig_pressure_asph(eos, cal, num)
    fig_viscosity(cal, num)
    fig_tabzar(eos, num)
    thermo_numbers(eos, cal, ds, val, params, th, num)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    num = {}
    sr, li, sd, pm, he = (_load(n) for n in ('sutton_roberts', 'li2024', 'sandyga2020', 'pore_models', 'he2020'))
    fig_sutton_roberts(sr, num)
    fig_li(li, num)
    fig_sandyga(sd, num)
    fig_he(pm, he, num)
    ml = _load('maloney2004')  # закачка холодной воды при остаточной нефти: прогноз без подбора
    if ml:
        v = sorted(ml['variants'].values(), key=lambda e: -e['wax'])
        num['ML_WAX_HI'], num['ML_WAX_LO'] = _f(100 * v[0]['wax'], 1), _f(100 * v[1]['wax'], 0)
        num['ML_K_HI'], num['ML_K_LO'] = _f(v[0]['k10_26'][1], 2), _f(v[1]['k10_26'][1], 2)
        num['ML_M_HI'] = _f(100 * (1.0 - v[0]['m_end']), 1)
    table_experiments(num)
    th_path = RESULTS / 'thermal.json'
    th = None
    if th_path.is_file():
        th = json.loads(th_path.read_text(encoding='utf-8'))
        if 'maps' in th:
            fig_field_maps(th, num)
            fig_field_evolution(th, num)
        fig_field_time(th, num)
        field_numbers(th, num)
    else:
        print('нет results/thermal.json - сначала python твт_статья_АСПО/run_thermal.py')
    thermo_rheology(num, th)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'numbers.json').write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding='utf-8')
    lines = ['# Числа статьи', '', 'Создается `python твт_статья_АСПО/make_article.py`, руками не править.', '']
    for key, value in num.items():
        lines += [f'**{key}**', '', value, ''] if '\n' in value else [f'- `{key}`: {value}']
    (HERE / 'metrics.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'{len(num)} чисел -> results/numbers.json, metrics.md; рисунки -> figures/')


if __name__ == '__main__':
    main()
