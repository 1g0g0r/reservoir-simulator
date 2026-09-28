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
    axes[0].set_ylabel('k/k₀')
    axes[1].legend(loc='lower left', fontsize=8, handlelength=2.6)
    fig.tight_layout()
    _save(fig, 'aspo_f1')


def fig_li(li, num):
    import li2024 as m
    fig, ax = plt.subplots(figsize=(6.7, 2.7))
    best = m.stage_curves(li['cold']['result'])
    legacy = m.stage_curves(li['legacy'])
    start = 0.0
    for t, pv_stage in m.STAGES:
        pts = np.array(m.DATA['k_pv'][str(int(t))])
        ax.plot(start + pts[:, 0], pts[:, 1], 'o', mfc='white', mec='k', ms=3.5)
        if t in best:
            x, y = best[t]
            ax.plot(start + x, y, '-', color='k', label='1' if start == 0 else None)
        if t in legacy:
            x, y = legacy[t]
            ax.plot(start + x, y, '--', color='k', lw=1.2, label='2' if start == 0 else None)
        ax.axvline(start, color='0.6', lw=0.6)
        ax.text(start + 0.5 * pv_stage, 1.08, f'{t:.0f}°C', ha='center', fontsize=8)
        start += pv_stage
    ax.set_xlim(0, start)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel('V')
    ax.set_ylabel('k/k₀')
    ax.legend(loc='lower left', fontsize=8, handlelength=2.6)
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
    num['T1'] = _table(['Опыт', 'Шумовой порог', 'Прежняя модель', 'Настоящая модель', 'Другие модели'], rows)
    own = [float(r[3]) for r in rows]
    num['RMS_MIN'], num['RMS_MAX'] = f'{min(own):.3f}', f'{max(own):.3f}'
    legacy = [float(r[2]) for r in rows if r[2] not in ('закупорка', '—')]
    num['LEG_MIN'], num['LEG_MAX'] = f'{min(legacy):.2f}', f'{max(legacy):.2f}'


# --- Поле --------------------------------------------------------------------------------------------------

def fig_field_maps(th, num):
    """Карты базового варианта: четверть элемента у нагнетательной скважины - вся остывшая и поврежденная зона."""
    from paraphin.constants import X_max, Y_max
    maps = th['maps']
    zoom = 0.5  # доля стороны элемента
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.1))
    for ax, key, levels, letter, fmt in (
            (axes[0], 'T', np.arange(25, 70, 5), 'а', '%.0f'),
            (axes[1], 'k', (0.1, 0.2, 0.3, 0.5, 0.7, 0.9), 'б', '%.1f')):
        z = np.array(maps[key]).T
        ny, nx = z.shape
        # значения в центрах ячеек, продленные на границы, чтобы заливка доходила до краев
        x = np.concatenate(([0.0], (np.arange(nx) + 0.5) * X_max / nx, [X_max]))
        y = np.concatenate(([0.0], (np.arange(ny) + 0.5) * Y_max / ny, [Y_max]))
        z = np.pad(z, 1, mode='edge')
        fill = np.concatenate(([z.min() - 1e-9], levels, [z.max() + 1e-9]))
        ax.contourf(x, y, z, levels=fill, cmap='Greys_r', alpha=0.5)
        cs = ax.contour(x, y, z, levels=levels, colors='k', linewidths=0.8)
        ax.clabel(cs, fmt=fmt, fontsize=7.5)
        ax.plot([0], [0], 'v', color='k', ms=8, clip_on=False)
        ax.set_xlim(0, zoom * X_max)
        ax.set_ylim(0, zoom * Y_max)
        ax.set_aspect('equal')
        ax.set_xlabel('x, м')
        ax.text(0.95, 0.95, f'({letter})', transform=ax.transAxes, va='top', ha='right', **PANEL)
    axes[0].set_ylabel('y, м')
    fig.tight_layout()
    _save(fig, 'aspo_f5')


FIELD_CURVES = (('t70', '1'), ('t40', '2'), ('base', '3'), ('t5', '4'))


def fig_field_time(th, num):
    v = th['variants']
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.7))
    for (name, label), style, marker in zip(FIELD_CURVES, STYLES, MARKERS):
        if name not in v:
            continue
        s = v[name]['series']
        t = np.array(s['years'])
        axes[0].plot(t, np.array(s['q_inj']) * 4.0, ls=style, color='k', lw=1.3, label=label)
        axes[1].plot(t, np.array(s['rf']), ls=style, color='k', lw=1.3, label=label)
    axes[0].set_ylabel('Q, м³/сут')
    axes[1].set_ylabel('КИН')
    for ax, letter in zip(axes, 'аб'):
        ax.set_xlabel('t, годы')
        _panel(ax, letter)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc='center right', fontsize=8, handlelength=2.6)
    fig.tight_layout()
    _save(fig, 'aspo_f6')


def field_numbers(th, num):
    from paraphin.constants import (init_T, h, K_f, K_w, K_o, c_w, ro_w, c_o, ro_o, c_f, ro_f, init_m, init_S,
                                    latent_heat, P_ref_wax)
    import paraphin.oil_composition as oc
    v = th['variants']
    base = v['base']
    rows = []
    order = ('base', 't70', 't40', 't5', 'hl0', 'hl1', 'nolatent', 'equil', 'nopress', 'ltne', 'noret')
    for name in order:
        if name not in v:
            continue
        e = v[name]
        d_rf = '—' if name == 'base' else _delta(e['rf'] - base['rf'])
        rows.append((e['label'], f"{e['rf']:.3f}", d_rf, f"{e['injected'] / 1e3:.1f}", _nz(e['skin_inj'], 1),
                     f"{e['wax_dep_t']:.0f}", _nz(e['retained_t'], 1).replace('-', '−'), f"{e['T_mean']:.1f}"))
    num['T2'] = _table(['Вариант', 'КИН', 'ΔКИН', 'Закачано, тыс. м³', 'Скин-фактор нагнетательной скважины',
                        'Парафин в осадке, т', 'Прирост удержания, т', 'Средняя температура, °C'], rows)
    for name, e in v.items():
        tag = name.upper()
        num[f'F_{tag}_RF'] = f"{e['rf']:.3f}"
        num[f'F_{tag}_DRF'] = f"{e['rf'] - base['rf']:+.4f}".replace('-', '−')
        num[f'F_{tag}_DRF_ABS'] = f"{abs(e['rf'] - base['rf']):.3f}"
        num[f'F_{tag}_INJ'] = f"{e['injected'] / 1e3:.1f}"
        num[f'F_{tag}_SKIN'] = f"{e['skin_inj']:.1f}"
        num[f'F_{tag}_KINJ'] = f"{e['k_inj']:.3f}"
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
    if 't70' in v:
        num['F_INJ_LOSS'] = f"{100 * (1 - base['injected'] / v['t70']['injected']):.0f}"
    num['F_FRONT_W'] = f"{base['front_water_1y']:.0f}"
    if 'noret' in v and 't70' in v:
        num['F_COOL_LOSS'] = f"{v['t70']['rf'] - base['rf']:.3f}"
        num['F_COOL_LOSS_NORET'] = f"{v['t70']['rf'] - v['noret']['rf']:.3f}"
        num['F_RET_SHARE'] = f"{100 * (v['noret']['rf'] - base['rf']) / (v['t70']['rf'] - base['rf']):.0f}"
        num['F_NORET_WAXRATIO'] = f"{v['noret']['wax_dep_t'] / base['wax_dep_t']:.1f}"
    if 'hl0' in v:
        num['F_HL0_WAXRATIO'] = f"{v['hl0']['wax_dep_t'] / base['wax_dep_t']:.1f}"
    if 'hl1' in v:
        num['F_HL1_WAXPCT'] = f"{100 * (v['hl1']['wax_dep_t'] / base['wax_dep_t'] - 1):.0f}"
    # Варианты, совпавшие с базовым: наибольшее относительное расхождение показателей
    for name in ('ltne', 'equil'):
        if name in v:
            rel = max(abs(v[name][key] - base[key]) / max(abs(base[key]), 1e-30)
                      for key in ('rf', 'T_mean', 'wax_dep_t', 'injected', 'k_inj'))
            num[f'F_{name.upper()}_REL'] = _sci(rel, 0) if rel > 0 else '0'
    num['F_WAT0'] = f"{base['wat0'][0]:.1f}–{base['wat0'][1]:.1f}"
    num['F_PB_YEARS'] = f"{base['years_below_pb']:.1f}"
    num['F_QDROP_DAYS'] = f"{365.0 * base['years_q_drop']:.0f}"
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
    w0 = oc.WAX_W0.sum()
    live = w0 - oc.sle_split_np(oc.WAX_W0, oc.WAX_M, oc.WAX_TM_K, oc.WAX_DH, oc.WAX_DV, t_inj,
                                dp=12e6 - P_ref_wax, n_g=oc.gas_moles(12e6)).sum()
    ste = rc_eff * (init_T - t_inj) / (init_m * (1 - init_S) * ro_o * live * latent_heat)
    num['F_STE'] = f'{ste:.0f}'
    num['F_WPREC'] = f'{100 * live:.1f}'
    num['F_WAX0'] = f'{100 * w0:.1f}'
    # Удержание: рост константы Ленгмюра при охлаждении до температуры закачки и начальное повреждение D(sigma_0)
    from paraphin import surf_0
    from paraphin.constants import R, ro_asph_dep
    from paraphin.oil_composition import initial_components, IA_D, I_R
    kin = th['kin']
    t_ref = kin.get('ADS_T_REF', 70.0) + 273.15
    k_l = lambda t: kin['ADS_K'] * math.exp(-kin['ADS_DH'] / R * (1.0 / (t + 273.15) - 1.0 / t_ref))
    num['F_KRATIO'] = _sci(k_l(t_inj) / k_l(init_T), 0)
    wc = initial_components()
    g_max = kin['ADS_GMAX'] * surf_0
    kl0 = k_l(init_T)
    g0 = sum(g_max * mult * kl0 * c / (1.0 + kl0 * c) for c, mult in ((wc[IA_D], 1.0), (wc[I_R], kin.get('ADS_RESIN', 0.5))))
    rel0 = kin['PERM_BETA'] * g0 / ro_asph_dep / (kin['PERM_SMAX'] * init_m)
    num['F_SIG0'] = f'{rel0:.2f}'
    num['F_D0'] = f"{max(1.0 - rel0, 0.0) ** kin['PERM_GAMMA']:.2f}"
    # Тепловое неравновесие: время релаксации температур флюида и зерна при Nu = 2 (нижняя граница Вакао-Кагеи)
    from paraphin.constants import ltne_dg
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


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    num = {}
    sr, li, sd, pm, he = (_load(n) for n in ('sutton_roberts', 'li2024', 'sandyga2020', 'pore_models', 'he2020'))
    fig_sutton_roberts(sr, num)
    fig_li(li, num)
    fig_sandyga(sd, num)
    fig_he(pm, he, num)
    table_experiments(num)
    th_path = RESULTS / 'thermal.json'
    if th_path.is_file():
        th = json.loads(th_path.read_text(encoding='utf-8'))
        if 'maps' in th:
            fig_field_maps(th, num)
        fig_field_time(th, num)
        field_numbers(th, num)
    else:
        print('нет results/thermal.json - сначала python твт_статья_АСПО/run_thermal.py')
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'numbers.json').write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding='utf-8')
    lines = ['# Числа статьи', '', 'Создается `python твт_статья_АСПО/make_article.py`, руками не править.', '']
    for key, value in num.items():
        lines += [f'**{key}**', '', value, ''] if '\n' in value else [f'- `{key}`: {value}']
    (HERE / 'metrics.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'{len(num)} чисел -> results/numbers.json, metrics.md; рисунки -> figures/')


if __name__ == '__main__':
    main()
