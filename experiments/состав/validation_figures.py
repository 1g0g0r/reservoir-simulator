"""Рисунки и таблицы к сравнению с опытами (`validate.py`): figures/val*.png, validation_tables.json.

Стиль и палитра - как у рисунков описания модели (`make_model_figures.py`): опыт - черные маркеры,
прежняя модель - первый цвет, варианты новой - следующие по порядку; у вариантов свой тип линии.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'docs'))  # стиль рисунков - общий с описанием модели
from make_model_figures import SERIES, STYLES, INK, INK2, plt  # noqa: E402
from validate import cf, li, sd  # noqa: E402

TABLES = HERE / 'validation_tables.json'
FIG = HERE / 'figures'
LINE_STYLES = STYLES + [(0, (6, 1.5, 1, 1.5, 1, 1.5)), (0, (1, 1))]


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ', name)


def _mult_label(m):
    return f'{m:g}'


def _trim(run):
    """Точки кривой до закупорки: драйвер дописывает (PV_END, 0) - на графике ее нет, закупорка - крест."""
    pv, k = np.array(run['pv']), np.array(run['k'])
    if run['plugged'] is not None:
        keep = pv <= run['plugged'] + 1e-12
        return pv[keep], k[keep], True
    return pv, k, False


def _plot_run(ax, run, color, ls, label):
    pv, k, plugged = _trim(run)
    ax.plot(pv, k, color=color, ls=ls, label=label)
    if plugged:
        ax.plot(pv[-1], k[-1], 'x', color=color, ms=7, mew=1.8)


def _exp(ax, points, label='опыт'):
    pv, k = np.array(points).T
    ax.plot(pv, k, 'o', mfc='white', mec=INK, ms=4.5, mew=1.1, label=label, zorder=5)


# --- 1. Sutton & Roberts ---------------------------------------------------------------------------------

def fig_sutton_roberts(data):
    runs, mult = data['runs'], data['mult']
    mults = sorted({float(k.rsplit('_', 1)[1]) for k in runs if k.startswith('sr1_gel_')}, reverse=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    _exp(ax1, cf.EXPERIMENTS[1]['exp'])
    _plot_run(ax1, runs['sr1_legacy'], SERIES[0], '-', f'прежняя модель (калибровка), СКО {runs["sr1_legacy"]["rms"]:.3f}')
    for n, m in enumerate(mults):
        run = runs[f'sr1_gel_{m}']
        _plot_run(ax1, run, SERIES[1 + n], LINE_STYLES[1 + n], f'+ гель, множитель {_mult_label(m)}: {run["rms"]:.3f}')
    ax1.set_title('а) опыт 1 (калибровка d_p, L_k и множителя)', fontsize=9)
    ax1.set_xlabel('прокачано, PV')
    ax1.set_ylabel('k / k₀')
    ax1.legend(fontsize=6.5, loc='upper right')

    _exp(ax2, cf.EXPERIMENTS[2]['exp'])
    _plot_run(ax2, runs['sr2_legacy'], SERIES[0], '-', f'прежняя модель, СКО {runs["sr2_legacy"]["rms"]:.3f}')
    gel = runs[f'sr2_gel_{mult}']
    k = 1 + mults.index(mult)
    _plot_run(ax2, gel, SERIES[k], LINE_STYLES[k], f'+ гель, множитель {_mult_label(mult)}: {gel["rms"]:.3f}')
    for name, ls, label in (('ring', ':', 'Ring et al. (1994)'), ('wang_civan', '--', 'Wang & Civan (2005)')):
        pv, kk = np.array(cf.EXPERIMENTS[2][name]).T
        ax2.plot(pv, kk, color=INK2, ls=ls, lw=1.2, label=label)
    ax2.set_title('б) опыт 2 (прогноз)', fontsize=9)
    ax2.set_xlabel('прокачано, PV')
    ax2.legend(fontsize=6.5, loc='upper right')
    for ax in (ax1, ax2):
        ax.set_xlim(0, 5.1)
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, 'val01_sutton_roberts')


# --- 2. Li et al., 25 C ------------------------------------------------------------------------------------

def _li_variants(data):
    runs, mult = data['runs'], data['mult']
    out = [('li25_legacy', 'прежняя модель'), ('li25_comp', '4 группы, вязкость P–R')]
    for m in sorted({1.0, mult}, reverse=True):
        out.append((f'li25_gel_{m}', f'+ гель, множитель {_mult_label(m)}'))
    return [(key, label) for key, label in out if key in runs]


def fig_li(data):
    runs = data['runs']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.4))
    _exp(ax1, li.K_PV[25], 'опыт 25 °C')
    for n, (key, label) in enumerate(_li_variants(data)):
        _plot_run(ax1, runs[key], SERIES[n], LINE_STYLES[n], f'{label}: СКО {runs[key]["rms"]:.3f}')
    ax1.set_xlim(0, 5.1)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('прокачано, PV')
    ax1.set_ylabel('k / k₀')
    ax1.set_title('а) керн 2, 18.8 мД, 25 °C', fontsize=9)
    ax1.legend(fontsize=6.5, loc='upper right')

    # Что дает спад: проницаемость по осадку (k_harm, без геля) и кажущаяся по перепаду (с гелем)
    gel_keys = [(key, label) for key, label in _li_variants(data) if 'gel' in key]
    for n, (key, label) in enumerate(gel_keys):
        run = runs[key]
        color = SERIES[2 + n]
        pv, k, _ = _trim(run)
        ax2.plot(pv, k, color=color, ls='-', label=f'{label}: по перепаду')
        kh = np.array(run['k_harm'])[:len(pv)]
        ax2.plot(pv, kh, color=color, ls=':', label=f'{label}: только осадок')
    if 'li25_gel_dep0_1.0' in runs:  # гель только из взвеси: по мере ее осаждения гель «тает»
        _plot_run(ax2, runs['li25_gel_dep0_1.0'], SERIES[4], '--', 'гель только из взвеси, множитель 1')
    ax2.axhline(1.0, color=INK2, lw=0.8)
    ax2.set_xlim(0, 5.1)
    ax2.set_ylim(0, 1.9)  # место под легенду над кривыми
    ax2.set_xlabel('прокачано, PV')
    ax2.set_ylabel('k / k₀')
    ax2.set_title('б) вклад осадка и геля', fontsize=9)
    ax2.legend(fontsize=6.5, loc='upper right')
    fig.tight_layout()
    _save(fig, 'val02_li2024')


# --- 3. Sandyga et al. ---------------------------------------------------------------------------------------

def _sd_curve(run):
    t, k = np.array(run['T_hist']), np.array(run['k'])
    if run['plugged'] is not None:
        t, k = t[:-1], k[:-1]
    return t, 1.0 / k


def fig_sandyga(data):
    runs, mult = data['runs'], data['mult']
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    t_exp, g_exp = np.array(sd.GRADIENT).T
    ax.plot(t_exp, g_exp / g_exp[0], 'o', mfc='white', mec=INK, ms=4.5, mew=1.1, label='опыт (рис. 4)', zorder=5)
    keys = [('sd_legacy', 'прежняя модель, WAT 33.8 °C')]
    for m in sorted({1.0, mult}, reverse=True):
        keys.append((f'sd_gel_{m}', f'+ гель, множитель {_mult_label(m)}'))
    keys.append(('sd_gelphi_0.02_1.0', '+ гель с порогом 2 % (статический)'))
    for n, (key, label) in enumerate(keys):
        if key not in runs:
            continue
        t, g = _sd_curve(runs[key])
        ax.plot(t, g, color=SERIES[n], ls=LINE_STYLES[n], label=label)
        if runs[key]['plugged'] is not None:
            ax.plot(t[-1], g[-1], 'x', color=SERIES[n], ms=7, mew=1.8)
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('T, °C')
    ax.set_ylabel('∇p / ∇p(40 °C)')
    ax.set_title('Охлаждение керна 1 °C/ч, 0.5 см³/мин', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    _save(fig, 'val03_sandyga2020')


# --- 4. Асфальтены в опыте Li et al. -------------------------------------------------------------------------

def fig_asphaltenes(data):
    a = data['asphaltenes']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.2), gridspec_kw={'width_ratios': (1.4, 1)})
    p = np.array(a['P'])
    for n, t in enumerate(a['T']):
        ax1.plot(p, a['live'][str(t)], color=SERIES[n], ls='-', label=f'{t:.0f} °C, пластовая (газ)')
        ax1.plot(p, a['dead'][str(t)], color=SERIES[n], ls=':', lw=1.4)
    ax1.plot([], [], color=INK2, ls=':', lw=1.4, label='дегазированная (опыт) — все T')
    ax1.set_xlabel('P, МПа')
    ax1.set_ylabel('выпало асфальтенов, % масс.')
    ax1.set_title('а) равновесие (Хиршберг), параметры поля', fontsize=9)
    ax1.legend(fontsize=6.5, loc='upper right')

    temps = [90, 65, 45]
    x = np.arange(len(temps))
    plateau = [a['plateau_exp'][str(t)] for t in temps]
    ax2.bar(x - 0.18, plateau, 0.34, color=SERIES[0], label='опыт: плато k/k₀')
    ax2.bar(x + 0.18, [1.0] * len(temps), 0.34, color=SERIES[1], label='модель')
    for xi, v in zip(x, plateau):
        ax2.text(xi - 0.18, v + 0.02, f'{v:.2f}', ha='center', fontsize=7.5, color=INK)
    ax2.set_xticks(x, [f'{t} °C' for t in temps])
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel('k / k₀ к 5 PV')
    ax2.set_title('б) керн 2 выше WAT', fontsize=9)
    ax2.legend(fontsize=7, loc='upper left')
    ax2.grid(axis='x', visible=False)
    fig.tight_layout()
    _save(fig, 'val04_asphaltenes')


def figures(data):
    print('Рисунки:')
    fig_sutton_roberts(data)
    fig_li(data)
    fig_sandyga(data)
    fig_asphaltenes(data)


# --- Таблицы -------------------------------------------------------------------------------------------------

def _at(run, points=(0.25, 1.0, 3.0, 5.0)):
    pv, k, _ = _trim(run)
    return ' / '.join(f'{np.interp(x, pv, k) if x <= pv[-1] else 0.0:.2f}' for x in points)


def _plug(run):
    return f'{run["plugged"]:.2f}' if run['plugged'] is not None else '—'


def tables(data):
    runs, mult = data['runs'], data['mult']
    t = {}

    def exp_at(points):
        pv, k = np.array(points).T
        return ' / '.join(f'{np.interp(x, pv, k):.2f}' for x in (0.25, 1.0, 3.0, 5.0))

    rows = [f'| опыт 1 | — | {exp_at(cf.EXPERIMENTS[1]["exp"])} | — | — | — |']
    mults = sorted({float(k.rsplit('_', 1)[1]) for k in runs if k.startswith('sr1_gel_')}, reverse=True)
    sr_rows = [('sr1_legacy', 'прежняя модель')] + [(f'sr1_gel_{m}', f'+ гель, множитель {_mult_label(m)}')
                                                     for m in mults]
    if 'sr1_gel_dep0_1.0' in runs:
        sr_rows.append(('sr1_gel_dep0_1.0', 'гель только из взвеси, множитель 1'))
    for key, label in sr_rows:
        run = runs[key]
        rows.append(f'| {label} | {run["rms"]:.3f} | {_at(run)} | {_plug(run)} | '
                    f'{np.mean(run["m_profile"]):.2f} | {min(run["phi_profile"]):.2f} |')
    rows.append(f'| опыт 2 | — | {exp_at(cf.EXPERIMENTS[2]["exp"])} | — | — | — |')
    for key, label in (('sr2_legacy', 'прежняя модель'), (f'sr2_gel_{mult}', f'+ гель, множитель {_mult_label(mult)}')):
        run = runs[key]
        rows.append(f'| {label} | {run["rms"]:.3f} | {_at(run)} | {_plug(run)} | '
                    f'{np.mean(run["m_profile"]):.2f} | {min(run["phi_profile"]):.2f} |')
    t['T_SR'] = ('| Вариант | СКО k/k0 | k/k0 при 0.25 / 1 / 3 / 5 PV | закупорка, PV | m/m0 | min Φ в конце |\n'
                 '|---|---|---|---|---|---|\n' + '\n'.join(rows))

    exp25 = li.K_PV[25]
    rows = [f'| опыт | — | — | {exp_at(exp25)} | — | — |']
    li_rows = _li_variants(data) + ([('li25_gel_dep0_1.0', 'гель только из взвеси, множитель 1')]
                                    if 'li25_gel_dep0_1.0' in runs else [])
    for key, label in li_rows:
        run = runs[key]
        ratio = li.rms_ratio(run)
        kh = np.array(run['k_harm'])
        rows.append(f'| {label} | {run["rms"]:.3f} | {ratio:.3f} | {_at(run)} | {kh[-1]:.2f} | '
                    f'{run["gel_start"]:.2f} |')
    t['T_LI'] = ('| Вариант | СКО k/k0 | СКО k(25)/k(90) | k/k0 при 0.25 / 1 / 3 / 5 PV | k/k0 по осадку в конце | '
                 'перепад в начале / без геля |\n|---|---|---|---|---|---|\n' + '\n'.join(rows))

    t_exp, g_exp = np.array(sd.GRADIENT).T
    ratio_exp = g_exp / g_exp[0]
    pts = (35.0, 34.0, 33.5, 33.0, 32.8)
    rows = [f'| опыт | {" / ".join(f"{np.interp(-x, -t_exp, ratio_exp):.1f}" for x in pts)} | ≈ 34.0 | '
            f'{sd.POROSITY_AFTER / sd.CORE_M:.2f} | — |']
    keys = [('sd_legacy', 'прежняя модель, WAT 33.8 °C')] + [(f'sd_gel_{m}', f'+ гель, множитель {_mult_label(m)}')
                                                                for m in sorted({1.0, mult}, reverse=True)]
    keys += [('sd_gelphi_0.02_1.0', '+ гель, порог 2 % (статический), множитель 1'),
             ('sd_gelphi_0.02_dep0_1.0', 'то же, гель только из взвеси')]
    for key, label in keys:
        if key not in runs:
            continue
        tt, g = _sd_curve(runs[key])
        order = np.argsort(tt, kind='stable')
        at = ' / '.join(f'{np.interp(x, tt[order], g[order]):.1f}' for x in pts)
        onset = tt[np.argmax(g > 2.0)] if np.any(g > 2.0) else None
        model_at = np.interp(t_exp, tt[order], g[order])
        err = np.sqrt(np.mean((np.log10(model_at) - np.log10(ratio_exp)) ** 2))
        rows.append(f'| {label} | {at} | {"—" if onset is None else f"{onset:.2f}"} | '
                    f'{np.mean(runs[key]["m_profile"]):.2f} | {err:.2f} |')
    t['T_SD'] = ('| Вариант | ∇p/∇p₀ при 35 / 34 / 33.5 / 33 / 32.8 °C | начало роста (×2), °C | m/m0 | '
                 'СКО lg(∇p/∇p₀) |\n|---|---|---|---|---|\n' + '\n'.join(rows))

    a = data['asphaltenes']
    p = np.array(a['P'])
    rows = []
    for tc in a['T']:
        dead, live = np.array(a['dead'][str(tc)]), np.array(a['live'][str(tc)])
        rows.append(f'| {tc:.0f} | {np.interp(0.1, p, dead):.3f} | {np.interp(13.1, p, dead):.3f} | '
                    f'{live.max():.3f} (при {p[live.argmax()]:.1f} МПа) | '
                    f'{a["plateau_exp"].get(str(int(tc)), "—")} |')
    t['T_ASPH'] = ('| T, °C | выпало, дегазир., 0.1 МПа | то же, 13.1 МПа | пластовая, максимум | '
                   'опыт: плато k/k0 |\n|---|---|---|---|---|\n' + '\n'.join(rows))
    t['MULT'] = _mult_label(mult)
    t['GEL_TIME'] = f'{data["lab_gel_time"]:.0f}'
    TABLES.write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'Таблицы: {TABLES}')
