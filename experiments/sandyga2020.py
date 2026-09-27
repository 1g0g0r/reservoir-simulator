"""Охлаждение керна с прокачкой раствора парафина в керосине (Sandyga et al., 2020).

    python experiments/sandyga2020.py           # подбор по сетке (прогоны кэшируются) -> results + params.json
    python experiments/sandyga2020.py --quick   # только итоговый набор из params.json
    python experiments/sandyga2020.py --plot    # только рисунок

Опыт (`data/sandyga2020.json`): 20 % парафина C20-C40 в керосине, песчаник 3 x 5 см, пористость 9 %, 0.5 см^3/мин,
весь стенд охлаждается от 40 C со скоростью 1 C/ч. Градиент давления плавно растет до 34 C и затем за 1.2 C - в 44
раза; томография после опыта: пористость 9 -> 2.1 %, все классы пор потеряли 76-87 % объема. Прежняя модель с WAT
пористой среды (33.8 C) угадывает начало роста, но дает лишь 4.1 раза и пористость 0.96: захват кристаллов
сортирует поры по размеру (узкие затыкаются, широкие почти не тронуты), а объем кристаллов мал.

Механизмы кинетики (`docs/кинетика_осаждения.md`), которые меняют именно это:
  - кристаллизация на стенках пор (`wax_kinetics`, разд. 13.2): пересыщение раствора при охлаждении уходит на
    стенки всех каналов одинаково - как на томографии;
  - гель-отложение (`deposit_aging`, разд. 13.9): отложение - гель с долей парафина C0, проводящие каналы
    сужаются по объему геля; проводящая пористость и есть то, что видит томография;
  - гель в поре со статическим порогом 2 % (Létoffé et al. 1995).
Подбор: k_wall, C0 и k_cryst по росту градиента; пористость после опыта и потери по классам пор - проверка.

Шумовой порог: у кривой почти скачок (в 3.5 раза за 0.3 C), и ошибка оцифровки температуры 0.1 C дает там
ошибку lg(grad) 0.2; вместе с ошибкой 0.3 МПа/м по градиенту это и есть нижняя граница СКО lg (`noise_log`).
"""
import importlib.util
import itertools
import sys

import numpy as np
from scipy.optimize import brentq, least_squares

from common import (load, core_constants, run_many, mode_from_argv, load_params, save_params, save_results,
                    load_results, FIGURES, ROOT, DARCY)

DATA = load('sandyga2020')
SOL, CORE = DATA['solution'], DATA['core']
SINGLE = {'wax_components': 'True', 'wax_characterization': "'single'"}
GEL2 = dict(SINGLE, gelation='True', wax_viscosity='1', gel_time='300.0', gel_phi='0.02')
KINETICS = dict(GEL2, wax_kinetics='True', deposit_aging='True')
GRID = list(itertools.product((1e-4, 1e-3, 1e-2), (0.03, 0.1, 0.3, 1.0), (1e-3, 1e-2)))  # k_wall, C0, k_cryst
POINTS = (35.0, 34.0, 33.5, 33.0, 32.8)  # температуры критерия «в пределах 1.25 раза»
T_ERR, G_ERR = 0.1, 0.3                 # точность оцифровки: C и МПа/м


def _cf():
    spec = importlib.util.spec_from_file_location('core_flood', ROOT / 'experiments' / 'исходная_модель' / 'core_flood.py')
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    return cf


def kosugi(r, r_m, sigma):
    return np.exp(-0.5 * (np.log(r / r_m) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma * r)


def bins_volume(r, fi):
    """Объем r^2*fi в классах томографии (диаметр d +- 5 мкм)."""
    r, fi = np.asarray(r), np.asarray(fi)
    dr = r[1] - r[0]
    return np.array([np.sum((r ** 2 * fi)[(r >= d / 2 * 1e-6 - 2.5e-6 + 1e-12) & (r < d / 2 * 1e-6 + 2.5e-6)]) * dr
                     for d in DATA['pores']['d_um']])


def pore_fit():
    """Kosugi по томографии до опыта."""
    r = np.linspace(0.05e-6, 60e-6, 4000)
    before = np.array(DATA['pores']['before']) / np.sum(DATA['pores']['before'])

    def shares(r_m, sigma):
        b = bins_volume(r, kosugi(r, r_m, sigma))
        return b / b.sum()

    res = least_squares(lambda x: shares(x[0] * 1e-6, x[1]) - before, [12.0, 0.4], bounds=([3, 0.1], [40, 1.5]))
    return res.x[0] * 1e-6, res.x[1]


def core_k0(r_m, sigma):
    """k0 пучка с извилистостью керна Berea, [м^2]."""
    r = np.linspace(0.0, 40e-6 * r_m / 12e-6, 31)[1:]
    fi = kosugi(r, r_m, sigma)
    return CORE['porosity'] * np.sum(r ** 4 * fi) / (8.0 * DATA['berea_tortuosity'] ** 2 * np.sum(r ** 2 * fi))


def exp_case():
    cf = _cf()
    r_m, sigma = pore_fit()
    tm = brentq(lambda t: float(cf.w_saturated(SOL['wax'], SOL['WAT_core'], SOL['MW_wax'], SOL['M_kerosene'], t,
                                               SOL['dH'])) - SOL['wax'] + 1e-9, SOL['WAT_core'] + 0.1, 200.0)
    area = np.pi * CORE['diameter'] ** 2 / 4.0
    pv0 = CORE['length'] * area * CORE['porosity']
    exp = dict(length=CORE['length'], side=float(np.sqrt(area)), porosity=CORE['porosity'],
               k0=float(core_k0(r_m, sigma) / DARCY), T=CORE['T_start'], T_end=CORE['T_end'], cooling=CORE['cooling_C_per_s'],
               q=CORE['q'], P_out=1900 * 6894.757, plugged=1e-3,
               pv_end=CORE['q'] * (CORE['T_start'] - CORE['T_end']) / CORE['cooling_C_per_s'] / pv0,
               w=SOL['wax'], MW=SOL['MW_wax'], M_o=SOL['M_kerosene'], dH=SOL['dH'], ro_o=SOL['ro_solution'],
               ro_p=SOL['ro_wax'], mu=SOL['mu_40'], Tm=float(tm))
    extra = {'r_m': repr(float(r_m)), 'sigma_r': repr(float(sigma)), 'r_max': repr(float(40e-6 * r_m / 12e-6))}
    return exp, extra


def curve(res):
    t, k = np.array(res['T_hist']), np.array(res['k'])
    order = np.argsort(t, kind='stable')
    return t[order], 1.0 / np.maximum(k[order], 1e-12)


def noise_log():
    """Нижняя граница СКО lg(grad/grad0) от точности оцифровки по обеим осям."""
    t, g = np.array(DATA['gradient']['points']).T
    slope = np.gradient(np.log10(g), t)
    s = np.sqrt((G_ERR / (g * np.log(10))) ** 2 + (T_ERR * slope) ** 2)
    return float(np.sqrt(np.mean(s ** 2)))


def pore_loss(res):
    """Доля объема, потерянная классами пор томографии: модель (проводящие каналы у входа, в середине и у выхода
    керна - среднее) и опыт."""
    r, fi0 = res['r'], res['fi0']
    b0 = bins_volume(r, fi0)
    model = np.mean([1.0 - bins_volume(r, f) / np.maximum(b0, 1e-300) for f in res['fi_end']], axis=0)
    exp = 1.0 - np.array(DATA['pores']['after']) / np.array(DATA['pores']['before'])
    return model.tolist(), exp.tolist()


def metrics(res):
    t_exp, g_exp = np.array(DATA['gradient']['points']).T
    ratio_exp = g_exp / g_exp[0]
    t, g = curve(res)
    model = np.interp(t_exp, t, g)
    at = [float(np.interp(x, t, g)) for x in POINTS]
    exp_at = [float(np.interp(-x, -t_exp, ratio_exp)) for x in POINTS]
    loss_model, loss_exp = pore_loss(res)
    return dict(rms_log=float(np.sqrt(np.mean((np.log10(model) - np.log10(ratio_exp)) ** 2))),
                at=at, exp_at=exp_at, max_factor=float(max(max(a / e, e / a) for a, e in zip(at, exp_at))),
                m_conductive=float(np.mean(res['m_conductive_profile'])), m_total=float(np.mean(res['m_profile'])),
                pore_loss=loss_model, pore_loss_exp=loss_exp)


def kin(k_wall, c0, k_cr):
    return {'K_WALL': k_wall, 'AGE_C0': c0, 'K_CRYST': k_cr, 'AGE_RATE': 0.0}


def run(mode: str = 'full') -> dict:
    if mode == 'plot':
        out = load_results('sandyga2020')
        plot(out)
        return out
    exp, extra = exp_case()
    base = dict(core_constants(exp, dt=2.0), **extra)
    case = {'exp': exp, 'mode': 'ramp'}
    grid = GRID if mode == 'full' else [tuple(load_params('sandyga2020')['best'])]
    jobs = [('exp_sd_legacy', base, dict(case, kin={})), ('exp_sd_gel2', dict(base, **GEL2), dict(case, kin={}))]
    jobs += [('exp_sd_kin', dict(base, **KINETICS), dict(case, kin=kin(*p))) for p in grid]
    res = run_many(jobs)
    out = {'legacy': dict(result=res[0], **metrics(res[0])), 'gel2': dict(result=res[1], **metrics(res[1])),
           'kinetics': [dict(k_wall=p[0], c0=p[1], k_cryst=p[2], result=r, **metrics(r)) for p, r in zip(grid, res[2:])],
           'noise_log': noise_log(), 'porosity_exp': CORE['porosity_after'] / CORE['porosity']}
    best = min(out['kinetics'], key=lambda e: e['rms_log'])
    out['best'] = {k: v for k, v in best.items() if k != 'result'}
    if mode == 'full':
        save_params('sandyga2020', {'best': [best['k_wall'], best['c0'], best['k_cryst']],
                                    'kin': 'K_WALL [1/с], AGE_C0, K_CRYST [1/с]; AGE_RATE = 0'})
    print(f'шумовой порог СКО lg {out["noise_log"]:.3f}; пористость после опыта {out["porosity_exp"]:.3f}', flush=True)
    for name, e in (('прежняя', out['legacy']), ('гель 2 %', out['gel2']), ('кинетика, лучший', best)):
        print(f'{name}: СКО lg {e["rms_log"]:.3f}, grad/grad0 {[round(x, 1) for x in e["at"]]} '
              f'(опыт {[round(x, 1) for x in e["exp_at"]]}), проводящая пористость {e["m_conductive"]:.2f}, '
              f'полная {e["m_total"]:.2f}', flush=True)
    save_results('sandyga2020', out)
    plot(out)
    return out


def summary(out) -> list:
    """Строки сводной таблицы: (вариант, СКО lg, наибольшее расхождение в 5 точках, раз, проводящая пористость)."""
    rows = [('шумовой порог опыта', out['noise_log'], None, out['porosity_exp'])]
    for name, e in (('прежняя модель', out['legacy']), ('гель, порог 2 %', out['gel2']),
                    ('кристаллизация на стенках + гель-отложение', out['best'])):
        rows.append((name, e['rms_log'], e['max_factor'], e['m_conductive']))
    return rows


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    best = min(out['kinetics'], key=lambda e: e['rms_log'])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw={'width_ratios': [1.35, 1]})
    t_exp, g_exp = np.array(DATA['gradient']['points']).T
    ax.plot(t_exp, g_exp / g_exp[0], 'o', mfc='white', mec='k', ms=4.5, label='опыт')
    for e, style, label in ((out['legacy'], '-', 'прежняя модель'), (out['gel2'], '--', 'гель, порог 2 %'),
                            (best, '-.', 'кристаллизация на стенках + гель-отложение')):
        t, g = curve(e['result'])
        ax.plot(t, g, ls=style, label=f'{label}: СКО lg {e["rms_log"]:.2f}')
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('T, °C')
    ax.set_ylabel('∇p / ∇p(40 °C)')
    ax.legend(fontsize=6.8, handlelength=3.0)
    d = np.array(DATA['pores']['d_um'])
    w = 2.2
    ax2.bar(d - w, best['pore_loss_exp'], width=2 * w, color='0.75', edgecolor='k', label='томография')
    ax2.bar(d + w, best['pore_loss'], width=2 * w, color='#2a78d6', label='модель (проводящие каналы)')
    ax2.plot(d, out['legacy']['pore_loss'], 'x', color='k', label='прежняя модель')
    ax2.set_xlabel('диаметр пор, мкм')
    ax2.set_ylabel('потерянная доля объема класса')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=6.8)
    fig.tight_layout()
    fig.savefig(FIGURES / 'sandyga2020.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run(mode_from_argv())
