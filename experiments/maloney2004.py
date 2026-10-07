"""Закачка холодной воды в керн мела с остаточной нефтью (Maloney, Oesthus, 2004): прогноз без подбора.

    python experiments/maloney2004.py           # прогоны (кэш) -> results/maloney2004.json и рисунок
    python experiments/maloney2004.py --plot    # только рисунок

Опыт (`data/maloney2004.json`): мел 3.8 x 7.6 см, m = 35 %, k = 1.5 мД; заводнение при 66 C до остаточной нефти
(S_w = 0.62), затем закачка воды 0.5 мл/ч при ступенчатом охлаждении до 10 C и еще 3 сут при 10 C. Точка помутнения
нефти 25 C. Относительная проницаемость по воде k_rw растет с 0.05 до 0.10 к 26 C (усадка нефти: S_w 0.62 -> 0.64),
ниже точки помутнения почти не меняется: 0.09 сразу после охлаждения до 10 C и 0.10 через 3 сут.

Все остальные опыты проекта однофазные, а повреждение пласта в статье (`твт_статья_АСПО`) идет у нагнетательной
скважины, то есть в промытой зоне. Этот опыт проверяет, что модель не дает повреждения от парафина остаточной
нефти, которого в опыте нет. Модель - та же, что в расчете пласта: нефть Жетыбая с детальным составом, кинетика
кристаллизации и параметры керна Li et al. (`params.json`). Состав нефти опыта не приведен, поэтому температуры
плавления групп сдвинуты так, чтобы WAT была 25 C, а содержание парафина - вариантами: 24.9 % (как у нефти
Жетыбая, верхняя оценка) и 5 %. Поровое пространство - пучок Косуги с медианным радиусом 0.5 мкм (горла мела
меньше 1 мкм). Усадки нефти в модели нет (фазы несжимаемы), поэтому рост k_rw выше точки помутнения не
воспроизводится; сравнивается изменение ниже нее: k_rw(10 C)/k_rw(26 C) = 0.9-1.0.

Измеряемая величина - проницаемость по воде k_w(T)/k_w(66 C): при S = S_max ОФП воды в модели постоянна, и она
равна отношению абсолютных проницаемостей (`k_harm` прогона).
"""
import sys

import numpy as np
from scipy.optimize import brentq

from common import load, core_constants, run_many, mode_from_argv, save_results, load_results, load_params, FIGURES
import li2024

DATA = load('maloney2004')
CORE = DATA['core']
STAGE_PV = 0.5  # поровых объемов на ступень охлаждения: длительность ступеней в статье не приведена
PV0 = CORE['porosity'] * np.pi / 4.0 * CORE['diameter'] ** 2 * CORE['length']
STAGES = [[66.0, STAGE_PV], [45.0, STAGE_PV], [26.0, STAGE_PV], [10.0, STAGE_PV],
          [10.0, DATA['q'] * DATA['days_at_10'] * 86400.0 / PV0]]
WAX = (0.2493, 0.05)  # содержание парафина вариантов: как у нефти Жетыбая и 5 %
R_M = 0.5e-6          # медианный радиус пучка, [м]


def tm_shift(total: float) -> float:
    """Сдвиг температур плавления групп Вона, при котором WAT нефти равна точке помутнения опыта."""
    import calibrate
    from paraphin import oil_composition as oc
    from paraphin.constants import scn_slope, wax_alpha_eff
    return brentq(lambda sh: float(calibrate.wat(oc.group_properties(scn_slope, wax_alpha_eff, sh, total=total)))
                  - DATA['oil']['cloud_point'], -40.0, 120.0)


def job(total: float):
    from paraphin.constants import MW, M_o, Tm, alpha, ro_p
    exp = dict(length=CORE['length'], side=float(np.sqrt(np.pi / 4.0) * CORE['diameter']), porosity=CORE['porosity'],
               k0=CORE['k_brine_mD'] * 1e-3, T=STAGES[0][0], P_out=101325.0, q=DATA['q'],
               pv_end=sum(p for _, p in STAGES), stages=STAGES, w=total, MW=MW, M_o=M_o, Tm=Tm, dH=alpha,
               ro_o=850.0, ro_p=ro_p, mu=3e-3, water=True)
    c = core_constants(exp)
    c.update(li2024.SCN, wax_kinetics='True', gel_time='3600.0',
             S_min=repr(DATA['S_wr']), S_max=repr(DATA['S_w_flood']), init_S='S_max',
             init_Wp=repr(total), init_Wps='0.0', wax_Tm_shift=repr(round(tm_shift(total), 3)),
             mu_o_ref='3e-3', T_mu_ref='66.0', r_m=repr(R_M), r_max=repr(R_M * 40.0 / 12.0))
    params = load_params('li2024')
    kin = li2024.cold_kin(load_params('li2024_cold')['x'], params[params['best']]['kin'])
    return f'exp_maloney_{total:g}', c, {'mode': 'stages', 'exp': exp, 'kin': kin}


def stage_end(res, n):
    """k/k0 (абсолютная, она же по воде) в конце ступени n."""
    stage, k = np.array(res['stage']), np.array(res['k_harm'])
    return float(k[stage == n][-1])


def run(mode: str = 'full') -> dict:
    if mode == 'plot':
        out = load_results('maloney2004')
        plot(out)
        return out
    res = run_many([job(w) for w in WAX])
    out = {'exp': {'k10_26': [DATA['k_rw'][2][1] / DATA['k_rw'][1][1], DATA['k_rw'][3][1] / DATA['k_rw'][1][1]]},
           'stages': STAGES, 'variants': {}}
    for w, r in zip(WAX, res):
        ends = [stage_end(r, n) for n in range(len(STAGES))]
        e = dict(wax=w, shift=tm_shift(w), k_end=ends, k10_26=[ends[3] / ends[2], ends[4] / ends[2]],
                 k10_66=ends[4] / ends[0], m_end=float(np.mean(r['m_profile'])), dep_wax=r.get('dep_wax'),
                 plugged=r['plugged'], result=r)
        out['variants'][f'{w:g}'] = e
        print(f'парафин {w * 100:.1f} %: k_w/k_w(66 C) по ступеням {[round(x, 3) for x in ends]}, '
              f'k(10 C)/k(26 C) = {e["k10_26"][0]:.3f} и через 3 сут {e["k10_26"][1]:.3f} (опыт 0.9 и 1.0), '
              f'пористость {e["m_end"]:.4f} m0', flush=True)
    save_results('maloney2004', out)
    plot(out)
    return out


def summary(out) -> list:
    """Строки сводной таблицы: (вариант, k_w(10 C)/k_w(26 C) сразу и через 3 сут, k_w(10 C)/k_w(66 C))."""
    rows = [('опыт', *out['exp']['k10_26'], None)]
    for e in out['variants'].values():
        rows.append((f'парафин {e["wax"] * 100:.1f} %, WAT 25 °C', *e['k10_26'], e['k10_66']))
    return rows


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for (name, e), style in zip(out['variants'].items(), ('-', '--')):
        r = e['result']
        ax.plot(r['pv'], r['k_harm'], ls=style, color='k', label=f'модель, парафин {e["wax"] * 100:.0f} %')
    ax.set_xlabel('V, поровых объемов')
    ax.set_ylabel('k_w / k_w(66 °C)')
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / 'maloney2004.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run(mode_from_argv())
