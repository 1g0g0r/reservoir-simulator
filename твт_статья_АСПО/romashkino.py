"""Оценка охлаждения пласта на Ромашкинском месторождении решением Ловерье (статья A, раздел о поле).

    python твт_статья_АСПО/romashkino.py      # секунды -> results/romashkino.json

Наблюдения - Sandyga et al. 2021 (Perm J. Petrol. Min. Eng.) по данным контрольных скважин: закачка с 1960 г.,
приемистость 700-1800 м^3/сут, толщина пласта ~10 м, пластовая температура 37 C, вода у пласта 7-12 C зимой и
18-25 C летом. Через 6 лет: в 250 м от нагнетательной скважины 23 C, в 390 м от ряда нагнетательных 26-27 C,
в 600 м - охлаждение на 1-2 C при промывке 1-2 поровыми объемами.

Течение от ряда - плоское (линейная задача Ловерье, вода уходит по обе стороны ряда), температура закачки -
среднегодовая. Неизвестен расход на единицу длины ряда q_w; для каждого наблюдения он находится обращением
решения Ловерье при температуре закачки 12.5-18.5 C (средние зимы и лета по краям интервалов). Согласованность
наблюдений - пересечение интервалов q_w; независимая проверка - число промытых поровых объемов в 600 м. Свойства
пород - из constants.py модели (свойства пласта Ромашкинского месторождения в источнике не приведены), K_ff
перебирается: без теплообмена фронт резкий, и три температуры одним расходом не описываются.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verify_heat import lauwerier  # noqa: E402

YEAR = 365.25 * 86400.0
T0 = 37.0
T_INJ = (12.5, 15.5, 18.5)
YEARS = 6.0
# расстояние, [м] -> наблюдаемая температура, [C] (интервал)
OBS = {250: (23.0, 23.0), 390: (26.0, 27.0), 600: (T0 - 2.0, T0 - 1.0)}
Q_WELL = (700.0, 1800.0)   # приемистость скважины, [м^3/сут]
POROSITY = (0.2, 0.3)      # для числа промытых поровых объемов: пористость в источнике не дана
K_SCAN = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)   # теплопроводность кровли и подошвы, [Вт/(м*C)]


def theta(x, q_w):
    return float(lauwerier(np.array([float(x)]), YEARS * YEAR, q_w=q_w)[0])


def q_for(x, th):
    """Расход на единицу длины ряда в одну сторону, [м^2/с], при котором решение Ловерье дает th в x за 6 лет."""
    return brentq(lambda q: theta(x, q) - th, 1e-9, 1e-2)


def analyse(c):
    """Интервалы q_w по наблюдениям, их пересечение и следствия при текущих свойствах constants."""
    out = {'obs': {}}
    for x, (t_lo, t_hi) in OBS.items():
        qs = [q_for(x, (T0 - t) / (T0 - ti)) for t in (t_lo, t_hi) for ti in T_INJ]
        out['obs'][x] = {'q_min': min(qs), 'q_max': max(qs),
                         'q_mid': q_for(x, (T0 - 0.5 * (t_lo + t_hi)) / (T0 - T_INJ[1]))}
    lo = max(o['q_min'] for o in out['obs'].values())
    hi = min(o['q_max'] for o in out['obs'].values())
    out['overlap'] = [lo, hi] if lo <= hi else None
    q_row = 0.5 * (lo + hi) if lo <= hi else out['obs'][390]['q_mid']  # середина общего интервала
    out['q'] = q_row
    out['pv_600'] = {m: q_row * YEARS * YEAR / (m * c.h * 600.0) for m in POROSITY}
    out['spacing'] = {q: q / 86400.0 / (2.0 * q_row) for q in Q_WELL}   # q_w = Q_скв/(2a)
    out['T'] = {x: T0 - (T0 - T_INJ[1]) * theta(x, q_row) for x in OBS}
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    from paraphin import constants as c
    out = {'T_inj': T_INJ, 'years': YEARS, 'h': c.h, 'K_ff': c.K_ff, 'scan': {}}
    out.update(analyse(c))
    k_ff = c.K_ff
    try:
        for k in K_SCAN:
            c.K_ff = k
            out['scan'][k] = analyse(c)
        c.K_ff = 1e-12                       # расход базового K_ff без теплообмена с кровлей и подошвой
        out['no_losses'] = {x: T0 - (T0 - T_INJ[1]) * theta(x, out['q']) for x in OBS}
    finally:
        c.K_ff = k_ff
    day = lambda v: None if v is None else [round(e * 86400, 2) for e in v]  # noqa: E731
    for k, r in out['scan'].items():
        print(f'K_ff {k}: q_w ' + ', '.join(f'{x} м {day([o["q_min"], o["q_max"]])}' for x, o in r['obs'].items())
              + f'; общий {day(r["overlap"])} м^2/сут; поровых объемов в 600 м '
              f'{[round(v, 1) for v in r["pv_600"].values()]}; T {[round(v, 1) for v in r["T"].values()]}')
    print('шаг ряда при базовом K_ff, м:', {q: round(v) for q, v in out['spacing'].items()},
          'T без теплообмена:', {x: round(v, 1) for x, v in out['no_losses'].items()})
    (HERE / 'results' / 'romashkino.json').write_text(json.dumps(out, ensure_ascii=False, default=float),
                                                      encoding='utf-8')


if __name__ == '__main__':
    main()
