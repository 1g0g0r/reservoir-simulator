"""Верификация тепловой части для статьи о термических процессах (`статья_твт/article.md`).

    python статья_твт/verify_heat.py      # ~5 мин -> results/verify_heat.json

1. Решение Ловерье. Закачка холодной воды с постоянным расходом в водонасыщенную полосу (Nx = 1) с теплообменом
   через кровлю и подошву. Расчет paraphin сравнивается с точным решением
       (T - T0)/(T_inj - T0) = erfc[ x*sqrt(K_ff*rho_ff*c_ff) / (h*U*sqrt(t - M*x/U)) ],  U = rho_w*c_w*u,
   M = (1 - m)*rho_f*c_f + m*rho_w*c_w (вывод - преобразованием Лапласа, как у Ловерье). Точное решение отвечает
   свертке истории температуры на границе пласта; метод Винсома - Вестервельда (heat_losses = 2) должен к нему
   приближаться, а «ступенчатая» схема (heat_losses = 1) - нет: она отсчитывает прогрев пород от начала расчета.
   Теплопроводность вдоль пласта в решении не учтена; тепловое число Пекле полосы ~10^3, и ее вклад мал.
   Сходимость по сетке - тот же расчет с методом Винсома - Вестервельда на 50, 100 и 200 ячейках.
2. MRST: показатели теста `tests/mrst_tests/test_thermal.py` (закачка воды 20 C в пласт 70 C против geothermal
   MRST 2024a) по готовому эталону, без рисунков.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import erfc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from tests.mrst_tests import common as c  # noqa: E402

PATCH = {**c.PATCH_TWO_PHASE, 'S_max': '1.0', 'init_S': 'S_max', 'geological_reserves': '1.0'}
Q = 300.0 / 86400.0       # расход закачки, [м^3/с]: тепловой фронт проходит ~70 м за год
DAYS = 730
FIELD_DAYS = (365, 730)
CELLS = (50, 100, 200)    # сетки полосы длиной 200 м
SCHEMES = {0: 'без теплообмена', 1: 'ступенчатая схема Ловерье', 2: 'Винсом - Вестервельд'}


def lauwerier(x, t, q_w=None, m=None):
    """Безразмерная температура (T - T0)/(T_inj - T0) по точному решению Ловерье в точках x, [м], в момент t, [с].

    q_w - расход на единицу ширины полосы, [м^2/с] (по умолчанию - полоса расчета), m - пористость."""
    from paraphin.constants import X_min, X_max, h, init_m, ro_w, c_w, ro_f, c_f, K_ff, ro_ff, c_ff
    m = init_m if m is None else m
    u = (Q / (X_max - X_min) if q_w is None else q_w) / h
    U = ro_w * c_w * u
    M = (1.0 - m) * ro_f * c_f + m * ro_w * c_w
    lag = M * np.asarray(x) / U
    arg = np.asarray(x) * np.sqrt(K_ff * ro_ff * c_ff) / (h * U * np.sqrt(np.maximum(t - lag, 1e-30)))
    return np.where(t > lag, erfc(arg), 0.0)


def strip_runs():
    from paraphin.constants import Twater
    jobs = {}
    for scheme in SCHEMES:
        for n in (CELLS if scheme == 2 else (100,)):
            jobs[f'lauwerier_hl{scheme}_n{n}'] = (
                f'tvt_lauwerier_hl{scheme}_n{n}', {'Nx, Ny': f'1, {n}', **PATCH, 'heat_losses': str(scheme)},
                {'days': DAYS, 'field_days': list(FIELD_DAYS), 'inj_q': Q, 'T_inj': Twater})
    return c.paraphin_runs(jobs)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    from paraphin.constants import Y_min, Y_max, init_T, Twater
    from tests.mrst_tests import test_thermal as tt
    out = {'q_m3_day': Q * 86400.0, 'days': list(FIELD_DAYS), 'schemes': SCHEMES, 'strip': {}}
    runs = strip_runs()
    for name, r in runs.items():
        n = r['T'].shape[1]
        x = (np.arange(n) + 0.5) * (Y_max - Y_min) / n
        theta = (r['T'] - 273.15 - init_T) / (Twater - init_T)
        rows = []
        for k, day in enumerate(FIELD_DAYS):
            exact = lauwerier(x, day * 86400.0)
            d = theta[k] - exact
            rows.append(dict(day=day, max=float(np.max(np.abs(d))), rms=float(np.sqrt(np.mean(d * d))),
                             x=x.tolist(), model=theta[k].tolist(), exact=exact.tolist()))
        out['strip'][name] = rows
        print(name, [(e['day'], round(e['max'], 4), round(e['rms'], 4)) for e in rows], flush=True)
    # MRST geothermal: тот же отчет, что строит тест, без рисунков
    ref = c.load_reference(tt.TEST, tt.physics())
    ref.update(c.paraphin_runs({f'paraphin_n{n}_thermal': (
        f'mrst_thermal_n{n}', {'Nx, Ny': f'{n}, {n}', **tt.PATCH},
        {'days': tt.DAYS, 'field_days': tt.FIELD_DAYS, 'T_inj': Twater}) for n in tt.GRIDS}))
    out['mrst'] = {n: c.differences(ref[f'thermal_n{n}'], ref[f'paraphin_n{n}_thermal']) for n in tt.GRIDS}
    # то же с эффектом Джоуля - Томсона (`joule_thomson`): MRST учитывает нагрев воды трением, а модель без флага - нет
    jt = c.paraphin_runs({f'paraphin_n{n}_thermal_jt': (
        f'mrst_thermal_jt_n{n}', {'Nx, Ny': f'{n}, {n}', **tt.PATCH, 'joule_thomson': 'True'},
        {'days': tt.DAYS, 'field_days': tt.FIELD_DAYS, 'T_inj': Twater}) for n in tt.GRIDS})
    out['mrst_jt'] = {n: c.differences(ref[f'thermal_n{n}'], jt[f'paraphin_n{n}_thermal_jt']) for n in tt.GRIDS}
    out['mrst_days'] = list(tt.FIELD_DAYS)
    out['dT_friction'] = tt.dT_friction()
    for key in ('mrst', 'mrst_jt'):
        for n, d in out[key].items():
            print(f'{key} {n}x{n}: L2 приемистости {d["q_inj"]:.4f}, ср. |dT| {np.round(d["dT_mean"], 2).tolist()}, '
                  f'max |dT| {np.round(d["dT_max"], 2).tolist()}, T добывающей {d["T_prod_max"]:.2f}, '
                  f'нагрев трением {out["dT_friction"]:.2f} K', flush=True)
    (HERE / 'results' / 'verify_heat.json').write_text(json.dumps(out, ensure_ascii=False, default=float),
                                                       encoding='utf-8')


if __name__ == '__main__':
    main()
