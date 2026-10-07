"""Сравнение модели детального состава нефти с опытами: прогоны керна прежней и новой моделью.

    python experiments/состав/validate.py           # прогоны (~40 мин) -> validation.json рядом, затем рисунки и таблицы
    python experiments/состав/validate.py --plot    # только рисунки figures/val*.png и таблицы validation_tables.json

Опыты и данные - те же, что в `experiments/исходная_модель/VALIDATION.md` (оцифровка там же, в `core_flood.py`,
`validation_li2024.py`, `validation_sandyga2020.py`); d_p = 15 мкм и L_k = 0.3 мм - калибровка по опыту 1
Sutton & Roberts (`constants.py`). Здесь к упрощенной модели добавляются механизмы детального состава:
  1. Sutton & Roberts (1974), опыты 1 и 2: гель с осадком в порах. Множитель `gel_tau_mult` - единственный
     свободный параметр геля (предел текучести перенесен из объема в пору), его допустимый диапазон
     определяется по опыту 1, как d_p и L_k; опыт 2 - прогноз.
  2. Li et al. (2024), керн 2 нефти Жетыбая при 25 C: группы парафина (это та самая нефть, по которой
     они подобраны), вязкость Pedersen-Ronningsen и гель - прогноз без подгонки.
  3. Sandyga et al. (2020), охлаждение керна с раствором парафина в керосине: гель - прогноз.
  4. Li et al. (2024), 45-90 C: может ли асфальтеновый блок объяснить повреждение выше WAT.

Каждый вариант считается в копии пакета с поправленными константами (`tests/_patched_copy.py`),
рабочий процесс - `core_worker.py` рядом. Тиксотропное время геля в опыте - 5 мин (`LAB_GEL_TIME`), а не
сутки, как в поле: опыт короче суток, а тиксотропия парафинистой нефти - минуты (Dimitriou & McKinley, 2014).
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'исходная_модель'))

import core_flood as cf  # noqa: E402
import validation_sandyga2020 as sd  # noqa: E402
from tests._patched_copy import make_copy  # noqa: E402
from paraphin import oil_composition as oc  # noqa: E402
from paraphin.constants import (D, Lk, R, v_asph, ro_asph, ro_o, sara_asphaltenes, sara_resins,  # noqa: E402
                                delta_sat, delta_aro, delta_res, c_oil_comp, beta_oil, P_ref_wax)

OUT = HERE / 'validation.json'
TABLES = HERE / 'validation_tables.md'
FIG = HERE / 'figures'
WORKER = HERE / 'core_worker.py'

LAB_GEL_TIME = 300.0  # [с]
GEL = {'gelation': 'True', 'wax_viscosity': '1', 'gel_time': repr(LAB_GEL_TIME)}
COMP = {'wax_components': 'True', 'wax_viscosity': '1'}
MULTS = (1.0, 0.1, 0.03, 0.01, 0.003)
# Множитель предела текучести для прогнозов: наибольший из MULTS, при котором гель не ухудшает калибровку
# по опыту 1 Sutton & Roberts больше чем на RMS_TOL (СКО k/k0)
RMS_TOL = 0.01


class _CalibrationStub:
    """`validation_li2024` при импорте читает d_p, L_k из outputs/data/core_flood.json (его пишет часовая
    калибровка `core_flood.py`). Те же значения стоят в constants.py - они и подставляются."""

    @staticmethod
    def read_text(encoding=None):
        return json.dumps({'best': {'1': {'d_p': D, 'lk': Lk}}})


def _li_module():
    saved, cf.RESULT = cf.RESULT, _CalibrationStub()
    try:
        import validation_li2024 as li
    finally:
        cf.RESULT = saved
    return li


li = _li_module()


# --- Прогон ------------------------------------------------------------------------------------------------

_DONE = {}  # прогоны из прошлого validation.json: при повторном запуске считаются только новые варианты


def run(name: str, exp: dict, flags: dict = None, mode: str = 'rate', extra: dict = None, dt: float = cf.DT) -> dict:
    if name in _DONE:
        return _DONE[name]
    values = cf._case_constants(exp, D, Lk, cf.NY, dt)
    values.update(extra or {})
    values.update(flags or {})
    root = make_copy(f'val_{name}', values)
    case, out = root / 'case.json', root / 'result.json'
    case.write_text(json.dumps(exp), encoding='utf-8')
    env = dict(os.environ, PYTHONPATH=str(root))
    subprocess.run([sys.executable, str(WORKER), str(case), str(out), mode], cwd=root, env=env, check=True)
    res = json.loads(out.read_text(encoding='utf-8'))
    res['rms'] = cf.rms(res, exp['exp'])
    idx = np.unique(np.linspace(0, len(res['pv']) - 1, 400).astype(int))
    for key in ('pv', 'k', 'k_harm', 'phi_min', 'T_hist'):
        if res.get(key):
            res[key] = [res[key][i] for i in idx]
    at = '/'.join(f'{np.interp(x, res["pv"], res["k"]):.2f}' for x in (0.25, 1.0, 3.0, 5.0))
    print(f'{name}: СКО {res["rms"]:.3f}, k/k0 при 0.25/1/3/5 PV = {at}, закупорка {res["plugged"]}, '
          f'гель в начале x{res["gel_start"]:.2f}', flush=True)
    return res


def sutton_roberts(results: dict) -> float:
    for number in (1, 2):
        results[f'sr{number}_legacy'] = run(f'sr{number}_legacy', cf.EXPERIMENTS[number])
    for mult in MULTS:
        results[f'sr1_gel_{mult}'] = run(f'sr1_gel_{mult}', cf.EXPERIMENTS[1], dict(GEL, gel_tau_mult=repr(mult)))
    base = results['sr1_legacy']['rms']
    ok = [m for m in MULTS if results[f'sr1_gel_{m}']['rms'] <= base + RMS_TOL]
    mult = max(ok) if ok else min(MULTS)
    print(f'Множитель предела текучести для прогнозов: {mult}', flush=True)
    results[f'sr2_gel_{mult}'] = run(f'sr2_gel_{mult}', cf.EXPERIMENTS[2], dict(GEL, gel_tau_mult=repr(mult)))
    return mult


def li_core(results: dict, mult: float) -> None:
    visc = li.viscosity()
    exp = dict(li.CORE, mu=visc['mu25'] * 1e-3)
    results['li25_legacy'] = run('li25_legacy', exp)
    results['li25_comp'] = run('li25_comp', exp, COMP)
    for m in sorted({1.0, mult}, reverse=True):
        results[f'li25_gel_{m}'] = run(f'li25_gel_{m}', exp, dict(COMP, **GEL, gel_tau_mult=repr(m)))


def sensitivity(results: dict) -> None:
    """Гель только из взвеси (`gel_deposit_weight = 0`) и статический порог гелеобразования 2 % для системы
    парафин-керосин (Letoffe et al., 1995): чем объясняется расхождение опытов между собой."""
    dep0 = dict(GEL, gel_tau_mult='1.0', gel_deposit_weight='0.0')
    results['sr1_gel_dep0_1.0'] = run('sr1_gel_dep0_1.0', cf.EXPERIMENTS[1], dep0)
    exp = dict(li.CORE, mu=li.viscosity()['mu25'] * 1e-3)
    results['li25_gel_dep0_1.0'] = run('li25_gel_dep0_1.0', exp, dict(COMP, **dep0))
    sd_exp, sd_extra = _sandyga_case()
    for name, flags in (('sd_gelphi_0.02_1.0', dict(GEL, gel_phi='0.02', gel_tau_mult='1.0')),
                        ('sd_gelphi_0.02_dep0_1.0', dict(dep0, gel_phi='0.02'))):
        results[name] = run(name, sd_exp, flags, mode='ramp', extra=sd_extra, dt=2.0)


def _sandyga_case():
    pore_fit, wat_res = sd.pores(), sd.wat()
    r_m, sigma = pore_fit['r_m'], pore_fit['sigma']
    extra = {'r_m': repr(r_m), 'sigma_r': repr(sigma), 'r_max': repr(40e-6 * r_m / 12e-6)}
    area = np.pi * sd.CORE_D ** 2 / 4.0
    pv0 = sd.CORE_L * area * sd.CORE_M
    exp = dict(
        title='wax in kerosene, sandstone (Sandyga et al., 2020)',
        k0=sd.core_k0(r_m, sigma) / sd.DARCY, porosity=sd.CORE_M, length=sd.CORE_L, side=float(np.sqrt(area)),
        q=sd.Q, T=sd.T_START, T_end=sd.T_END, cooling=sd.COOLING, plugged=1e-3,
        pv_end=sd.Q * (sd.T_START - sd.T_END) / sd.COOLING / pv0,
        w=sd.W_WAX, MW=sd.MW_WAX, M_o=sd.M_KEROSENE, dH=sd.DH, ro_o=sd.RO_SOLUTION, ro_p=sd.RO_WAX, mu=sd.MU_40,
        Tm=wat_res['tm_core'], exp=[],
    )
    return exp, extra


def sandyga_core(results: dict, mult: float) -> None:
    exp, extra = _sandyga_case()
    results['sd_legacy'] = run('sd_legacy', exp, mode='ramp', extra=extra, dt=2.0)
    for m in sorted({1.0, mult}, reverse=True):
        results[f'sd_gel_{m}'] = run(f'sd_gel_{m}', exp, dict(GEL, gel_tau_mult=repr(m)), mode='ramp',
                                     extra=extra, dt=2.0)


# --- Асфальтены: дегазированная нефть в опыте Li et al. ---------------------------------------------------

def asphaltene_lab() -> dict:
    """Доля выпавших асфальтенов по (Hirschberg, 1984) с параметрами поля, но для нефти опыта: дегазированной
    (газа нет), при давлении опыта. Возвращает выпавшее, % масс. нефти, по температуре и давлению."""
    w_sat = oc.REST0 * oc.F_SAT_REST + oc.WAX_TOTAL
    w_aro = oc.REST0 * (1.0 - oc.F_SAT_REST)
    phi0 = oc.asph_volume_fraction(sara_asphaltenes)

    def precipitated(t, p, gas):
        w_liq = w_sat + w_aro + sara_resins
        d_liq = (w_sat * delta_sat + w_aro * delta_aro + sara_resins * delta_res) / w_liq
        d_liq *= 1.0 + c_oil_comp * (p - P_ref_wax) - beta_oil * (t - 25.0)
        vg = (oc.gas_moles(p) if gas else 0.0) * oc.v_gas
        phi_g = vg / (vg + w_liq * oc.V_LIQ)
        d_m = (1.0 - phi_g) * d_liq + phi_g * oc.delta_gas
        ln_max = v_asph / oc.V_M - 1.0 - v_asph * ((oc.DELTA_ASPH - d_m) * 1e3) ** 2 / (R * (t + 273.15))
        phi_max = math.exp(ln_max)
        dphi = max(phi0 - phi_max, 0.0)
        return 100.0 * dphi * ro_asph / (dphi * ro_asph + (1.0 - phi0) * ro_o) if dphi > 0 else 0.0

    temps = [25.0, 45.0, 65.0, 90.0]
    pressures = np.linspace(0.1e6, 30e6, 60)
    return {
        'T': temps, 'P': (pressures / 1e6).tolist(),
        'dead': {str(t): [precipitated(t, p, False) for p in pressures] for t in temps},
        'live': {str(t): [precipitated(t, p, True) for p in pressures] for t in temps},
        'CII': float(oc.CII0), 'delta_a': float(oc.DELTA_ASPH),
        'plateau_exp': {str(t): li.K_PV[t][-1][1] for t in (90, 65, 45)},
    }


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    if sys.argv[1:2] != ['--plot']:
        if OUT.is_file():
            _DONE.update(json.loads(OUT.read_text(encoding='utf-8'))['runs'])
        results = {}
        mult = sutton_roberts(results)
        li_core(results, mult)
        sandyga_core(results, mult)
        sensitivity(results)
        data = {'runs': results, 'mult': mult, 'asphaltenes': asphaltene_lab(), 'lab_gel_time': LAB_GEL_TIME}
        OUT.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        print(f'Записано: {OUT}')
    from validation_figures import figures, tables
    data = json.loads(OUT.read_text(encoding='utf-8'))
    figures(data)
    tables(data)


if __name__ == '__main__':
    main()
