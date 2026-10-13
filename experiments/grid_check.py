"""Сеточная сходимость керновых прогонов: итоговые наборы опытов при NY = 30/60/120 ячеек и шаге 2/1 с.

    python experiments/grid_check.py        # прогоны (кэш common.run_many) -> results/grid_check.json

Сравнения с опытами считаются на NY = 60 и DT = 2 с (`common.py`). Здесь те же прогоны повторяются на грубой и
мелкой сетке и с половинным шагом; мерой служит наибольшее отклонение k/k0 в опытных точках от прогона на самой
мелкой сетке и изменение СКО от опыта. Задания берутся из модулей опытов с итоговыми параметрами (`params.json`),
а сетка и шаг подставляются в их константы: копия пакета на каждую сетку своя.
"""
import sys

import numpy as np

import li2024 as li
import sandyga2020 as sd
import sutton_roberts as sr
from common import run_many, load_params, save_results, k_residuals

GRIDS = ((30, 2.0), (60, 2.0), (120, 2.0), (60, 1.0))  # (NY, DT); эталон - NY = 120


def _regrid(job, ny, dt):
    name, consts, case = job
    return f'{name}_g{ny}_dt{dt:g}', dict(consts, **{'Nx, Ny': f'1, {ny}', 'dt': repr(dt)}), case


def _jobs():
    """{опыт: (задание при NY = 60, невязки модели в опытных точках)} для итоговых наборов."""
    out = {}
    p = load_params('sutton_roberts_lsq')
    for n in (1, 2):
        fam = sr.MAIN_FAMILY
        job = sr.job(fam, n, sr.LSQ_FAMILIES[fam]['flags'], sr.lsq_kin(fam, p[fam][f'own{n}']))
        out[f'Sutton & Roberts {n}'] = (job, lambda r, n=n: k_residuals(r, sr.EXPS[n]['k_pv']))
    pl, pc = load_params('li2024'), load_params('li2024_cold')
    kin_ret = pl[pl['best']]['kin']
    out['Li et al.'] = (li.cold_job(pc['gel_time'], pc['x'], kin_ret), lambda r: li.stage_residuals(r, li.STAGES))
    ps = load_params('sandyga2020')
    wat, p_best, net = ps['best'][0], tuple(ps['best'][1:]), ps.get('net')
    out['Sandyga et al.'] = (sd.kin_case(wat, p_best, net), sd.residuals)
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    jobs = _jobs()
    flat = [(name, g) for name in jobs for g in GRIDS]
    res = run_many([_regrid(jobs[name][0], *g) for name, g in flat])
    by = {(name, g): r for (name, g), r in zip(flat, res)}
    out = {}
    for name, (_, resid) in jobs.items():
        ref = resid(by[(name, (120, 2.0))])
        rows = {}
        for g in GRIDS:
            d = resid(by[(name, g)])
            # модель в опытных точках = опыт + невязка, поэтому разность невязок - разность моделей
            rows[f'NY={g[0]}, dt={g[1]:g} с'] = dict(rms=float(np.sqrt(np.mean(d * d))),
                                                    max_dev=float(np.max(np.abs(d - ref))))
        out[name] = rows
        print(name, {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in rows.items()}, flush=True)
    save_results('grid_check', out)


if __name__ == '__main__':
    main()
