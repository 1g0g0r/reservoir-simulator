"""Рабочий процесс проверок со включенными флагами детального состава (см. `tests/_patched_copy.py`).

Запускается против копии пакета с поправленными константами:
    python _flags_on_worker.py out.json дни [прогонов] [fields.npz]
Последний аргумент - сохранить поля конца первого прогона (сверка режима 'single' с эталоном).
Считает невязки законов сохранения по ходу расчета и пишет их в JSON; сами утверждения - в
`tests/test_composition_flags_on.py`. Постановка - как `test_mass_balance._make_solver`: закачка холодной
воды в угол, отбор из противоположного, границы непроницаемы.
"""
import hashlib
import json
import sys

import numpy as np


def _run(t_days: float, fields_path: str = None) -> dict:
    from paraphin.constants import (Nx, Ny, Pw, Po, rw, day_to_sec, ro_o, ro_p, ro_asph_dep, init_m, volume,
                                    dt_min, gel_mobility_min, asphaltenes, gelation)
    from paraphin.oil_composition import N_W, NCB, IA_D, IA_F, I_R
    from paraphin.kinetics_params import KX_QW, KX_QG, KX_QADA, KX_QADR, KX_GA, KX_GR
    from paraphin.solver import Solver

    solver = Solver()
    solver.add_well(name='Injector', i=0, j=0, p=Pw, rw=rw, mult=0.25, is_injector=True, T=5.0)
    solver.add_well(name='Producer', i=Nx - 1, j=Ny - 1, p=Po, rw=rw, mult=0.25, is_injector=False)
    solver.initialize()
    prod = solver.wells[solver._producer]
    pi, pj = prod.i, prod.j

    def masses():
        # Только «массовые» компоненты (NCB): переносимые состояния кинетики (взвесь групп, число флокул) - не массы.
        # Адсорбированные асфальтены и смолы - отдельный резервуар (`kx`), они входят в баланс своих компонентов.
        oil = solver.m * (1.0 - solver.S) * ro_o * volume
        out = np.array([(oil * solver.Wc[..., c]).sum() + solver.Dep[..., c].sum() * volume for c in range(NCB)])
        out[IA_D] += solver.kx[..., KX_GA].sum() * volume
        out[I_R] += solver.kx[..., KX_GR].sum() * volume
        return out

    solver.upd_time_step(0.0)  # прогрев JIT - полноценный шаг, поэтому начальные массы берутся после него
    initial = masses()
    produced = np.zeros(NCB)

    t, t_end = 0.0, t_days * day_to_sec
    worst_q = worst_pore = worst_dep = worst_closure = worst_groups = 0.0
    min_dt, phi_hist, n_steps = np.inf, [], 0
    while t < t_end:
        step_dt = solver.dt
        m_before = solver.m.copy()
        wc_prod = solver.Wc[pi, pj, :NCB].copy()
        t += step_dt
        solver.upd_time_step(t)
        n_steps += 1
        min_dt = min(min_dt, step_dt)

        # Отбор: дебит нефти этого шага (q < 0) с составом ячейки на начало шага - как в `components_equation`
        produced += -prod.q[0] * ro_o * wc_prod * step_dt

        q_in, q_out = solver.wells[0].q[2], -solver.wells[1].q[2]
        worst_q = max(worst_q, abs((q_out - q_in) / q_in))
        dm = m_before - solver.m
        q_new = solver.kx[..., KX_QW] + solver.kx[..., KX_QG] + solver.kx[..., KX_QADA] + solver.kx[..., KX_QADR]
        res = np.abs((solver.qp1 + solver.qp2 + solver.qpa + q_new) * step_dt - dm).max()
        # Знаменатель не меньше 1e-6: m - m_new - разность чисел ~0.3, ее точность ограничена округлением (~5e-17),
        # и при изменении пористости за шаг ~1e-8 (одна адсорбция) относительная невязка мерила бы только его
        worst_pore = max(worst_pore, res / max(np.abs(dm).max(), 1e-6))
        dep_vol = (solver.Dep[..., :N_W].sum(axis=-1) / ro_p
                   + (solver.Dep[..., IA_F] + solver.Dep[..., I_R] + solver.kx[..., KX_GA] + solver.kx[..., KX_GR]) / ro_asph_dep)
        loss = init_m - solver.m
        worst_dep = max(worst_dep, np.abs(dep_vol - loss).max() / max(np.abs(loss).max(), 1e-30))
        worst_closure = max(worst_closure, np.abs(solver.Wo + solver.Wp + solver.Wps - 1.0).max())
        worst_groups = max(worst_groups, np.abs(solver.Wc[..., :N_W].sum(axis=-1) - solver.Wp - solver.Wps).max())
        phi_hist.append(solver.Phi.copy())

    # Асфальтены растворенные и флокулы переходят друг в друга - сохраняется их сумма
    final = masses() + produced
    final[IA_D] += final[IA_F]
    final[IA_F] = 0.0
    init = initial.copy()
    init[IA_D] += init[IA_F]
    init[IA_F] = 0.0
    balance = np.where(init > 0.0, (final - init) / np.maximum(init, 1e-30), 0.0)
    phi = np.array(phi_hist[-51:])
    dphi = np.diff(phi, axis=0)
    flips = int((np.sign(dphi[1:]) * np.sign(dphi[:-1]) < 0).sum(axis=0).max()) if dphi.shape[0] > 1 else 0
    fingerprint = hashlib.sha256(b''.join(np.ascontiguousarray(a).tobytes() for a in (
        solver.p, solver.S, solver.T, solver.m, solver.k, solver.Wc, solver.Ws, solver.fi, solver.Phi,
        solver.Dep))).hexdigest()

    if fields_path:
        np.savez(fields_path, p=solver.p, S=solver.S, T=solver.T, m=solver.m, k=solver.k, Wp=solver.Wp,
                 Wps=solver.Wps, fi=solver.fi, KIN=np.array(solver.KIN))

    return {
        'n_steps': n_steps, 'min_dt': min_dt, 'dt_min': dt_min,
        'worst_q': worst_q, 'worst_pore': worst_pore, 'worst_dep': worst_dep,
        'worst_closure': worst_closure, 'worst_groups': worst_groups,
        'balance': balance.tolist(), 'initial': initial.tolist(),
        'min_frac': float(min(solver.Wc.min(), solver.Wp.min(), solver.Wps.min())),
        'max_flocs': float(solver.Wc[..., IA_F].max()), 'min_asph_dissolved': float(solver.Wc[..., IA_D].min()),
        'dep_wax': float(solver.Dep[..., :N_W].sum()), 'dep_asph': float(solver.Dep[..., IA_F].sum()),
        'dep_resin': float(solver.Dep[..., I_R].sum()),
        'phi_min': float(solver.Phi.min()), 'phi_max': float(solver.Phi.max()), 'phi_flips': flips,
        'gel_mobility_min': gel_mobility_min, 'asphaltenes': asphaltenes, 'gelation': gelation,
        'KIN': float(solver.KIN), 'fingerprint': fingerprint, 'max_suspended': float(solver.Wps.max()),
        'Wp_mean': float(solver.Wp.mean()), 'Wps_mean': float(solver.Wps.mean()), 'k_min': float(solver.k.min()),
        'T_min': float(solver.T.min()),
    }


def main() -> None:
    out, t_days = sys.argv[1], float(sys.argv[2])
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    fields_path = sys.argv[4] if len(sys.argv) > 4 else None
    results = [_run(t_days, fields_path if r == 0 else None) for r in range(runs)]
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'runs': results}, f)


if __name__ == '__main__':
    main()
