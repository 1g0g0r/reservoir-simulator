"""Прогон одномерного керна в копии пакета: общий для всех сравнений с опытами (`experiments/`).

    python experiments/core_runner.py case.json out.json

Как `experiments/исходная_модель/core_flood.worker`, но:
  - пакет `paraphin` берется из PYTHONPATH - из копии с поправленными константами (`tests/_patched_copy.py`), а не
    из репозитория: флаги моделей - константы уровня модуля, править их на месте опасно;
  - числовые параметры кинетики (`solver.kin`, `paraphin/kinetics_params.py`) задаются в case.json и ставятся
    в уже скомпилированный решатель - подбор по опыту не перекомпилирует пакет;
  - гель: вязкость при заданной температуре ставится с множителем подвижности, а до начала прокачки множитель
    приводится к равновесию с перепадом при заданном расходе (нефть уже гелирована);
  - в результат пишутся профили вдоль керна и интегральные величины всех включенных механизмов.

case.json: {"exp": {...свойства опыта...}, "mode": "rate" | "ramp" | "thermal" | "stages", "kin": {...}}
- см. `experiments/common.py`, который его пишет. Режим 'stages' - ступенчатый протокол (Li et al. 2024): один керн
охлаждается ступенями exp["stages"] = [[T, PV], ...]; на каждой ступени керн и втекающая нефть при T, проницаемость
нормируется на начало ступени, отложения и адсорбция переходят на следующую ступень.

Постановка (Ring et al. 1994; `core_flood.worker`): керн вдоль j, слева прокачка нефти с постоянным расходом,
справа противодавление. Расход держится перепадом dP = q*sum(hy*mu_j/(A*k_j)); измеряемая величина та же, что в
опыте, k/k0 = dP_0/dP при постоянном расходе.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

T_CORE_DEFAULT = 21.1


def _gel_equilibrium(solver, u, phi_min):
    """Равновесный множитель подвижности геля по ячейкам при скорости фильтрации u (бисекция, см. `docs/`)."""
    from paraphin.constants import init_m, asphaltenes, ro_asph_dep
    from paraphin.equations import yield_stress, gel_phi_eq, pore_solid_fraction
    from paraphin.utils import crystal_volume_fraction

    for j in range(solver.Phi.shape[1]):
        v_dep = init_m - solver.m[0, j]
        if asphaltenes:
            from paraphin.oil_composition import IA_F, I_R
            v_dep -= (solver.Dep[0, j, IA_F] + solver.Dep[0, j, I_R]) / ro_asph_dep
        phi_s = pore_solid_fraction(solver.m[0, j], solver.S[0, j], crystal_volume_fraction(solver.Wps[0, j]),
                                    max(v_dep, 0.0))
        tau_y = yield_stress(phi_s)

        def phi_eq(phi):
            return gel_phi_eq(0, j, solver.fi, u * solver.mu_p[0, j] / (solver.k[0, j] * max(phi, phi_min)), tau_y)

        if phi_eq(phi_min) < phi_min:
            solver.Phi[0, j] = phi_eq(phi_min)
            continue
        lo, hi = phi_min, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if phi_eq(mid) > mid else (lo, mid)
        solver.Phi[0, j] = 0.5 * (lo + hi)
    solver.mu_o[0] = solver.mu_p[0] / np.maximum(solver.Phi[0], phi_min)


def _inflow_suspension(solver, t_c):
    """Взвесь групп во втекающей нефти - равновесная при t_c (флаг `wax_kinetics`: взвесь - переносимое
    состояние). Нужна ступенчатому протоколу: нефть охлаждали до температуры ступени до закачки (Li et al., 2024,
    разд. 2.2.3), а `Solver.initialize` ставит взвесь по температуре границы один раз, при первой ступени."""
    from paraphin.constants import init_p, data_type
    from paraphin.oil_composition import N_W, IS0
    from paraphin.equations.Thermo_wax import sle_split
    from paraphin.solver import Bound
    left = Bound.Left.value
    sus = np.zeros(N_W, data_type)
    sle_split(solver.bc_Wc[left, :N_W].copy(), float(t_c), float(init_p), sus)
    solver.bc_Wc[left, IS0:IS0 + N_W] = sus


def run(case: dict) -> dict:
    from paraphin import eta, r, fi_0, w2_cv
    from paraphin.constants import (Ny, hy, hx, h, init_Wp, init_Wps, _re, init_m, gelation,
                                    gel_mobility_min, wax_components, deposition_kinetics, wax_kinetics)
    from paraphin.kinetics_params import kin_index
    import paraphin.solver as solver_module
    from paraphin.solver import Solver, Bound, TypeBC, DataField
    from paraphin.utils import calc_mu_o, calc_mu_p

    # Слои полей керну не нужны (результат - кривая k/k0 и профили ниже). Выгрузка отключается: параллельные
    # прогоны в одной копии пакета делили бы общий файл слоев, и очистка одного прогона ломала бы другой.
    solver_module.save_fields = lambda solver, t: None
    solver_module._logging_solution = lambda solver, t: None

    exp, mode = case['exp'], case.get('mode', 'rate')
    p_out = exp['P_out']
    t_core = exp.get('T', T_CORE_DEFAULT)
    plugged_k = exp.get('plugged', 0.02)
    stages = exp['stages'] if mode == 'stages' else [[t_core, exp['pv_end']]]
    pv_end = float(sum(pv for _, pv in stages))
    area = hx * h
    pv0 = exp['length'] * area * init_m

    solver = Solver()
    for name, value in case.get('kin', {}).items():
        solver.kin[kin_index(name)] = value
    solver.add_well(name='Producer', i=0, j=Ny - 1, p=p_out, rw=0.5 * _re, mult=1e-12, is_injector=False)
    profile = np.linspace(exp.get('T_hot', t_core), t_core, Ny)

    def viscosity():
        for j in range(Ny):
            if gelation:
                solver.mu_p[0, j] = calc_mu_p(profile[j], solver.Wps[0, j], solver.p[0, j])
                solver.mu_o[0, j] = solver.mu_p[0, j] / max(solver.Phi[0, j], gel_mobility_min)
            else:
                solver.mu_o[0, j] = calc_mu_o(profile[j], solver.Wps[0, j])

    def impose_temperature():
        solver.T[0] = profile
        solver.T_0[0] = profile
        viscosity()

    if mode == 'thermal':
        impose_temperature()

    def pressure_drop():
        return exp['q'] * float(np.sum(hy * solver.mu_o[0] / (area * solver.k[0])))

    solver.add_bc(field=DataField.Pressure, bound=Bound.Left, type_bc=TypeBC.Dirichlet, value=p_out)
    solver.add_bc(field=DataField.Pressure, bound=Bound.Right, type_bc=TypeBC.Dirichlet, value=p_out)
    solver.add_bc(field=DataField.Saturation, bound=Bound.Left, type_bc=TypeBC.Dirichlet, value=0.0)
    solver.add_bc(field=DataField.Temperature, bound=Bound.Left, type_bc=TypeBC.Dirichlet,
                  value=exp.get('T_hot', t_core) if mode == 'thermal' else t_core)
    solver.add_bc(field=DataField.Paraffin, bound=Bound.Left, type_bc=TypeBC.Dirichlet, value=init_Wp + init_Wps)
    solver.initialize()

    dp_newton = pressure_drop()
    if gelation:
        _gel_equilibrium(solver, exp['q'] / area, gel_mobility_min)
    dp0 = pressure_drop()
    solver.boundary_conditions[Bound.Left.value, DataField.Pressure.value, 1] = p_out + dp0

    k0 = solver.k[0, 0]
    t, pv, k_app, k_harm, t_hist = 0.0, [0.0], [1.0], [1.0], [stages[0][0]]
    stage_of = [0]
    plugged = None
    injected = 0.0
    pv_stage_end = 0.0
    for n_stage, (t_stage, pv_stage) in enumerate(stages):
        pv_stage_end += pv_stage
        if mode == 'stages':
            profile[:] = t_stage
            impose_temperature()
            solver.boundary_conditions[Bound.Left.value, DataField.Temperature.value, 1] = t_stage
            if wax_kinetics:
                _inflow_suspension(solver, t_stage)
            # Выдержка без прокачки: керн и нефть охлаждали до температуры ступени до закачки (Li et al., 2024,
            # разд. 2.2.3), и начальная проницаемость ступени измерена уже по охлажденной нефти. Без выдержки
            # взвесь в поровой нефти выпадает после нормировки, и рост вязкости принимается за повреждение.
            # Перепада нет - перенос и блокирование стоят, идут равновесие, гель и кристаллизация на стенках.
            hold = exp.get('stage_hold', 0.0) if n_stage > 0 else 0.0
            if hold > 0.0:
                solver.boundary_conditions[Bound.Left.value, DataField.Pressure.value, 1] = p_out
                t_hold = 0.0
                while t_hold < hold and plugged is None:
                    step = solver.dt
                    try:
                        solver.upd_time_step(t + step)
                    except ValueError:  # вырожденная матрица давления: где-то нулевая подвижность - закупорка
                        plugged = pv[-1]
                    t += step
                    t_hold += step
                if plugged is not None:
                    break
                impose_temperature()
            if gelation:
                _gel_equilibrium(solver, exp['q'] / area, gel_mobility_min)
            dp0 = pressure_drop()  # проницаемость ступени нормируется на ее начало, как в опыте
            if n_stage > 0:
                pv.append(pv[-1])
                k_app.append(1.0)
                k_harm.append(k_harm[-1])
                t_hist.append(t_stage)
                stage_of.append(n_stage)
        while injected < pv_stage_end * pv0:
            if mode == 'thermal':
                impose_temperature()
            elif mode == 'ramp':
                t_now = max(t_core - exp['cooling'] * t, exp['T_end'])
                profile[:] = t_now
                impose_temperature()
                solver.boundary_conditions[Bound.Left.value, DataField.Temperature.value, 1] = t_now
            dp = pressure_drop()
            k_now = dp0 / dp
            if k_now < plugged_k:
                plugged = pv[-1]
                break
            solver.boundary_conditions[Bound.Left.value, DataField.Pressure.value, 1] = p_out + dp
            step = solver.dt
            try:
                solver.upd_time_step(t + step)
            except ValueError:  # вырожденная матрица давления - керн закупорен
                plugged = pv[-1]
                break
            t += step
            injected += exp['q'] * step
            pv.append(injected / pv0)
            k_app.append(k_now)
            k_harm.append(float(Ny / np.sum(k0 / solver.k[0])))
            t_hist.append(float(solver.T[0, 0]))
            stage_of.append(n_stage)
        if plugged is not None:
            break

    # Прореживание истории до ~600 точек: весь ход кривой, без мегабайтных json
    idx = np.unique(np.linspace(0, len(pv) - 1, 600).astype(int))
    m_cond = (solver.fi[0] * w2_cv).sum(axis=-1) / solver.integr_r2_fi0  # проводящая пористость, доли m0
    result = {
        'pv': [pv[i] for i in idx], 'k': [k_app[i] for i in idx], 'k_harm': [k_harm[i] for i in idx],
        'T_hist': [t_hist[i] for i in idx] if mode in ('ramp', 'stages') else [],
        'stage': [stage_of[i] for i in idx],
        'plugged': plugged, 'mode': mode, 'eta': float(eta), 'gel_start': dp_newton / dp0,
        'k_profile': (solver.k[0] / k0).tolist(), 'm_profile': (solver.m[0] / init_m).tolist(),
        'm_conductive_profile': m_cond.tolist(), 'phi_profile': solver.Phi[0].tolist(),
        'r': r.tolist(), 'fi0': fi_0.tolist(), 'fi_end': [solver.fi[0, j].tolist() for j in (0, Ny // 2, Ny - 1)],
        'kin': solver.kin.tolist(),
    }
    if wax_components:
        from paraphin.oil_composition import N_W, IA_F, I_R
        result['dep_wax'] = float(solver.Dep[0, :, :N_W].sum())
        result['dep_asph'] = float(solver.Dep[0, :, IA_F].sum() + solver.Dep[0, :, I_R].sum())
    if deposition_kinetics:
        from paraphin.kinetics_params import KX_GA, KX_GR, KX_VGEL
        result['adsorbed'] = float(solver.kx[0, :, KX_GA].sum() + solver.kx[0, :, KX_GR].sum())
        result['gel_volume'] = (solver.kx[0, :, KX_VGEL] / init_m).tolist()
    return result


def main() -> None:
    case = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    out = Path(sys.argv[2])
    out.write_text(json.dumps(run(case)), encoding='utf-8')


if __name__ == '__main__':
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    main()
