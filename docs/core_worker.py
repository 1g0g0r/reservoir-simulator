"""Рабочий процесс прогона керна в копии пакета с поправленными константами (`docs/validate.py`).

    python core_worker.py case.json out.json режим

То же, что `твт_статья/core_flood.worker` (одномерный керн, постоянный расход, k/k0 = dP_0/dP), но пакет
`paraphin` берется из PYTHONPATH - из копии, а не из репозитория: флаги детального состава и геля -
константы уровня модуля, и править ради них `constants.py` на месте опасно (см. `tests/_patched_copy.py`).

Отличия от `core_flood.worker`, нужные гелю (флаг `gelation`):
  - в режимах с заданной температурой вязкость ставится с множителем подвижности геля,
    mu_o = mu_p/max(Phi, Phi_min), а не `calc_mu_o`, - иначе перепад считался бы без геля;
  - до начала прокачки множитель подвижности приводится к равновесию с перепадом при заданном расходе
    (нефть в керне и на входе уже гелирована). Опыт нормирует k на начало ступени, поэтому и здесь
    k/k0 = 1 в начале, а отношение начального перепада к перепаду без геля пишется отдельно (`gel_start`).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / 'твт_статья'))  # в конец: paraphin должен браться из копии (PYTHONPATH)
import core_flood as cf  # noqa: E402


def _gel_equilibrium(solver, u, phi_min):
    """Равновесный множитель подвижности по ячейкам керна при скорости фильтрации u, [м/с].

    В одномерном однофазном керне градиент в ячейке j известен: |grad p| = u*mu_p/(k*max(Phi, Phi_min)),
    а Phi_eq(|grad p|) растет с градиентом. Корень Phi = Phi_eq(u*mu_p/(k*Phi)) единственный - бисекция.
    """
    from paraphin.constants import init_m, asphaltenes, ro_asph_dep
    from paraphin.equations import yield_stress, gel_phi_eq, pore_solid_fraction
    from paraphin.utils import crystal_volume_fraction

    ny = solver.Phi.shape[1]
    for j in range(ny):
        v_dep = init_m - solver.m[0, j]
        if asphaltenes:
            from paraphin.oil_composition import IA_F, I_R
            v_dep -= (solver.Dep[0, j, IA_F] + solver.Dep[0, j, I_R]) / ro_asph_dep
        phi_s = pore_solid_fraction(solver.m[0, j], solver.S[0, j], crystal_volume_fraction(solver.Wps[0, j]),
                                    max(v_dep, 0.0))
        tau_y = yield_stress(phi_s)

        def phi_eq(phi):
            return gel_phi_eq(0, j, solver.fi, u * solver.mu_p[0, j] / (solver.k[0, j] * max(phi, phi_min)), tau_y)

        if phi_eq(phi_min) < phi_min:  # и при наибольшем градиенте подвижность ниже границы
            solver.Phi[0, j] = phi_eq(phi_min)
            continue
        lo, hi = phi_min, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if phi_eq(mid) > mid else (lo, mid)
        solver.Phi[0, j] = 0.5 * (lo + hi)
    solver.mu_o[0] = solver.mu_p[0] / np.maximum(solver.Phi[0], phi_min)


def worker(case: Path, out: Path, mode: str = 'rate') -> None:
    from shutil import rmtree
    from paraphin import eta, r, fi_0
    from paraphin.constants import Ny, hy, hx, h, init_Wp, init_Wps, results_path, _re, init_m, gelation
    from paraphin.constants import gel_mobility_min, wax_components
    from paraphin.solver import Solver, Bound, TypeBC, DataField
    from paraphin.utils import calc_mu_o, calc_mu_p

    exp = json.loads(Path(case).read_text(encoding='utf-8'))
    t_core, pv_end, plugged_k = exp.get('T', cf.T_CORE), exp.get('pv_end', cf.PV_END), exp.get('plugged', cf.PLUGGED)
    area = hx * h
    pv0 = exp.get('length', cf.LENGTH) * area * init_m

    solver = Solver()
    # Заглушка: `initialize` требует одну добывающую скважину на забойном давлении; расход задают ГУ
    solver.add_well(name='Producer', i=0, j=Ny - 1, p=cf.P_OUT, rw=0.5 * _re, mult=1e-12, is_injector=False)
    profile = np.linspace(exp.get('T_hot', cf.T_HOT), t_core, Ny)

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

    def resistance():
        return float(np.sum(hy * solver.mu_o[0] / (area * solver.k[0])))

    def pressure_drop():
        return exp['q'] * resistance()

    solver.add_bc(field=DataField.Pressure, bound=Bound.Left, type_bc=TypeBC.Dirichlet, value=cf.P_OUT)
    solver.add_bc(field=DataField.Pressure, bound=Bound.Right, type_bc=TypeBC.Dirichlet, value=cf.P_OUT)
    solver.add_bc(field=DataField.Saturation, bound=Bound.Left, type_bc=TypeBC.Dirichlet, value=0.0)
    solver.add_bc(field=DataField.Temperature, bound=Bound.Left, type_bc=TypeBC.Dirichlet,
                  value=exp.get('T_hot', cf.T_HOT) if mode == 'thermal' else t_core)
    solver.add_bc(field=DataField.Paraffin, bound=Bound.Left, type_bc=TypeBC.Dirichlet, value=init_Wp + init_Wps)
    solver.initialize()

    dp_newton = pressure_drop()  # перепад без геля (Phi = 1 при создании решателя)
    if gelation:
        _gel_equilibrium(solver, exp['q'] / area, gel_mobility_min)
    dp0 = pressure_drop()
    solver.boundary_conditions[Bound.Left.value, DataField.Pressure.value, 1] = cf.P_OUT + dp0

    k0 = solver.k[0, 0]
    t, pv, k_app, k_harm, t_hist, phi_hist = 0.0, [0.0], [1.0], [1.0], [t_core], [float(solver.Phi[0].min())]
    plugged = None
    injected = 0.0
    while injected < pv_end * pv0:
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
            pv.append(pv_end)
            k_app.append(0.0)
            k_harm.append(0.0)
            t_hist.append(t_hist[-1])
            phi_hist.append(phi_hist[-1])
            break
        solver.boundary_conditions[Bound.Left.value, DataField.Pressure.value, 1] = cf.P_OUT + dp
        step = solver.dt
        solver.upd_time_step(t + step)
        t += step
        injected += exp['q'] * step
        pv.append(injected / pv0)
        k_app.append(k_now)
        k_harm.append(float(Ny / np.sum(k0 / solver.k[0])))
        t_hist.append(float(solver.T[0, 0]))
        phi_hist.append(float(solver.Phi[0].min()))

    rmtree(results_path, ignore_errors=True)
    result = {
        'pv': pv, 'k': k_app, 'k_harm': k_harm, 'phi_min': phi_hist,
        'k_profile': (solver.k[0] / k0).tolist(), 'm_profile': (solver.m[0] / init_m).tolist(),
        'phi_profile': solver.Phi[0].tolist(), 'gel_start': dp_newton / dp0,
        'eta': float(eta), 'plugged': plugged, 'mode': mode,
        'r': r.tolist(), 'fi0': fi_0.tolist(), 'fi_end': [solver.fi[0, j].tolist() for j in (0, Ny // 2, Ny - 1)],
        'T_hist': t_hist if mode == 'ramp' else [],
    }
    if wax_components:
        from paraphin.oil_composition import N_W
        result['dep_groups'] = solver.Dep[0, :, :N_W].sum(axis=0).tolist()  # [кг/м^3 породы], сумма по ячейкам
    out.write_text(json.dumps(result), encoding='utf-8')


if __name__ == '__main__':
    worker(Path(sys.argv[1]), Path(sys.argv[2]), *(sys.argv[3:4]))
