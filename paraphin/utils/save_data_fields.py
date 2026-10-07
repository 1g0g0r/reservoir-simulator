"""Дозапись полей данных очередного временного слоя в общий файл расчета."""
from pickle import dump, HIGHEST_PROTOCOL

import numpy as np

from paraphin.constants import (layers_file, init_k, init_m, Nx, Ny, wax_components, gelation, ro_o, ro_p,
                                ro_asph_dep, volume, _re, deposition_kinetics, adsorption, deposit_aging,
                                asph_aggregation, wax_kinetics, thermal_nonequilibrium, wax_eos, wax_pressure,
                                asphaltenes, wettability)
from paraphin.layout import (N_W, IA_D, IA_F, I_R, IN_F, KX_GA, KX_GR, KX_VGEL, KX_TS, KX_QW, KX_QG, KX_QADA, KX_QADR,
                             KX_GMAX, KX_UA, KX_QPA)

# Точка, в которой снимается кривая fi(r) для графиков: `visualisation._visualize_plots_fi` и `graphs._plot_fi`.
# Прижата к сетке: на одномерном керне (Nx = 1) точки (3, 3) нет.
FI_PROBE = (min(3, Nx - 1), min(3, Ny - 1))


def save_fields(solver, t: float) -> None:
    """Дозапись очередного временного слоя в общий файл расчета.

    Слои пишутся подряд в один открытый файл, а не в отдельный pkl на каждое сохранение: это убирает
    открытия файла при каждом сохранении, а также долгую пост-обработку множества файлов.
    Память при этом не растет: `dump` отдает байты в ОС и возвращается, в памяти живет ровно один слой.
    """
    wells, wells_accumulated = {}, {}
    for i in range(solver.n_wells):
        name, well = solver._wells_names[i], solver.wells[i]
        q, Q = well.q, well.Q
        wells[f'{name}_oil'] = q[0]
        wells[f'{name}_water'] = q[1]
        wells[f'{name}_total'] = q[2]
        wells[f'{name}_eta'] = well.eta
        wells_accumulated[f'{name}_Q_oil'] = Q[0]
        wells_accumulated[f'{name}_Q_water'] = Q[1]
        wells_accumulated[f'{name}_Q_total'] = Q[2]
        wells[f'{name}_bhp'] = well.p
        # Скин-фактор по Хокинсу (обзор 4.3): ячейка скважины - поврежденная зона радиуса r_e (Писман)
        wells[f'{name}_skin'] = (init_k / solver.k[well.i, well.j] - 1.0) * np.log(_re / well.rw)

    layer = {
        'Time': t,
        'Pressure': solver.p,
        'Saturation': solver.S,
        'Temperature': solver.T
    }

    if solver._paraphin:
        m_mult = solver.m / init_m
        x_idx, y_idx = FI_PROBE
        layer.update({
            'm': solver.m / init_m,
            'k': solver.k / init_k,
            'Wo': solver.Wo,
            'Wp': solver.Wp,
            'Wps': solver.Wps,
            # Осевший парафин - это ровно потерянный поровый объем (m_0 - m), отдельного поля нет: q_p1 + q_p2 = -dm/dt по построению
            'Wps dep': 1.0 - m_mult,
            'plots': {'fi': solver.fi[x_idx, y_idx]},
        })
        if wax_components:
            layer.update(_composition_fields(solver))

    if gelation:
        from paraphin.equations.Gel import yield_stress_field
        tau_y = np.empty_like(solver.S)
        yield_stress_field(solver.Wps, solver.S, solver.m, solver.Dep, tau_y)
        layer['Gel'] = {'Phi': solver.Phi, 'mu_o': solver.mu_o, 'mu_p': solver.mu_p, 'tau_y': tau_y}

    layer.update({
        'Wells': wells,
        'Wells_accumulated': wells_accumulated
    })

    # layer['Other params'] = {
    #     'KIN': solver.KIN,
    #     'S [0,0]': solver.S[0, 0], 'S [-1,-1]': solver.S[-1, -1],
    #     'P [0,0]': solver.p[0, 0], 'p [-1,-1]': solver.p[-1, -1],
    # }

    if solver._results_file is None:
        solver._results_file = open(layers_file, 'wb')

    dump(layer, solver._results_file, protocol=HIGHEST_PROTOCOL)
    solver.logger.info("Данные записаны в файл.")


def _composition_fields(solver) -> dict:
    """Поля детального состава (флаг `wax_components`) - компактно, по двумерному полю на величину.

    Полные массивы по компонентам (Wc, Ws, Dep) на каждом слое заняли бы в N_w + 3 раза больше места, чем
    остальной слой; их финальное состояние пишет `Solver.start` в `{case}_final_composition.npz`.
    'Wps dep' здесь - только парафин: пористость теряет и осадок асфальтенов со смолами ('Asph dep').
    """
    from paraphin.equations import calc_wat_field
    from paraphin.oil_composition import F_SAT_REST

    wc, dep = solver.Wc, solver.Dep
    calc_wat_field(wc, solver.p, solver.WAT)
    wax = wc[..., :N_W].sum(axis=-1)
    asph = wc[..., IA_D] + wc[..., IA_F]
    rest = np.maximum(1.0 - wax - asph - wc[..., I_R], 0.0)
    # Индекс коллоидной неустойчивости (Yen, Yin & Asomaning, SPE 65376, 2001): > 0.9 - асфальтены неустойчивы
    cii = (rest * F_SAT_REST + wax + asph) / np.maximum(rest * (1.0 - F_SAT_REST) + wc[..., I_R], 1e-12)
    comp = {'WAT': solver.WAT, 'Asph dissolved': wc[..., IA_D], 'Asph flocs': wc[..., IA_F], 'Resins': wc[..., I_R],
            'Asph dep': (dep[..., IA_F] + dep[..., I_R]) / ro_asph_dep / init_m, 'CII': cii,
            # Отложения по видам - в долях m0, как 'Wps dep'; скорость потери пористости на парафин этого шага, [1/с]
            'Asph flocs dep': dep[..., IA_F] / ro_asph_dep / init_m, 'Resins dep': dep[..., I_R] / ro_asph_dep / init_m,
            'Wax dep rate': solver.qp1 + solver.qp2}
    if asphaltenes:
        comp['Asph dep rate'] = solver.kx[..., KX_QPA]  # асфальтены + смолы, [1/с]
        comp['Asph Ua'] = solver.kx[..., KX_UA]         # сужение капилляров флокулами u_a = Ua*r^(1/3), [м^(2/3)/с]
    for k in range(N_W):
        comp[f'Wax {k + 1}'] = wc[..., k]
        comp[f'Wax {k + 1} susp'] = solver.Ws[..., k]
        comp[f'Wax {k + 1} dep'] = dep[..., k] / ro_p / init_m
    if wax_eos and wax_pressure:
        # Диагностика: объем газа, который выделился бы ниже P_b, на объем нефти (`thermo.pvt`); в уравнения не входит
        from paraphin.thermo.tables import GASV, eos_interp
        comp['Free gas'] = np.vectorize(lambda t, p: eos_interp(GASV, t, p))(solver.T, solver.p)

    # Массы компонентов в пласте, [кг]: в нефтяной фазе и в отложениях
    oil = solver.m * (1.0 - solver.S) * ro_o * volume
    totals = {}
    names = [f'wax {k + 1}' for k in range(N_W)] + ['asph', 'asph flocs', 'resins']
    for c, name in enumerate(names):
        totals[f'{name} in oil'] = float((oil * wc[..., c]).sum())
        totals[f'{name} deposited'] = float(dep[..., c].sum() * volume)

    if deposition_kinetics:
        comp.update(_kinetics_fields(solver))
        totals['asph adsorbed'] = float(solver.kx[..., KX_GA].sum() * volume)
        totals['resins adsorbed'] = float(solver.kx[..., KX_GR].sum() * volume)

    return {'Wps dep': dep[..., :N_W].sum(axis=-1) / ro_p / init_m, 'Composition': comp, 'Totals': totals}


def _kinetics_fields(solver) -> dict:
    """Поля моделей кинетики осаждения (`equations/Deposition.py`).

    'm conductive' - пористость проводящих каналов m0*int r^2*fi/int r^2*fi0, в долях m0: ее, а не m, видит
    томография (гель в тупиковых порах и захваченная нефть для нее - отложение, Sandyga et al. 2020)."""
    from paraphin.geometry import w2_cv
    from paraphin.equations.Deposition import floc_diameter

    kx = solver.kx
    # Скорости потери порового объема этого шага по механизмам, [1/с]: контракт ядра (`equations/Deposition.py`)
    out = {'m conductive': (solver.fi * w2_cv).sum(axis=-1) / solver.integr_r2_fi0, 'Wall cryst rate': kx[..., KX_QW]}
    if adsorption:
        out['Adsorbed asph'] = kx[..., KX_GA]    # [кг/м^3 породы]
        out['Adsorbed resins'] = kx[..., KX_GR]
        out['Asph ads rate'] = kx[..., KX_QADA]
        out['Resins ads rate'] = kx[..., KX_QADR]
    if wettability:
        # Доля покрытия поверхности асфальтенами - как в `calc_mobility_w`: 0 - водо-, 1 - нефтесмачиваемая
        g_max = kx[..., KX_GMAX]
        out['Wettability omega'] = np.where(g_max > 0.0, np.minimum(kx[..., KX_GA] / np.maximum(g_max, 1e-30), 1.0), 0.0)
    if deposit_aging:
        v_gel = kx[..., KX_VGEL]
        wax_v = solver.Dep[..., :N_W].sum(axis=-1) / ro_p
        out['Gel volume'] = v_gel / init_m
        out['Gel wax fraction'] = np.where(v_gel > 0.0, wax_v / np.maximum(v_gel, 1e-30), 0.0)
        out['Gel aging rate'] = kx[..., KX_QG]
    if wax_kinetics:
        out['Supersaturation'] = np.maximum(kx[..., :N_W] - solver.Ws, 0.0).sum(axis=-1)
    if asph_aggregation:
        out['Floc number'] = solver.Wc[..., IN_F]
        d = np.zeros(solver.kx.shape[:2])
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                d[i, j] = floc_diameter(solver.Wc[i, j, IA_F], solver.Wc[i, j, IN_F], solver.kin)
        out['Floc diameter'] = d
    if thermal_nonequilibrium:
        out['T rock'] = solver.kx[..., KX_TS]
    return out
