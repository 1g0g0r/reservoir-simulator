"""Дозапись полей данных очередного временного слоя в общий файл расчета."""
from pickle import dump, HIGHEST_PROTOCOL

import numpy as np

from paraphin.constants import (layers_file, init_k, init_m, Nx, Ny, gelation, ro_o, ro_p,
                                ro_asph_dep, volume, _re, deposition_kinetics, adsorption, deposit_aging,
                                asph_aggregation, wax_kinetics, thermal_nonequilibrium, wax_eos, wax_pressure,
                                asphaltenes, wettability)
from paraphin.layout import N_W, NC_T, named

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
        wells[f'{name}_oil'] = well.q_o
        wells[f'{name}_water'] = well.q_w
        wells[f'{name}_total'] = well.q_t
        wells[f'{name}_eta'] = well.eta
        wells_accumulated[f'{name}_Q_oil'] = well.Q_o
        wells_accumulated[f'{name}_Q_water'] = well.Q_w
        wells_accumulated[f'{name}_Q_total'] = well.Q_t
        wells[f'{name}_bhp'] = well.p
        # Скин-фактор по Хокинсу (обзор 4.3): ячейка скважины - поврежденная зона радиуса r_e (Писман)
        wells[f'{name}_skin'] = (init_k / solver.k[well.i, well.j] - 1.0) * np.log(_re / well.rw)

    layer = {
        'Time': t,
        # Шаг, которым посчитан слой, [с]: при адаптивном шаге иначе его не восстановить. В группе рядов от времени
        'Other params': {'dt': solver.dt_last},
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
        if N_W > 1 or NC_T > N_W or deposition_kinetics:  # однокомпонентной модели хватает полей выше
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
    """Поля детального состава (группы парафина, асфальтены, кинетика) - компактно, по двумерному полю на величину.

    Полные массивы по компонентам (Wc, Ws, Dep) на каждом слое заняли бы в N_w + 3 раза больше места, чем
    остальной слой; их финальное состояние пишет `Solver.start` в `{case}_final_composition.npz`.
    'Wps dep' здесь - только парафин: пористость теряет и осадок асфальтенов со смолами ('Asph dep').
    """
    from paraphin.equations import calc_wat_field
    from paraphin.oil_composition import F_SAT_REST

    wc, dep = named(solver.Wc), named(solver.Dep)
    calc_wat_field(solver.Wc, solver.p, solver.WAT)
    # Скорость потери пористости на парафин этого шага, [1/с]
    comp = {'WAT': solver.WAT, 'Wax dep rate': solver.qp1 + solver.qp2}
    if NC_T > N_W:  # асфальтены и смолы переносятся; иначе их доли - начальные и ничего не значат
        wax = wc['wax'].sum(axis=-1)
        asph = wc['asph_d'] + wc['asph_f']
        rest = np.maximum(1.0 - wax - asph - wc['resin'], 0.0)
        # Индекс коллоидной неустойчивости (Yen, Yin & Asomaning, SPE 65376, 2001): > 0.9 - асфальтены неустойчивы
        cii = (rest * F_SAT_REST + wax + asph) / np.maximum(rest * (1.0 - F_SAT_REST) + wc['resin'], 1e-12)
        # Отложения по видам - в долях m0, как 'Wps dep'
        comp.update({'Asph dissolved': wc['asph_d'], 'Asph flocs': wc['asph_f'], 'Resins': wc['resin'],
                     'Asph dep': (dep['asph_f'] + dep['resin']) / ro_asph_dep / init_m, 'CII': cii,
                     'Asph flocs dep': dep['asph_f'] / ro_asph_dep / init_m,
                     'Resins dep': dep['resin'] / ro_asph_dep / init_m})
    if asphaltenes:
        comp['Asph dep rate'] = solver.kx['qpa']  # асфальтены + смолы, [1/с]
        comp['Asph Ua'] = solver.kx['ua']         # сужение капилляров флокулами u_a = Ua*r^(1/3), [м^(2/3)/с]
    for k in range(N_W):
        comp[f'Wax {k + 1}'] = wc['wax'][..., k]
        comp[f'Wax {k + 1} susp'] = solver.Ws[..., k]
        comp[f'Wax {k + 1} dep'] = dep['wax'][..., k] / ro_p / init_m
    if wax_eos and wax_pressure:
        # Диагностика: объем газа, который выделился бы ниже P_b, на объем нефти (`thermo.pvt`); в уравнения не входит
        from paraphin.thermo.tables import GASV, eos_interp
        comp['Free gas'] = np.vectorize(lambda t, p: eos_interp(GASV, t, p))(solver.T, solver.p)

    # Массы компонентов в пласте, [кг]: в нефтяной фазе и в отложениях
    oil = solver.m * (1.0 - solver.S) * ro_o * volume
    totals = {}
    parts = [(f'wax {k + 1}', wc['wax'][..., k], dep['wax'][..., k]) for k in range(N_W)]
    if NC_T > N_W:
        parts += [(name, wc[f], dep[f]) for name, f in (('asph', 'asph_d'), ('asph flocs', 'asph_f'), ('resins', 'resin'))]
    for name, w, d in parts:
        totals[f'{name} in oil'] = float((oil * w).sum())
        totals[f'{name} deposited'] = float(d.sum() * volume)

    if deposition_kinetics:
        comp.update(_kinetics_fields(solver))
        totals['asph adsorbed'] = float(solver.kx['ga'].sum() * volume)
        totals['resins adsorbed'] = float(solver.kx['gr'].sum() * volume)

    return {'Wps dep': dep['wax'].sum(axis=-1) / ro_p / init_m, 'Composition': comp, 'Totals': totals}


def _kinetics_fields(solver) -> dict:
    """Поля моделей кинетики осаждения (`equations/Deposition.py`).

    'm conductive' - пористость проводящих каналов m0*int r^2*fi/int r^2*fi0, в долях m0: ее, а не m, видит
    томография (гель в тупиковых порах и захваченная нефть для нее - отложение, Sandyga et al. 2020)."""
    from paraphin.geometry import w2_cv
    from paraphin.equations.Deposition import floc_diameter

    kx = solver.kx
    # Скорости потери порового объема этого шага по механизмам, [1/с]: контракт ядра (`equations/Deposition.py`)
    out = {'m conductive': (solver.fi * w2_cv).sum(axis=-1) / solver.integr_r2_fi0, 'Wall cryst rate': kx['qw']}
    if adsorption:
        out['Adsorbed asph'] = kx['ga']    # [кг/м^3 породы]
        out['Adsorbed resins'] = kx['gr']
        out['Asph ads rate'] = kx['qada']
        out['Resins ads rate'] = kx['qadr']
    if wettability:
        # Доля покрытия поверхности асфальтенами - как в `calc_mobility_w`: 0 - водо-, 1 - нефтесмачиваемая
        g_max = kx['gmax']
        out['Wettability omega'] = np.where(g_max > 0.0, np.minimum(kx['ga'] / np.maximum(g_max, 1e-30), 1.0), 0.0)
    if deposit_aging:
        v_gel = kx['vgel']
        wax_v = named(solver.Dep)['wax'].sum(axis=-1) / ro_p
        out['Gel volume'] = v_gel / init_m
        out['Gel wax fraction'] = np.where(v_gel > 0.0, wax_v / np.maximum(v_gel, 1e-30), 0.0)
        out['Gel aging rate'] = kx['qg']
    if wax_kinetics:
        out['Supersaturation'] = np.maximum(kx['weq'] - solver.Ws, 0.0).sum(axis=-1)
    if asph_aggregation:
        wc = named(solver.Wc)
        out['Floc number'] = wc['nflocs']
        d = np.zeros(solver.kx.shape)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                d[i, j] = floc_diameter(wc['asph_f'][i, j], wc['nflocs'][i, j], solver.kin)
        out['Floc diameter'] = d
    if thermal_nonequilibrium:
        out['T rock'] = solver.kx['ts']
    return out
