"""Дозапись полей данных очередного временного слоя в общий файл расчета."""
from pickle import dump, HIGHEST_PROTOCOL

from paraphin.constants import layers_file, init_k, init_m

# Точка, в которой снимается кривая fi(r) для графиков: `visualisation._visualize_plots_fi` и `graphs._plot_fi`.
FI_PROBE = (3, 3)


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
