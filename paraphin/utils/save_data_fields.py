"""Дозапись полей данных очередного временного слоя в общий файл расчета."""
from pickle import dump, HIGHEST_PROTOCOL

from paraphin.constants import layers_file, init_k, init_m

# Точка, в которой снимается кривая fi(r) для графиков: `visualisation._visualize_plots_fi` и `graphs._plot_fi`.
FI_PROBE = (5, 5)


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

    layer = {
        'Time': t,
        'Pressure': solver.p,
        'Saturation': solver.S,
        'Temperature': solver.T
    }

    if solver._paraphin:
        x_idx, y_idx = FI_PROBE
        layer.update({
            'Wo': solver.Wo,
            'Wp': solver.Wp,
            'Wps': solver.Wps,
            'Wps dep': solver.Wps_dep,
            'qp': solver.qp,
            'm': solver.m / init_m,
            'k': solver.k / init_k,
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

    if solver._layers_file is None:
        solver._layers_file = open(layers_file, 'wb')

    dump(layer, solver._layers_file, protocol=HIGHEST_PROTOCOL)
    solver.logger.info("Данные записаны в файл.")
