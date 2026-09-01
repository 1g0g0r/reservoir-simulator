"""Модуль конвертирует поля данных класса в pkl."""
from pickle import dump, HIGHEST_PROTOCOL

from paraphin import fi_0_np
from paraphin.constants import layers_file, init_k, init_m

cached_data = {}
wells_data = {}
wells_accumulated_data = {}


def save_fields(solver, t: float):
    """Дозапись полей данных очередного временного слоя в общий файл расчета.

    Слои пишутся подряд в один открытый файл, а не в отдельный pkl на каждое сохранение: это убирает
    открытия файла при каждом сохранении, а также долгую пост-обработку множества файлов.
    Память при этом не растет: `dump` отдает байты в ОС и возвращается, в памяти живет ровно один слой.
    """
    for i in range(solver.n_wells):
        well_name, well = solver._wells_names[i], solver.wells[i]
        q_value, Q_value = well.q, well.Q
        wells_data.update({
            f'{well_name}_oil': q_value[0], f'{well_name}_water': q_value[1],
            f'{well_name}_total': q_value[2], f'{well_name}_eta': well.eta
        })
        wells_accumulated_data.update({
            f'{well_name}_Q_oil': Q_value[0], f'{well_name}_Q_water': Q_value[1], f'{well_name}_Q_total': Q_value[2]
        })

    if solver._paraphin:
        x_idx = 0  # int(Nx / 2)
        y_idx = 0  # int(Ny / 2)

        cached_data['Time'] = t
        cached_data['Pressure'] = solver.p
        cached_data['Saturation'] = solver.S
        cached_data['Temperature'] = solver.T
        cached_data['Wo'] = solver.Wo
        cached_data['Wp'] = solver.Wp
        cached_data['Wps'] = solver.Wps
        cached_data['Wps dep'] = solver.Wps_dep
        cached_data['qp'] = solver.qp
        cached_data['m'] = solver.new_m / init_m
        cached_data['k'] = solver.new_k / init_k
        cached_data['plots'] = {'fi_o': fi_0_np, 'fi': solver.fi}
        cached_data['Wells'] = wells_data
        cached_data['Wells_accumulated'] = wells_accumulated_data
        # cached_data['Other params'] = {'KIN': solver.KIN, f'T [{x_idx},{y_idx}]': solver.T[x_idx, y_idx]}
    else:
        cached_data['Time'] = t
        cached_data['Pressure'] =  solver.p
        cached_data['Saturation'] =  solver.S
        cached_data['Temperature'] =  solver.T
        cached_data['Wells'] = wells_data
        cached_data['Wells_accumulated'] = wells_accumulated_data
        cached_data['Other params'] = {'KIN': solver.KIN,
                                       f'S [{0},{0}]': solver.S[0, 0], f'S [{-1},{-1}]': solver.S[-1, -1],
                                       f'P [{0},{0}]': solver.p[0, 0], f'p [{-1},{-1}]': solver.p[-1, -1],
        }

    if solver._layers_file is None:
        solver._layers_file = open(layers_file, 'wb')

    dump(cached_data, solver._layers_file, protocol=HIGHEST_PROTOCOL)
    solver.logger.info("Данные записаны в файл.")
