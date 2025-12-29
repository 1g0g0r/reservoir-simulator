""" Модуль конвертирует поля данных taichi в словарь массивов numpy."""
from pickle import dump, HIGHEST_PROTOCOL

from paraphin import fi_0_np
from paraphin.constants import results_path, init_k, init_m, day_to_sec


def save_fields(solver, t: float):
    """Преобразование taichi -> numpy и охранение полей данных в файл формата pkl."""
    wells_data = {}
    wells_accumulated_data = {}
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

        # TODO создаь один раз словарь, а потом только перезаписывать
        data = {
            'Time': t,
            'Pressure': solver.p,
            'Saturation': solver.S,
            'Temperature': solver.T,
            'Wo': solver.Wo,
            'Wp': solver.Wp,
            'Wps': solver.Wps,
            'Wps dep': solver.Wps_dep,
            'qp': solver.qp,
            'm': solver.new_m / init_m,
            'k': solver.new_k / init_k,
            'plots': {'fi_o': fi_0_np, 'fi': solver.fi},
            'Wells': wells_data,
            'Wells_accumulated': wells_accumulated_data,
            # 'Other params': {'KIN': solver.KIN[None], f'T [{x_idx},{y_idx}]': T_np[x_idx, y_idx]}
        }
    else:
        data = {
            'Time': t,
            'Pressure': solver.p,
            'Saturation': solver.S,
            'Temperature': solver.T,
            'Wells': wells_data,
            'Wells_accumulated': wells_accumulated_data,
            # 'Other params': {'KIN': solver.KIN[None], f'T [{0},{0}]': T_np[0, 0]}
        }

    with open(results_path / f'data_{t / day_to_sec}.pkl', 'wb') as file:
        dump(data, file, protocol=HIGHEST_PROTOCOL)
        solver.logger.info("Данные записаны в файл.")
