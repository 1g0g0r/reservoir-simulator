""" Модуль конвертирует поля данных taichi в словарь массивов numpy."""
from pickle import dump, HIGHEST_PROTOCOL

from paraphin import fi_0_np
from paraphin.constants import results_path, init_k, init_m, day_to_sec

cached_data = {}
wells_data = {}
wells_accumulated_data = {}


def save_fields(solver, t: float):
    """Преобразование taichi -> numpy и охранение полей данных в файл формата pkl."""

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
        cached_data['Wps'] = solver.Wps,
        cached_data['Wps dep'] = solver.Wps_dep
        cached_data['qp'] = solver.qp
        cached_data['m'] = solver.new_m / init_m
        cached_data['k'] = solver.new_k / init_k
        cached_data['plots'] = {'fi_o': fi_0_np, 'fi': solver.fi}
        cached_data['Wells'] = wells_data
        cached_data['Wells_accumulated'] = wells_accumulated_data
        # cached_data['Other params'] = {'KIN': solver.KIN[None], f'T [{x_idx},{y_idx}]': solver.T[x_idx, y_idx]}
    else:
        cached_data['Time'] = t
        cached_data['Pressure'] =  solver.p
        cached_data['Saturation'] =  solver.S
        cached_data['Temperature'] =  solver.T
        cached_data['Wells'] = wells_data
        cached_data['Wells_accumulated'] = wells_accumulated_data
        # cached_data['Other params'] = {'KIN': solver.KIN[None], f'T [{0},{0}]': solver.T[0, 0]}

    with open(results_path / f'data_{t / day_to_sec}.pkl', 'wb') as file:
        dump(cached_data, file, protocol=HIGHEST_PROTOCOL)
        solver.logger.info("Данные записаны в файл.")
