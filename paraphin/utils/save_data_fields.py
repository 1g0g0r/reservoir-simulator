""" Модуль конвертирует поля данных taichi в словарь массивов numpy."""
import numpy as np
import taichi as ti
from taichi.lang.impl import grouped
from pickle import dump

from paraphin import fi_0_np
from paraphin.constants import Nx, Ny, Nr, results_path, init_k, init_m, day_to_sec, np_data_type

p_np = np.zeros((Nx, Ny), dtype=np_data_type)
S_np = np.zeros((Nx, Ny), dtype=np_data_type)
T_np = np.zeros((Nx, Ny), dtype=np_data_type)
Wo_np = np.zeros((Nx, Ny), dtype=np_data_type)
Wp_np = np.zeros((Nx, Ny), dtype=np_data_type)
Wps_np = np.zeros((Nx, Ny), dtype=np_data_type)
Wps_dep_np = np.zeros((Nx, Ny), dtype=np_data_type)
qp_np = np.zeros((Nx, Ny), dtype=np_data_type)
new_m_np = np.zeros((Nx, Ny), dtype=np_data_type)
new_k_np = np.zeros((Nx, Ny), dtype=np_data_type)
fi_np = np.zeros(Nr, dtype=np_data_type)


def save_fields(solver, t: float):
	"""Преобразование taichi -> numpy и охранение полей данных в файл формата pkl."""
	wells_data = {}
	for i in range(solver.n_wells):
		well_name, well = solver._wells_names[i], solver.wells[i]
		q_value, Q_value = well.q, well.Q
		wells_data.update({
			f'{well_name}_oil': q_value[0], f'{well_name}_water': q_value[1],
			f'{well_name}_total': q_value[2], f'{well_name}_eta': well.eta,
			# f'{well_name}_Q_oil': Q_value[0], f'{well_name}_Q_water': Q_value[1], f'{well_name}_Q_total': Q_value[2]
		})

	if solver._paraphin:
		x_idx = int(Nx / 2)
		y_idx = int(Ny / 2)
		_loop_with_paraphin_data(solver, x_idx, y_idx, fi_np, p_np, S_np, T_np, Wo_np, Wp_np,
								 Wps_np, Wps_dep_np, qp_np, new_m_np, new_k_np)
		data = {
			'Time': t,
			'Pressure': p_np,
			'Saturation': S_np,
			'Temperature': T_np,
			'Wo': Wo_np,
			'Wp': Wp_np,
			'Wps': Wps_np,
			'Wps dep': Wps_dep_np,
			'qp': qp_np,
			'm': new_m_np / init_m,
			'k ': new_k_np / init_k,
			'plots': {'fi_o': fi_0_np, 'fi': fi_np},
			'Wells': wells_data,
			# 'Other params': {
			#   'KIN': solver.KIN[None],
			# 	f'Wps [{x_idx},{y_idx}]': Wps_np[x_idx, y_idx],
			# 	f'Wp [{x_idx},{y_idx}]': Wp_np[x_idx, y_idx],
			# 	f'Wo [{x_idx, y_idx}]': Wo_np[x_idx, y_idx],
			# 	f'k [{x_idx},{y_idx}]': new_k_np[x_idx, y_idx] / init_k,
			# 	f'm [{x_idx},{y_idx}]': new_m_np[x_idx, y_idx] / init_m,
			# 	f'qp [{x_idx},{y_idx}]': qp_np[x_idx, y_idx]
			# }
		}
	else:
		_loop(solver, p_np, S_np, T_np)
		data = {
			'Time': t,
			'Pressure': p_np,
			'Saturation': S_np,
			'Temperature': T_np,
			'Wells': wells_data,
			# 'Other params': {
			# 	'KIN': solver.KIN[None],
			# 	f'T [{0},{0}]': T_np[0, 0],
			# }
		}

	with open(results_path / f'data_{t / day_to_sec}.pkl', 'wb') as f:
		dump(data, f)
		solver.logger.info("Данные записаны в файл.")


@ti.kernel
def _loop(solver: ti.template(), _p_np: ti.types.ndarray(), _S_np: ti.types.ndarray(), _T_np: ti.types.ndarray()):
	for I in grouped(solver.S):
		_p_np[I] = solver.p[I]
		_S_np[I] = solver.S[I]
		_T_np[I] = solver.T[I]


@ti.kernel
def _loop_with_paraphin_data(solver: ti.template(), x_idx: int, y_idx: int, _fi_np: ti.types.ndarray(),
							 _p_np: ti.types.ndarray(), _S_np: ti.types.ndarray(), _T_np: ti.types.ndarray(),
							 _Wo_np: ti.types.ndarray(), _Wp_np: ti.types.ndarray(), _Wps_np: ti.types.ndarray(),
							 _Wps_dep_np: ti.types.ndarray(), _qp_np: ti.types.ndarray(), _new_m_np: ti.types.ndarray(),
							 _new_k_np: ti.types.ndarray()):
	for I in grouped(solver.S):
		_p_np[I] = solver.p[I]
		_S_np[I] = solver.S[I]
		_T_np[I] = solver.T[I]
		_Wo_np[I] = solver.Wo[I]
		_Wp_np[I] = solver.Wp[I]
		_Wps_np[I] = solver.Wps[I]
		_Wps_dep_np[I] = solver.Wps_dep[I]
		_qp_np[I] = solver.qp[I]
		_new_m_np[I] = solver.new_m[I]
		_new_k_np[I] = solver.new_k[I]

	for i in ti.ndrange(Nr):
		_fi_np[i] = solver.fi[x_idx, y_idx, i]
