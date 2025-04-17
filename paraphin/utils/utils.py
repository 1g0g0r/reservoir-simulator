import re
from pickle import load, PickleError

import numpy as np
import taichi as ti

from paraphin.constants import data_type, results_path
from paraphin.utils.phase_f import pf_o, pf_w


@ti.func
def mid(k_i: data_type, s_i: data_type, mu_o_i: data_type, mu_w_i: data_type,
		k_j: data_type, s_j: data_type, mu_o_j: data_type, mu_w_j: data_type) -> data_type:
    """mid(Ko + Kw)_ij"""
    x = K_o(k_i, s_i, mu_o_i) + K_w(k_i, s_i, mu_w_i)
    y = K_o(k_j, s_j, mu_o_j) + K_w(k_j, s_j, mu_w_j)

    return 2.0 * x * y / (x + y)


@ti.func
def up_kw(k_i: data_type, s_i: data_type, p_i: data_type, mu_o_i: data_type, mu_w_i: data_type,
		  k_j: data_type, s_j: data_type, p_j: data_type, mu_o_j: data_type, mu_w_j: data_type) -> data_type:
    """Значение берется вверх по потоку: up(kw / (ko + kw)"""
    ret = 0.0

    if p_i >= p_j:
        ret = K_w(k_i, s_i, mu_w_i) / (K_w(k_i, s_i, mu_w_i) + K_o(k_i, s_i, mu_o_i))
    else:
        ret = K_w(k_j, s_j, mu_w_j) / (K_w(k_j, s_j, mu_w_j) + K_o(k_j, s_j, mu_o_j))

    return ret


@ti.func
def up_ko(k_i: data_type, s_i: data_type, p_i: data_type, mu_o_i: data_type, mu_w_i: data_type,
		  k_j: data_type, s_j: data_type, p_j: data_type, mu_o_j: data_type, mu_w_j: data_type) -> data_type:
    """Значение берется вверх по потоку: up(ko / (ko + kw)"""
    ret = 0.0

    if p_i >= p_j:
        ret = K_o(k_i, s_i, mu_o_i) / (K_w(k_i, s_i, mu_w_i) + K_o(k_i, s_i, mu_o_i))
    else:
        ret = K_o(k_j, s_j, mu_o_j) / (K_w(k_j, s_j, mu_w_j) + K_o(k_j, s_j, mu_o_j))

    return ret


@ti.func
def K_o(k: data_type, s: data_type, mu_o: data_type) -> data_type:
    return k * pf_o(s) / mu_o


@ti.func
def K_w(k: data_type, s: data_type, mu_w: data_type) -> data_type:
    return k * pf_w(s) / mu_w


def read_pkl_files() -> dict:
    """Считывает содержимое всех бинарных файлов расширения .pkl"""
    def extract_number(_path):
        numbers = re.findall(r'\d+', _path.stem)  # Находим все числа в имени файла
        return int(numbers[0]) if numbers else 0

    # Сортировка данных расчета по времени
    sorted_paths = sorted(list(results_path.glob('*.pkl')), key=extract_number)

    with open(sorted_paths[0], 'rb') as f:
        file = load(f)

    data = {}
    for name, file_data in file.items():
        if name == 'Wells':
            data['Wells'] = {
				'inj': np.array([file_data['inj']]),
				'prod': np.array([file_data['prod']]),
			}

        else:
            data[name] = np.array([file_data])

    if len(sorted_paths) <= 1:
        return data
    for file_path in sorted_paths[1:]:
        try:
            with open(file_path, 'rb') as f:
                file = load(f)
            for name, file_data in file.items():
                if name == 'Wells':
                    data['Wells']['inj']  = np.concatenate((data['Wells']['inj'],  [file_data['inj']]), axis=0)
                    data['Wells']['prod'] = np.concatenate((data['Wells']['prod'], [file_data['prod']]), axis=0)
                else:
                    data[name] = np.concatenate((data[name], [file_data]), axis=0)
        except (PickleError, EOFError) as e:
            print(f"Ошибка при чтении файла {file_path.name}: {e}")

    return data
