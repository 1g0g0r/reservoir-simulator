from pickle import load, PickleError

import numpy as np
import taichi as ti

from paraphin.constants import data_type, results_path
from paraphin.utils.phase_f import pf_o, pf_w


@ti.func
def mid(k1: data_type, s1: data_type, mu_o1: data_type, mu_w1: data_type,
		k2: data_type, s2: data_type, mu_o2: data_type, mu_w2: data_type) -> data_type:
    """mid(Ko + Kw)_ij"""
    x = K_o(k1, s1, mu_o1) + K_w(k1, s1, mu_w1)
    y = K_o(k2, s2, mu_o2) + K_w(k2, s2, mu_w2)
    return 2.0 * x * y / (x + y)


@ti.func
def up_kw(k1: data_type, s1: data_type, p1: data_type, mu_o1: data_type, mu_w1: data_type,
		  k2: data_type, s2: data_type, p2: data_type, mu_o2: data_type, mu_w2: data_type) -> data_type:
    """up(kw / (ko + kw)"""
    ret = 0.0

    if p1 >= p2:
        ret = K_w(k1, s1, mu_w1) / (K_w(k1, s1, mu_w1) + K_o(k1, s1, mu_o1))
    else:
        ret = K_w(k2, s2, mu_w2) / (K_w(k2, s2, mu_w2) + K_o(k2, s2, mu_o2))

    return ret


@ti.func
def up_ko(k1: data_type, s1: data_type, p1: data_type, mu_o1: data_type, mu_w1: data_type,
		  k2: data_type, s2: data_type, p2: data_type, mu_o2: data_type, mu_w2: data_type) -> data_type:
    """up(ko / (ko + kw)"""
    ret = 0.0

    if p1 >= p2:
        ret = K_o(k1, s1, mu_o1) / (K_w(k1, s1, mu_w1) + K_o(k1, s1, mu_o1))
    else:
        ret = K_o(k2, s2, mu_o2) / (K_w(k2, s2, mu_w2) + K_o(k2, s2, mu_o2))

    return ret


@ti.func
def K_o(k: data_type, s: data_type, mu_o: data_type) -> data_type:
    return k * pf_o(s) / mu_o


@ti.func
def K_w(k: data_type, s: data_type, mu_w: data_type) -> data_type:
    return k * pf_w(s) / mu_w


def read_pkl_files() -> dict:
    """Считывает содержимое всех бинарных файлов (расширение .pkl)."""
    files_paths = list(results_path.glob('*.pkl'))

    with open(files_paths[0], 'rb') as f:
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

    for file_path in files_paths[1:]:
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
