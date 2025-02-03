import re
from pickle import load, PickleError

import numpy as np
import taichi as ti

from paraphin.constants import data_type, results_path, Nx, Ny, hx, hy, area
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
    """Значение берется вверх по потоку: up(kw / (ko + kw)"""
    ret = 0.0

    if p1 >= p2:
        ret = K_w(k1, s1, mu_w1) / (K_w(k1, s1, mu_w1) + K_o(k1, s1, mu_o1))
    else:
        ret = K_w(k2, s2, mu_w2) / (K_w(k2, s2, mu_w2) + K_o(k2, s2, mu_o2))

    return ret


@ti.func
def up_ko(k1: data_type, s1: data_type, p1: data_type, mu_o1: data_type, mu_w1: data_type,
		  k2: data_type, s2: data_type, p2: data_type, mu_o2: data_type, mu_w2: data_type) -> data_type:
    """Значение берется вверх по потоку: up(ko / (ko + kw)"""
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
    def extract_number(_path):
        numbers = re.findall(r'\d+', _path.stem) # Находим все числа в имени файла
        return int(numbers[0]) if numbers else 0

    # Сортировка списка
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


def calculate_temp_data(p, S, T, k, mu_o, mu_w, dt_val, up_kw_val, up_ko_val) -> None:
    """
    Parameters
    ----------
    p: taichi.field(Nx, Ny)
        Давление, [Па]
    S: taichi.field(Nx, Ny)
        Водонасыщенность, [-]
    T: taichi.field(Nx, Ny)
        Температура, [C]
    k: taichi.field(Nx, Ny)
        Проницаемость, [м^2]
    mu_o: taichi.field(Nx, Ny)
        Вязкость нефти, [Па*с]
    mu_w: taichi.field(Nx, Ny)
        Вязкость воды, [Па*с]
    dt_val: taichi.field(Nx, Ny)
        Величина (T_i - T_j) * area / h_ij, [C*м]
    up_kw_val: taichi.field(Nx, Ny)
        Перетоки воды в ячейках, [Па*м]
    up_ko_val: taichi.field(Nx, Ny)
        Перетоки нефти в ячейках, [Па*м]
    """
    @ti.kernel
    def temp_val_loop():
        for i in ti.ndrange(Nx):
            for j in ti.ndrange(Ny):
                # цикл по граням
                arr = [[i + 1, j, hx], [i - 1, j, hx], [i, j + 1, hy], [i, j - 1, hy]]
                kw, ko, dtemp = 0.0, 0.0, 0.0
                for idx in ti.static(ti.ndrange(4)):
                    i1, j1, hij = arr[idx]
                    if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                        value = area * (p[i, j] - p[i1, j1]) / hij * mid(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                                k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1])
                        kw += up_kw(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
                                    k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * value
                        ko += up_ko(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
                                    k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * value
                        dtemp += area * (T[i, j] - T[i1, j1]) / hij
                if (i == Nx - 1 and j == Ny - 1) or (i == 0 and j == 0):
                    kw = 0.0
                    ko = 0.0
                    dtemp = 0.0

                up_kw_val[i, j] = kw
                up_ko_val[i, j] = ko
                dt_val[i, j] = dtemp

    temp_val_loop()
