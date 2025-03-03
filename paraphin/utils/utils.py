import re
from pickle import load, PickleError

import numpy as np
import taichi as ti

from paraphin.constants import data_type, results_path, Nx, Ny, hx, hy, h
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

    return -ret


@ti.func
def up_ko(k_i: data_type, s_i: data_type, p_i: data_type, mu_o_i: data_type, mu_w_i: data_type,
		  k_j: data_type, s_j: data_type, p_j: data_type, mu_o_j: data_type, mu_w_j: data_type) -> data_type:
    """Значение берется вверх по потоку: up(ko / (ko + kw)"""
    ret = 0.0

    if p_i >= p_j:
        ret = K_o(k_i, s_i, mu_o_i) / (K_w(k_i, s_i, mu_w_i) + K_o(k_i, s_i, mu_o_i))
    else:
        ret = K_o(k_j, s_j, mu_o_j) / (K_w(k_j, s_j, mu_w_j) + K_o(k_j, s_j, mu_o_j))

    # TODO почему отток положительный, приток отрицательный

    return -ret


@ti.func
def K_o(k: data_type, s: data_type, mu_o: data_type) -> data_type:
    return k * pf_o(s) / mu_o


@ti.func
def K_w(k: data_type, s: data_type, mu_w: data_type) -> data_type:
    return k * pf_w(s) / mu_w


def calculate_flows_in_cells(p, S, T, k, mu_o, mu_w, dt_val, up_kw_val, up_ko_val) -> None:
    """
    Вычисление потоков в ячейках.

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
                arr = [[i + 1, j, hx, hy*h], [i - 1, j, hx, hy*h], [i, j + 1, hy, hx*h], [i, j - 1, hy, hx*h]]
                kw, ko, dtemp = 0.0, 0.0, 0.0
                for idx in ti.static(ti.ndrange(4)):
                    i1, j1, hij, areaij = arr[idx]
                    if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                        value = areaij * (p[i, j] - p[i1, j1]) / hij * mid(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                                k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1])
                        kw += up_kw(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
                                    k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * value
                        ko += up_ko(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
                                    k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * value
                        dtemp += areaij * (T[i, j] - T[i1, j1]) / hij
                #         if i < 2 and j < 2:
                #             print([i,j], [i1, j1], kw, ko)
                # if i < 2 and j < 2:
                #     print()

                up_kw_val[i, j] = kw
                up_ko_val[i, j] = ko
                dt_val[i, j] = dtemp

                # TODO сравнить перетоки с MRST

    temp_val_loop()
    # print()


def read_pkl_files() -> dict:
    """Считывает содержимое всех бинарных файлов (расширение .pkl)."""
    def extract_number(_path):
        numbers = re.findall(r'\d+', _path.stem) # Находим все числа в имени файла
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
