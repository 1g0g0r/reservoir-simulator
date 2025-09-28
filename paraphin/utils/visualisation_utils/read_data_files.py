"""Обработка и чтение файлов с данными расчета в формате .pkl"""
import re
from pickle import dump, load

import numpy as np
from joblib import Parallel, delayed

from paraphin.constants import results_path, data_path, init_Wp


def read_solution_data(name: str) -> (int, dict):
    """Открывает бинарный файл формата .pkl, содержащий данные расчета"""
    _processed_data_path = data_path / name

    if _processed_data_path.exists():
        with open(_processed_data_path, 'rb') as f:
            n_files, data = load(f)
        return n_files, data

    else:
        raise ValueError(f'Файл {name} отсутствует !!')


def convert_pkl_files():
    """Считывает содержимое всех бинарных файлов расширения .pkl"""
    def extract_number(_path):
        """Находим все числа в имени файла"""
        numbers = re.findall(r'\d+', _path.stem)
        return int(numbers[0]) if numbers else 0

    files_paths = [path for path in results_path.glob('*.pkl') if 'processed_data' not in path.name]  # Все файлы формата pkl
    sorted_paths = sorted(files_paths, key=extract_number)  # Сортировка данных расчета по времени
    n_files = len(files_paths)

    # Если файлов нет, то завершаем выполнение
    if n_files == 0:
        return None

    data = {}
    with open(sorted_paths[0], 'rb') as f:
        for name, file_data in load(f).items():
            if name in ['Wells', 'Other params']:
                data[name] = {_name: np.zeros(n_files) for _name, _val in file_data.items()}
            elif name == 'plots':
                data[name] = {_name: np.zeros((n_files, len(_val))) for _name, _val in file_data.items()}
            elif name == 'Time':
                data[name] = np.zeros(n_files)
            else:
                nx, ny = file_data.shape
                data[name] = np.zeros((n_files, nx, ny))

    # Обработка бинарных файлов формата .pkl
    Parallel(n_jobs=-1, backend='threading')(
        delayed(process_single_file)(idx, data, file_path) for idx, file_path in enumerate(sorted_paths)
    )

    # Сохранение обработанных данных
    with open(data_path / f'Wp={init_Wp}_processed_data.pkl', 'wb') as f:
        dump([n_files, data], f)


def process_single_file(idx, data, file_path):
    """Обработка одного файла."""
    with open(file_path, 'rb') as f:
        try:
            for name, file_data in load(f).items():
                if name in ['Wells', 'Other params', 'plots']:
                    for _name, _val in file_data.items():
                        data[name][_name][idx] = file_data[_name]
                else:
                    data[name][idx] = file_data
        except Exception:
            pass
