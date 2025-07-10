"""Модуль читает файлы расчета в формате .pkl и преобразовывает в удобный формат."""
import re
from pickle import dump, load

import numpy as np

from paraphin.constants import results_path


def read_pkl_files(name: str = 'processed_data.pkl') -> (int, dict):
    """Считывает содержимое всех бинарных файлов расширения .pkl"""
    def extract_number(_path):
        """Находим все числа в имени файла"""
        numbers = re.findall(r'\d+', _path.stem)
        return int(numbers[0]) if numbers else 0

    _processed_data_path = results_path / name
    data = {}
    files_paths = [path for path in results_path.glob('*.pkl') if 'processed_data' not in path.name]  # Все файлы формата pkl
    n_files = len(files_paths)

    # Если данные уже визуализировались, то открывается файл обработанных данных
    if _processed_data_path.exists():
        with open(_processed_data_path, 'rb') as f:
            _n_files, _data = load(f)
            if _n_files == n_files or n_files == 0:
                return _n_files, _data

    sorted_paths = sorted(files_paths, key=extract_number)  # Сортировка данных расчета по времени

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

    for idx, file_path in enumerate(sorted_paths):
        with open(file_path, 'rb') as f:
            for name, file_data in load(f).items():
                if name in ['Wells', 'Other params', 'plots']:
                    for _name, _val in file_data.items():
                        data[name][_name][idx] = file_data[_name]
                else:
                    data[name][idx] = file_data

    with open(results_path / 'processed_data.pkl', 'wb') as f:  # Сохранение обработанных данных
        dump([n_files, data], f)

    return n_files, data

