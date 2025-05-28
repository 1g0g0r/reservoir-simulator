import re
from pickle import dump, load, PickleError

import numpy as np

from paraphin.constants import results_path

_processed_data_path = results_path / f'processed_data.pkl'


def read_pkl_files() -> (int, dict):
    """Считывает содержимое всех бинарных файлов расширения .pkl"""
    def extract_number(_path):
        numbers = re.findall(r'\d+', _path.stem)  # Находим все числа в имени файла
        return int(numbers[0]) if numbers else 0

    data = {}
    files_paths = [path for path in results_path.glob('*.pkl') if path != _processed_data_path]  # Все файлы формата pkl
    n_files = len(files_paths)

    # Если данные уже визуализировались, то открывается файл обработанных данных
    if _processed_data_path.exists():
        with open(_processed_data_path, 'rb') as f:
            _n_files, _data = load(f)

            if _n_files == n_files:
                return _n_files, _data

    # Сортировка данных расчета по времени
    sorted_paths = sorted(files_paths, key=extract_number)

    with open(sorted_paths[0], 'rb') as f:
        for name, file_data in load(f).items():
            if name in ['Wells', 'Other params', 'plots']:
                data[name] = {_name: np.array([_val]) for _name, _val in file_data.items()}
            else:
                data[name] = np.array([file_data])

    for file_path in sorted_paths[1:]:
        try:
            with open(file_path, 'rb') as f:
                for name, file_data in load(f).items():
                    if name in ['Wells', 'Other params', 'plots']:
                        for _name, _val in file_data.items():
                            data[name][_name] = np.concatenate((data[name][_name],  [file_data[_name]]), axis=0)
                    else:
                        data[name] = np.concatenate((data[name], [file_data]), axis=0)
        except (PickleError, EOFError) as e:
            print(f"Ошибка при чтении файла {file_path.name}: {e}")

    # Сохранение обработанных данных
    with open(_processed_data_path, 'wb') as f:
        dump([n_files, data], f)

    return n_files, data
