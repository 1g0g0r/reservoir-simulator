import re
from pickle import dump, load, PickleError

import numpy as np

from paraphin.constants import results_path

_processed_data_path = results_path / f'processed_data.pkl'


def read_pkl_files() -> dict:
    """Считывает содержимое всех бинарных файлов расширения .pkl"""
    # Если данные уже визуализировались, то открывается только файл обработанных данных
    if _processed_data_path.exists():
        with open(_processed_data_path, 'rb') as f:
            data = load(f)
        return data

    def extract_number(_path):
        numbers = re.findall(r'\d+', _path.stem)  # Находим все числа в имени файла
        return int(numbers[0]) if numbers else 0

    # Сортировка данных расчета по времени
    sorted_paths = sorted(list(results_path.glob('*.pkl')), key=extract_number)

    with open(sorted_paths[0], 'rb') as f:
        file = load(f)

    data = {}
    for name, file_data in file.items():
        if name in ['Wells', 'Other params', 'plots']:
            data[name] = {_name: np.array([_val]) for _name, _val in file_data.items()}
        else:
            data[name] = np.array([file_data])

    if len(sorted_paths) <= 1:
        return data
    for file_path in sorted_paths[1:]:
        try:
            with open(file_path, 'rb') as f:
                file = load(f)
            for name, file_data in file.items():
                if name in ['Wells', 'Other params', 'plots']:
                    for _name, _val in file_data.items():
                        data[name][_name] = np.concatenate((data[name][_name],  [file_data[_name]]), axis=0)
                else:
                    data[name] = np.concatenate((data[name], [file_data]), axis=0)
        except (PickleError, EOFError) as e:
            print(f"Ошибка при чтении файла {file_path.name}: {e}")

    with open(_processed_data_path, 'wb') as f:
        # Сохранение обработанных данных
        dump(data, f)

    return data
