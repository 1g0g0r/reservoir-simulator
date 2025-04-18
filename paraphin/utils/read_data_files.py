import re
from pickle import load, PickleError

import numpy as np

from paraphin.constants import results_path


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
