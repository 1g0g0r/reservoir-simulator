"""Обработка и чтение файлов с данными расчета в формате .pkl"""
from pickle import dump, load, UnpicklingError

import numpy as np

from paraphin.constants import layers_file, data_path, case_name


def read_solution_data(name: str) -> (int, dict):
    """Открывает бинарный файл формата .pkl, содержащий данные расчета"""
    _processed_data_path = data_path / name

    if _processed_data_path.exists():
        with open(_processed_data_path, 'rb') as f:
            n_files, data = load(f)
        return n_files, data

    else:
        raise ValueError(f'Файл {name} отсутствует !!')


def convert_pkl_files(n_layers: int = None):
    """Склейка слоев расчета в массивы с ведущей осью по времени.

    Слои лежат подряд в одном файле (`save_fields` дозаписывает их по ходу расчета), поэтому
    читаются последовательно: ни сортировки по именам файлов, ни разбора времени из имени,
    ни параллельного чтения не нужно.

    `n_layers` знает `Solver` и передает его сам. Без него число слоев считается первым проходом
    по файлу - это лишнее чтение, поэтому так делается только при вызове из визуализации.
    """
    if not layers_file.is_file():
        return None

    if n_layers is None:
        n_layers = _count_layers()
    if n_layers == 0:
        return None

    data = {}
    with open(layers_file, 'rb') as f:
        first = load(f)
        for name, value in first.items():
            data[name] = _allocate(value, n_layers)
        _store(data, 0, first)

        n_read = 1
        for idx in range(1, n_layers):
            try:
                _store(data, idx, load(f))
            except (EOFError, UnpicklingError):
                break  # запись оборвалась - расчет прервали в момент сохранения
            n_read = idx + 1

    # Слоев может оказаться меньше обещанного, если запись последнего не дошла до диска
    if n_read < n_layers:
        n_layers = n_read
        data = {name: _trim(value, n_layers) for name, value in data.items()}

    with open(data_path / f'{case_name}_processed_data.pkl', 'wb') as f:
        dump([n_layers, data], f)


def _count_layers() -> int:
    """Число слоев в файле. Индекса у pickle нет, поэтому только проходом по записям."""
    n = 0
    with open(layers_file, 'rb') as f:
        while True:
            try:
                load(f)
            except (EOFError, UnpicklingError):
                return n
            n += 1


def _allocate(value, n_layers: int):
    """Массив под все слои одного поля: ведущая ось - время, остальные оси как у самого поля.

    Форма берется из первого слоя, а не задается по имени поля, поэтому одинаково работает и для
    скаляров (`Time`, дебиты скважин), и для полей `(Nx, Ny)`, и для распределений по радиусам пор.
    """
    if isinstance(value, dict):
        return {_name: np.zeros((n_layers,) + np.shape(_val)) for _name, _val in value.items()}

    return np.zeros((n_layers,) + np.shape(value))


def _store(data: dict, idx: int, layer: dict) -> None:
    """Раскладка одного слоя по массивам."""
    for name, value in layer.items():
        if isinstance(value, dict):
            for _name, _val in value.items():
                data[name][_name][idx] = _val
        else:
            data[name][idx] = value


def _trim(value, n_layers: int):
    """Обрезка массивов до фактического числа прочитанных слоев."""
    if isinstance(value, dict):
        return {_name: _val[:n_layers] for _name, _val in value.items()}

    return value[:n_layers]
