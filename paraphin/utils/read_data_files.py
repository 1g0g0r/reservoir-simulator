"""Обработка и чтение файлов с данными расчета в формате .pkl"""
from pathlib import Path
from pickle import dump, load, UnpicklingError

import numpy as np

from paraphin.constants import init_Wp, heat_losses, wax_characterization, Nx, Ny
from paraphin.geometry import fi_0

# Пути и имена выгрузки: корень - каталог над пакетом (в копии пакета - `outputs/.flags_on/<имя>`)
outputs_path = Path(__file__).parents[2] / 'outputs'
_heat_tag = {0: '_noheat', 1: '_lauwerier'}.get(heat_losses, '')
_comp_tag = '' if wax_characterization == 'single' else '_comp'
case_name = f'Wp={init_Wp}{_heat_tag}{_comp_tag}'
results_path = outputs_path / f'results_{case_name}'
layers_file = results_path / 'layers.pkl'  # промежуточные слои расчета, один файл на весь расчет
data_path = outputs_path / 'data'

# Ячейка, в которой графики без выбранной ячейки строят кривую fi(r) (`fi_curve`).
# Прижата к сетке: на одномерном керне (Nx = 1) точки (3, 3) нет.
FI_CELL = (min(3, Nx - 1), min(3, Ny - 1))


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
        # fi пишется только в измененных ячейках, число их от слоя к слою разное: склеивается отдельно (`_glue_fi`)
        fi = [first.pop('plots')] if 'plots' in first else None
        for name, value in first.items():
            data[name] = _allocate(value, n_layers)
        _store(data, 0, first)

        n_read = 1
        for idx in range(1, n_layers):
            try:
                layer = load(f)
            except (EOFError, UnpicklingError):
                break  # запись оборвалась - расчет прервали в момент сохранения
            if fi is not None:
                fi.append(layer.pop('plots'))
            _store(data, idx, layer)
            n_read = idx + 1

    # Слоев может оказаться меньше обещанного, если запись последнего не дошла до диска
    if n_read < n_layers:
        n_layers = n_read
        data = {name: _trim(value, n_layers) for name, value in data.items()}
    if fi is not None:
        data['plots'] = _glue_fi(fi)

    with open(data_path / f'{case_name}_processed_data.pkl', 'wb') as f:
        dump([n_layers, data], f)


def _glue_fi(layers: list) -> dict:
    """fi слоев (`save_data_fields._fi_changed`) - на общий набор ячеек: 'cells' - номера всех ячеек, где fi хоть раз
    отошел от fi_0, 'fi' - (n_layers, n_cells, Nr); ячейка, которой на слое нет в его наборе, на нем равна fi_0."""
    cells = np.unique(np.concatenate([layer['cells'] for layer in layers]))
    fi = np.empty((len(layers), cells.size, fi_0.size), dtype=layers[0]['fi'].dtype)
    fi[:] = fi_0
    for t, layer in enumerate(layers):
        fi[t, np.searchsorted(cells, layer['cells'])] = layer['fi']
    return {'cells': cells, 'fi': fi}


def fi_curve(plots: dict, i: int, j: int) -> np.ndarray:
    """Кривые fi(r) в ячейке (i, j) во все моменты, (n_layers, Nr), из склейки `_glue_fi`; массивы в `plots` могут
    быть и memmap. Ячейки нет в наборе - fi в ней не менялся, это fi_0."""
    cells, fi = plots['cells'], plots['fi']
    cell = i * Ny + j  # номер в fi.reshape(-1, Nr) поля (Nx, Ny, Nr)
    k = int(np.searchsorted(cells, cell))
    if k < cells.size and cells[k] == cell:
        return fi[:, k]
    return np.broadcast_to(fi_0, (fi.shape[0], fi_0.size))


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
    Оттуда же dtype: поля пишутся во float32, скаляры - float64 (`save_data_fields._compact`).
    """
    if isinstance(value, dict):
        return {_name: _allocate(_val, n_layers) for _name, _val in value.items()}

    return np.zeros((n_layers,) + np.shape(value), dtype=np.asarray(value).dtype)


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
