"""Препроцессинг массива скважин."""
from paraphin.constants import Nx


def preprocess_matrix_and_wells(wells, wells_buffer):
    """Раскладка скважин из буфера в массив с вычислением индекса неизвестной.

    Профиль разреженности матрицы больше не нужен: уравнение давления собирается сразу в три
    диагонали ленты (`equations/Pressure.py`), поэтому скважина правит `diag[idx]` и `rhs[idx]`
    по одному и тому же индексу `idx = i + j*Nx`.
    """
    for i in range(len(wells)):
        wells[i] = wells_buffer[i]['well']
        wells[i].idx_rhs = wells[i].i + wells[i].j * Nx

    return wells
