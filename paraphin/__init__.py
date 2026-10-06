"""Модуль решения задачи двухфазной неизотермической фильтрации с учетом кольматации пласта парафином.

Здесь только сброс дискового кеша numba по хешу исходников пакета; сетка радиусов пор - `paraphin/geometry.py`.
"""
import hashlib
from pathlib import Path


def _drop_stale_numba_cache() -> None:
    """Сброс дискового кеша numba при правке любого исходника пакета.

    Горячие функции помечены njit(cache=True) - без этого компиляция всего графа занимает 25 секунд
    при каждом запуске, что больше самого расчета. Но numba инвалидирует кеш по mtime файла с самой
    функцией, а все, что она вызывает, вшивается в ее машинный код: значения из constants.py - как
    литералы, njit-функции других модулей - телом. Поменяв Nx или dt, без этой проверки мы считали бы
    по старой сетке; поправив `calc_qp_m_k_fi`, гоняли бы старое тело внутри кешированной
    `_equations_loop` из solver.py (проверено: правка не подхватывалась, пока не сброшен кеш).
    Поэтому кеш сбрасывается по хешу всех .py пакета.
    """
    pkg = Path(__file__).parent
    digest = hashlib.md5(b''.join(path.read_bytes() for path in sorted(pkg.rglob('*.py')))).hexdigest()
    stamp = pkg / '__pycache__' / 'constants_hash.txt'
    if stamp.is_file() and stamp.read_text(encoding='ascii') == digest:
        return None

    for cached in pkg.rglob('__pycache__/*.nb[ic]'):
        cached.unlink(missing_ok=True)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(digest, encoding='ascii')


_drop_stale_numba_cache()
