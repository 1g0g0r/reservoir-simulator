"""Эталон регрессии модели по умолчанию: полный прогон решателя и снимок всех полей.

Модель по умолчанию - один псевдокомпонент парафина (`wax_characterization = 'single'`), механизмы выключены.
Эталон `regression_flags_off_<хеш>.npz` снят кодом единой модели; с ним расчет обязан совпадать побитово
(`test_regression_flags_off.py`). Эталон `legacy_single_<хеш>.npz` - тот же прогон прежнего однокомпонентного кода
(до объединения с моделью АСПО): с ним сравнение до 1e-8, это проверка, что частный случай общей модели - прежняя модель.

Постановка повторяет `test_mass_balance._make_solver` (закачка при 5 C, чтобы парафин выпадал и блок
кольматации реально работал), но заморожена здесь: правка тестов баланса не должна сдвигать эталон.

Побитовое совпадение возможно только в том же окружении (платформа, версии numba/llvmlite/numpy, python),
поэтому эталонов несколько - по одному на окружение, имя файла содержит хеш `metadata()` (`baseline_path`).
Эталон для нового окружения снимают кодом ДО правок (worktree на Windows падает на длинных именах в `resources/`):
    git archive <коммит> paraphin tests | tar -x -C <tmp>; cd <tmp>; python -m tests.regression_baseline --write
и копируют tests/test_data/regression_flags_off_<хеш>.npz из <tmp> обратно. Пересъем существующего - только если
сознательно меняется базовая физика.
"""
import argparse
import hashlib
import platform
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / 'test_data'
T_END_DAYS = 100.0
T_INJECTION = 5.0

FIELDS = ('p', 'S', 'T', 'T_0', 'm', 'k', 'Wo', 'Wp', 'Wps', 'fi', 'h_sloy', 'Ur', 'Ub', 'qp1', 'qp2',
          'mu_o', 'mu_w', 'E_ff', 'grad_p')


def metadata() -> dict:
    """Все, от чего зависит побитовый результат: сетка, версии компилятора и библиотек, платформа."""
    import numba
    import llvmlite
    from paraphin.constants import Nx, Ny, Nr, init_Wp

    return {'Nx': Nx, 'Ny': Ny, 'Nr': Nr, 'init_Wp': init_Wp,
            'numpy': np.__version__, 'numba': numba.__version__, 'llvmlite': llvmlite.__version__,
            'machine': platform.machine(), 'libc': ' '.join(platform.libc_ver()),
            'python': platform.python_version()}


def _env_hash(meta: dict = None) -> str:
    meta = metadata() if meta is None else meta
    return hashlib.sha256(repr(sorted(meta.items())).encode()).hexdigest()[:12]


def baseline_path(meta: dict = None) -> Path:
    """Файл эталона окружения `meta` (по умолчанию - текущего)."""
    return DATA / f'regression_flags_off_{_env_hash(meta)}.npz'


def legacy_path(meta: dict = None) -> Path:
    """Эталон прежнего однокомпонентного кода для окружения `meta`."""
    return DATA / f'legacy_single_{_env_hash(meta)}.npz'


def run() -> dict:
    """Прогон на T_END_DAYS суток, возвращает поля конца расчета, скважины и историю шагов."""
    from paraphin.constants import Nx, Ny, Pw, Po, rw, day_to_sec
    from paraphin.solver import Solver

    solver = Solver()
    solver.add_well(name='Injector', i=0, j=0, p=Pw, rw=rw, mult=0.25, is_injector=True, T=T_INJECTION)
    solver.add_well(name='Producer', i=Nx - 1, j=Ny - 1, p=Po, rw=rw, mult=0.25, is_injector=False)
    solver.initialize()
    solver.upd_time_step(0.0)

    t, t_end, dts = 0.0, T_END_DAYS * day_to_sec, []
    while t < t_end:
        dts.append(solver.dt)
        t += solver.dt
        solver.upd_time_step(t)

    if solver._results_file is not None:
        solver._results_file.close()

    out = {name: np.array(getattr(solver, name), copy=True) for name in FIELDS}
    for w, well in enumerate(solver.wells):
        out[f'well{w}_q'] = np.array([well.q_o, well.q_w, well.q_t])  # прежняя раскладка эталонов: (нефть, вода, сумма)
        out[f'well{w}_Q'] = np.array([well.Q_o, well.Q_w, well.Q_t])
        out[f'well{w}_p'] = np.array(well.p)
        out[f'well{w}_eta'] = np.array(well.eta)
    out['KIN'] = np.array(solver.KIN)
    out['dt_hist'] = np.array(dts)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--write', action='store_true', help='записать эталон в tests/test_data')
    args = parser.parse_args()

    fields = run()
    meta = metadata()
    if args.write:
        path = baseline_path(meta)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, __meta__=np.array(repr(meta)), **fields)
        print('Эталон записан:', path, 'шагов:', fields['dt_hist'].size)
    else:
        print('Шагов:', fields['dt_hist'].size, 'КИН:', float(fields['KIN']))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
