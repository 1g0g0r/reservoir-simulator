"""Эталон регрессии «все новые флаги выключены»: полный прогон решателя и снимок всех полей.

Эталон снят кодом ДО внедрения детального состава нефти (асфальтены, группы парафинов, гель,
давление в WAT). Все новые механизмы включаются флагами-константами `constants.py`, и с выключенными
флагами расчет обязан совпадать с прежним побитово - это проверяет `test_regression_flags_off.py`.

Постановка повторяет `test_mass_balance._make_solver` (закачка при 5 C, чтобы парафин выпадал и блок
кольматации реально работал), но заморожена здесь: правка тестов баланса не должна сдвигать эталон.

Пересъем эталона (только если сознательно меняется базовая физика):
    git worktree add ../baseline <коммит>; cd ../baseline; python -m tests.regression_baseline --write
и скопировать tests/test_data/regression_flags_off.npz обратно.
"""
import argparse
import platform
import sys
from pathlib import Path

import numpy as np

BASELINE = Path(__file__).parent / 'test_data' / 'regression_flags_off.npz'
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
        out[f'well{w}_q'] = np.array(well.q, copy=True)
        out[f'well{w}_Q'] = np.array(well.Q, copy=True)
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
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(BASELINE, __meta__=np.array(repr(meta)), **fields)
        print('Эталон записан:', BASELINE, 'шагов:', fields['dt_hist'].size)
    else:
        print('Шагов:', fields['dt_hist'].size, 'КИН:', float(fields['KIN']))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
