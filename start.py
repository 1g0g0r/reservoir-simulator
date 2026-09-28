"""Модуль запуска всего расчета."""
from paraphin.constants import Pw, Po, rw, Twater, Ny, Nx, S_max, day_to_sec
from paraphin.solver import Solver, Bound, TypeBC, DataField


def solve(kin: dict = None):
    """Запуск расчета. kin - параметры кинетики осаждения по именам (`paraphin/kinetics_params.py`), например
    подобранные по керновым опытам (`experiments/params.json`); без него - значения из constants.py."""
    solver = Solver()
    if kin:
        from paraphin.kinetics_params import kin_index
        for name, value in kin.items():
            solver.kin[kin_index(name)] = value

    # Добавление скважин
    solver.add_well(name='Injector', i=0,    j=0,    p=Pw, rw=rw, mult=0.25, is_injector=True, T=Twater)
    solver.add_well(name='Producer', i=Nx-1, j=Ny-1, p=Po, rw=rw, mult=0.25, is_injector=False)

    # Вместо забойного давления скважине можно задать дебит (q > 0). Знак ставится по is_injector.
    # solver.add_well(name='Injector', i=0, j=0, q=50.0 / day_to_sec, rw=rw, mult=0.25, is_injector=True, T=Twater)

    # Добавление ГУ на границе
    # solver.add_bc(field=DataField.Pressure,    bound=Bound.Left,  type_bc=TypeBC.Dirichlet, value=Pw)
    # solver.add_bc(field=DataField.Pressure,    bound=Bound.Right, type_bc=TypeBC.Dirichlet, value=Po)
    # solver.add_bc(field=DataField.Saturation,  bound=Bound.Left,  type_bc=TypeBC.Dirichlet, value=S_max)
    # solver.add_bc(field=DataField.Temperature, bound=Bound.Left,  type_bc=TypeBC.Dirichlet, value=Twater)

    solver.start()


if __name__ == '__main__':
    solve()
