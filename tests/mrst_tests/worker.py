"""Рабочий процесс сравнений с MRST: расчет paraphin в копии пакета (`tests/_patched_copy.py`, запуск - common.py).

    python worker.py out.json opts.json

opts: days - время расчета, [сут]; field_days - сутки сохранения полей; flooded - промыть ячейку нагнетательной
(S = S_max); inj_q - дебит нагнетательной, [м^3/с] (иначе она на забойном давлении Pw); T_inj - температура закачки,
[C] (по умолчанию пластовая - вязкости постоянны); perm - путь к .npy с полем проницаемости (Nx, Ny), [м^2];
S0 - начальная водонасыщенность всей области; bc - граничные условия [[поле, граница, тип, значение], ...] по именам
DataField, Bound, TypeBC (например ['Pressure', 'Left', 'Dirichlet', 1.5e7]).
Скважины - как в start.py: нагнетательная в (0, 0), добывающая в (Nx-1, Ny-1) на Po, четверть скважины (mult=0.25).
С bc нагнетательной нет, течение задают границы, а добывающая (без нее Solver.initialize не работает) - с ничтожной
продуктивностью, как в experiments/core_runner.py; тогда «закачка» и «отбор» - поток через правую границу полосы
(Nx = 1): отток последней ячейки `cells_Q_out`, по фазам - в долях подвижности слоя, с которым считались перетоки.
Пишет на каждом шаге t, dt, дебиты (q > 0 - закачка), забойное давление нагнетательной и температуру ячейки
добывающей, [K], поля p, sw, T, [K], - в первый шаг не раньше суток field_days (ячейка (i, j) - элемент i + j*Nx).
"""
import json
import sys
from time import perf_counter

import numpy as np


def main(out: str, opts: dict) -> None:
    from paraphin.constants import Nx, Ny, Pw, Po, rw, init_T, S_max, day_to_sec
    from paraphin.solver import Solver, Bound, DataField, TypeBC

    solver = Solver()
    T_inj = opts.get('T_inj', init_T)
    strip = opts.get('bc') is not None
    if strip:
        for field, bound, kind, value in opts['bc']:
            solver.add_bc(field=DataField[field], bound=Bound[bound], type_bc=TypeBC[kind], value=value)
    elif opts.get('inj_q'):
        solver.add_well(name='Injector', i=0, j=0, q=opts['inj_q'], rw=rw, mult=0.25, is_injector=True, T=T_inj)
    else:
        solver.add_well(name='Injector', i=0, j=0, p=Pw, rw=rw, mult=0.25, is_injector=True, T=T_inj)
    solver.add_well(name='Producer', i=Nx - 1, j=Ny - 1, p=Po, rw=rw, mult=1e-12 if strip else 0.25, is_injector=False)
    solver.initialize()
    if opts.get('perm'):
        solver.k[:] = np.load(opts['perm'])
    if opts.get('S0') is not None:
        solver.S[:] = opts['S0']
    if opts.get('flooded'):  # ячейка нагнетательной заранее промыта: подвижность закачки не зависит от соглашения
        solver.S[0, 0] = S_max
    prod = solver.wells[solver._producer]
    inj = None if strip else solver.wells[0]

    r = {key: [] for key in ('t', 'dt', 'q_inj', 'qw_prod', 'qo_prod', 'bhp_inj', 'T_prod', 'field_t', 'p', 'sw', 'T')}
    pending = sorted(opts['field_days'])
    t, t_end, elapsed = 0.0, opts['days'] * day_to_sec, 0.0
    while t < t_end:
        step_dt = solver.dt
        t += step_dt
        tt = perf_counter()
        solver.upd_time_step(t)
        if len(r['t']):  # первый шаг - компиляция
            elapsed += perf_counter() - tt
        r['t'].append(t)
        r['dt'].append(step_dt)
        if strip:  # поток через правую границу полосы; приток через левую ему равен (несжимаемость)
            q_out = float(solver.cells_Q_out[Nx - 1, Ny - 1])
            lw, lo = solver.lam_w[Nx - 1, Ny - 1], solver.lam_o[Nx - 1, Ny - 1]
            f_w = float(lw / (lw + lo))
            r['q_inj'].append(q_out)
            r['qw_prod'].append(f_w * q_out)
            r['qo_prod'].append((1.0 - f_w) * q_out)
            r['bhp_inj'].append(0.0)
        else:
            r['q_inj'].append(float(inj.q[2]))
            r['qw_prod'].append(float(-prod.q[1]))
            r['qo_prod'].append(float(-prod.q[0]))
            r['bhp_inj'].append(float(inj.p))
        r['T_prod'].append(float(solver.T[Nx - 1, Ny - 1]) + 273.15)
        while pending and t >= pending[0] * day_to_sec:
            pending.pop(0)
            r['field_t'].append(t)
            r['p'].append(solver.p.ravel(order='F').tolist())
            r['sw'].append(solver.S.ravel(order='F').tolist())
            r['T'].append((solver.T.ravel(order='F') + 273.15).tolist())
    r['elapsed'] = elapsed
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(r, f)


if __name__ == '__main__':
    with open(sys.argv[2], encoding='utf-8') as f:
        main(sys.argv[1], json.load(f))
