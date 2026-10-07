"""Неоднородная проницаемость: paraphin против MRST incomp и ad-blackoil на логнормальном поле k.

Постановка start.py (скважины на забойном давлении) на сетке 40x40, но k - логнормальное поле: ln k нормален со
средним ln(init_k) и СКО 1 (k от ~0.02 до ~2 Д), радиус корреляции 20 м (гауссово сглаживание белого шума,
`perm_field` - детерминирован зерном). Однородная k прячет ошибки ориентации осей и осреднения на гранях (решение
симметрично относительно диагонали); здесь оба кода получают одно поле: paraphin - в solver.k (worker.py),
MRST - в rock.perm. Проводимость грани у обоих - гармоническое среднее, скважины по k своей ячейки.

MRST-часть - mrst_heterogeneous.m (MATLAB/Octave; без аргументов - ручной запуск, возвращает результат).

    python -m pytest tests/mrst_tests/test_heterogeneous.py -s       # ~2 мин
    python -m tests.mrst_tests.test_heterogeneous --mrst              # эталон: 2 прогона octave-cli, ~25 мин
"""
import argparse

import numpy as np
import pytest

from tests.mrst_tests import common as c

TEST = 'test_heterogeneous'
N, SEED, SIGMA, CORR = 40, 1, 1.0, 20.0
DT = {'incomp': 0.25, 'ad': 1.0}


def perm_field() -> np.ndarray:
    """k[i, j], [м^2]: гауссов шум (RandomState - одинаков во всех версиях numpy), сглаженный ядром exp(-r^2/CORR^2)
    в спектре (периодически), нормированный на СКО ln k = SIGMA вокруг ln(init_k)."""
    from paraphin.constants import init_k, X_min, X_max
    noise = np.random.RandomState(SEED).standard_normal((N, N))
    f = np.fft.fftfreq(N, d=(X_max - X_min) / N)
    kernel = np.exp(-(np.pi * CORR) ** 2 * (f[:, None] ** 2 + f[None, :] ** 2))
    g = np.real(np.fft.ifft2(np.fft.fft2(noise) * kernel))
    return init_k * np.exp(SIGMA * (g - g.mean()) / g.std())


def physics() -> dict:
    return c.physics(perm_seed=SEED, perm_sigma=SIGMA, perm_corr=CORR)


def cases() -> dict:
    perm = perm_field()
    return {f'{s}_n{N}_hetero': c.mrst_case(physics(), s, N, c.DAYS, dt, c.FIELD_DAYS, perm=perm)
            for s, dt in DT.items()}


@pytest.fixture(scope='module')
def diffs():
    runs = c.load_reference(TEST, physics())
    c.RUNS.mkdir(parents=True, exist_ok=True)
    perm_path = c.RUNS / f'{TEST}_perm.npy'
    np.save(perm_path, perm_field())
    runs.update(c.paraphin_runs({f'paraphin_n{N}_hetero': (
        f'mrst_n{N}', {'Nx, Ny': f'{N}, {N}', **c.PATCH_TWO_PHASE},
        {'days': c.DAYS, 'field_days': c.FIELD_DAYS, 'perm': str(perm_path)})}))
    names = [f'paraphin_n{N}_hetero', *cases()]
    d = c.report(TEST, runs, {name: names[0] for name in names[1:]}, c.FIELD_DAYS)
    c.plot_series(f'{TEST}_rates.png', [('Приемистость нагнетательной, м³/сут', 'q_inj', names, 1.0),
                                        ('Дебит нефти, м³/сут', 'q_o', names, 1.0),
                                        ('Обводненность, доли', 'eta', names, 1.0)], runs)
    fields = [('sw', 1.0, [0.2, 0.3, 0.4, 0.5, 0.6, 0.68], 'S_w'), ('p', 1e-5, list(range(55, 150, 5)), 'p, бар')]
    for name in names[1:]:
        c.plot_isolines(f'{TEST}_isolines_{name}.png', runs, names[0], name, N, c.FIELD_DAYS[:3], fields,
                        f'{N}x{N}, логнормальная k (СКО ln k = {SIGMA:g})')
        c.plot_maps(f'{TEST}_maps_{name}.png', runs, names[0], name, N, c.FIELD_DAYS.index(500),
                    [('sw', 1.0, 'Blues', 'S_w'), ('p', 1e-5, 'Purples', 'p, бар')], f'{N}x{N}, логнормальная k, 500 сут')
    plot_perm()
    return d


def plot_perm() -> None:
    """Поле проницаемости, общее для paraphin и MRST, [мД] в логарифмической шкале."""
    from matplotlib.colors import LogNorm
    plt = c._plt()
    fig, ax = plt.subplots(figsize=(4.6, 3.8), layout='constrained')
    im = ax.imshow(perm_field().T / 9.869233e-16, origin='lower', extent=(0, 200, 0, 200), cmap='Greens', norm=LogNorm())
    fig.colorbar(im, ax=ax, label='k, мД')
    ax.set_title(f'Проницаемость {N}x{N}: СКО ln k = {SIGMA:g}, корреляция {CORR:g} м', loc='left', fontsize=8)
    ax.grid(False)
    ax.set_xlabel('x, м')
    ax.set_ylabel('y, м')
    fig.savefig(c.FIG / f'{TEST}_perm.png', dpi=150)
    plt.close(fig)


def test_rates(diffs):
    c.assert_limits(diffs, {'q_inj': 0.02, 'q_o': 0.03, 'Qo': 0.005})


def test_water_cut(diffs):
    c.assert_limits(diffs, {'eta_mean': 0.005, 't_bt': 10})


def test_saturation_field(diffs):
    c.assert_limits(diffs, {'dS_mean': 0.005, 'dS_max': 0.1})


def test_pressure_field(diffs):
    c.assert_limits(diffs, {'dp_rms': 1.0, 'dp_max': 2.0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--mrst', action='store_true', help='снять эталон MRST в Octave')
    if parser.parse_args().mrst:
        c.save_reference(TEST, physics(), cases())
