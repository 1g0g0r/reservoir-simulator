"""Выгрузка fi по измененным ячейкам: сохранение (`_fi_changed`) -> склейка (`_glue_fi`) -> кривая в ячейке
(`fi_curve`) обязаны вернуть исходное поле в каждой ячейке и в каждый момент. Ничего не читает и не пишет на диск."""
import numpy as np

from paraphin.constants import Nx, Ny, Nr
from paraphin.geometry import fi_0
from paraphin.utils.read_data_files import _glue_fi, fi_curve
from paraphin.utils.save_data_fields import _fi_changed


def test_fi_sparse_roundtrip():
    rng = np.random.default_rng(0)
    n_t = 5
    full = np.ones((n_t, Nx, Ny, Nr)) * fi_0
    for t in range(n_t):  # наборы ячеек разные, в том числе пустой: ячейка уходит из набора и снова равна fi_0
        cells = rng.choice(Nx * Ny, size=3 * t, replace=False)
        full[t].reshape(-1, Nr)[cells] = rng.random((cells.size, Nr))
    plots = _glue_fi([_fi_changed(full[t]) for t in range(n_t)])

    assert plots['cells'].size <= 3 * sum(range(n_t))
    for i in range(Nx):
        for j in range(Ny):
            assert np.array_equal(fi_curve(plots, i, j), full[:, i, j]), (i, j)
