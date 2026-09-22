"""Проверки схемы для функции пор по размерам (`_update_fi`).

1. `test_fi_conserved_across_r_pass` - без блокирования интеграл fi по сетке (сумма fi*dr_cv)
   обязан сохраняться:
   при u_r = 0 ниже r_pass и u_r < 0 выше поток через r = 0 и приток через r = r_max нулевые.
   Прежняя запись с ветвлением по знаку u_ij теряла поток на стыке r_pass (узел n_pass-1 не видел
   приток из узла n_pass), и сумма убывала.

2. `test_fi_positive_under_strong_blocking` - при b*dt >> 1 fi не уходит в минус: слагаемое
   блокирования неявное, матрица - M-матрица.
"""
import numpy as np

from paraphin import r1, fi_0, n_pass, dr_cv
from paraphin.constants import Nr
from paraphin.equations.Qp_m_k_fi import _update_fi

dt = 86400.0 / 20


def _fields(u0: float, b0: float):
    fi = np.broadcast_to(fi_0, (1, 1, Nr)).copy()
    idx = np.arange(Nr)
    Ur = np.where(idx >= n_pass, -u0 * np.cbrt(r1 / r1[-1]), 0.0).reshape(1, 1, Nr)
    Ub = np.where(idx < n_pass, b0, 0.0).reshape(1, 1, Nr)
    return fi, np.empty_like(fi), Ur, Ub, np.zeros(Nr), np.zeros(Nr)


def test_fi_conserved_across_r_pass():
    """Сужение без блокирования: интеграл fi сохраняется, в том числе через стык r_pass."""
    fi, new_fi, Ur, Ub, a, b = _fields(u0=0.3 * dr_cv.min() / dt, b0=0.0)  # Куранта 0.3 за шаг
    for _ in range(200):
        _update_fi(new_fi, fi, Ur, Ub, 0, 0, a, b, dt)
        fi, new_fi = new_fi, fi

    assert fi[0, 0, n_pass - 1] > fi_0[n_pass - 1], 'поток из узла n_pass не дошел до n_pass-1'
    assert abs((fi * dr_cv).sum() - (fi_0 * dr_cv).sum()) < 1e-12 * (fi_0 * dr_cv).sum()


def test_fi_positive_under_strong_blocking():
    """b*dt = 1e3: fi узких капилляров обнуляется, но не становится отрицательной."""
    fi, new_fi, Ur, Ub, a, b = _fields(u0=0.3 * dr_cv.min() / dt, b0=1e3 / dt)
    _update_fi(new_fi, fi, Ur, Ub, 0, 0, a, b, dt)

    assert new_fi.min() >= 0.0
    # Неявное блокирование делит узел на (1 + b*dt); сверху в последний блокируемый узел еще притекает
    # из сужающихся капилляров не больше Куранта (0.3) от их fi - отсюда множитель 1.3.
    assert new_fi[0, 0, :n_pass].max() < 1.3 * fi_0.max() / (1.0 + 1e3)


if __name__ == '__main__':
    test_fi_conserved_across_r_pass()
    test_fi_positive_under_strong_blocking()
    print('OK')
