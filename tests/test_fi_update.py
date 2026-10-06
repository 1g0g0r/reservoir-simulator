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

from paraphin.geometry import r1, fi_0, n_pass, dr_cv, w2_cv, plug_cv
from paraphin.constants import Nr, init_m, init_k
from paraphin.equations.Qp_m_k_fi import _update_fi, _calculate_integrals, calc_qp_m_k_fi

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


def test_blocking_keeps_pore_volume():
    """Блокирование без сужения: k падает, а пористость - только на пробки, объем канала остается тупиковым.

    Убыль пористости обязана совпасть с объемом пробок (один кристалл на канал, `plug_cv`) по
    фактически выбывшим из fi каналам - и быть много меньше объема самих каналов (`w2_cv`).
    Попутно: веса w2_cv дают тот же интеграл r^2*fi, что `_calculate_integrals`.
    """
    fi, new_fi, Ur, Ub, a, b = _fields(u0=0.0, b0=1e3 / dt)
    zero = np.zeros_like(Ur)
    _, i2, i4 = _calculate_integrals(fi, zero, 0, 0)
    assert abs(i2 - (w2_cv * fi_0).sum()) < 1e-12 * i2

    def cell(value):
        return np.full((1, 1), value)

    new_qp1, new_qp2, new_k, new_m = cell(0.0), cell(0.0), cell(0.0), cell(0.0)
    calc_qp_m_k_fi(0, 0, cell(0.0), cell(0.05), cell(0.1), cell(init_m), cell(init_k), fi, Ur, Ub, i2, i4,
                   a, b, new_qp1, new_qp2, new_fi, new_k, new_m, dt)

    gone = fi_0 - new_fi[0, 0]  # выбывшие из проводящих каналы
    blocked = init_m * (w2_cv * gone).sum() / i2
    plugs = init_m * (plug_cv * gone).sum() / i2
    loss = init_m - new_m[0, 0]
    assert new_k[0, 0] < 0.9 * init_k, 'блокирование не снизило проницаемость'
    assert new_qp1[0, 0] == 0.0, 'без сужения осадка на стенках нет'
    assert abs(loss - plugs) < 1e-12 * loss, 'пористость убыла не на объем пробок'
    assert abs(loss - new_qp2[0, 0] * dt) < 1e-12 * loss
    assert loss < 0.1 * blocked, f'пробки {loss:.3e} не малы по сравнению с каналами {blocked:.3e}'


if __name__ == '__main__':
    test_fi_conserved_across_r_pass()
    test_fi_positive_under_strong_blocking()
    test_blocking_keeps_pore_volume()
    print('OK')


def test_update_fi_rows_matches_legacy():
    """Общая прогонка по профилям u = lim*Ur, b = lim*Ub (`Pore_bundle.update_fi_rows`, ее зовут сиблинги) совпадает с
    прежней `_update_fi(limiter=lim)` до округления, а одно сужение флокулами u = ua*r^(1/3) (без парафина) сохраняет
    число капилляров sum(fi*dr_cv) - они только сползают к малым r."""
    from paraphin.equations.Pore_bundle import update_fi_rows
    from paraphin.geometry import cbrt_r1, n_pass_a

    fi, new_fi, Ur, Ub, a, b = _fields(-1e-9, 1e-3)
    _update_fi(new_fi, fi, Ur, Ub, 0, 0, a, b, dt, 0.7)
    legacy = new_fi[0, 0].copy()
    update_fi_rows(new_fi, fi, 0, 0, 0.7 * Ur[0, 0], 0.7 * Ub[0, 0], a, b, dt)
    assert np.allclose(new_fi[0, 0], legacy, rtol=1e-13, atol=0.0)

    fi, new_fi, Ur, Ub, a, b = _fields(0.0, 0.0)
    u = np.where(np.arange(Nr) >= n_pass_a, -1e-6 * cbrt_r1, 0.0)
    update_fi_rows(new_fi, fi, 0, 0, u, np.zeros(Nr), a, b, dt)
    assert np.isclose((new_fi[0, 0] * dr_cv).sum(), (fi[0, 0] * dr_cv).sum(), rtol=1e-12)
    assert np.all(new_fi[0, 0] >= 0.0)
