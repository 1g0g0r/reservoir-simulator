"""Равновесие групп парафина (multi-solid) с давлением и газом: `paraphin/equations/Thermo_wax.py`.

Ядра проверяются с выключенными флагами из constants.py: там, где нужен другой режим (давление, газ,
другие группы), проверяется та же формула, записанная здесь на numpy, - сами njit-ядра с другими
константами проверяет `test_composition_flags_on.py` в копии пакета.
"""
import itertools
import math

import numpy as np

from paraphin.constants import MW, M_o, Tm_K, alpha, R, init_Wp, init_Wps
from paraphin.equations.Thermo_wax import sle_split, wat_cell, x_saturation
from paraphin.equations.Wp_balance import _wp_saturated
from paraphin.layout import N_W
from paraphin.oil_composition import (WAX_M, WAX_TM_K, WAX_DH, WAX_DV, WAX_W0, scn_distribution,
                                      lump_groups, won_tm)


def _split_numpy(w, M, tm, dh, dv, t_c, dp=0.0, n_g=0.0, m_o=M_o):
    """Эталон: перебор всех 2^N множеств насыщенных групп, выбор самосогласованного."""
    n = w.size
    t = t_c + 273.15
    x = np.minimum(1.0, np.exp(-dh / R * (1.0 / t - 1.0 / tm) - dv * dp / (R * t)))
    n0 = (1.0 - w.sum()) / m_o + n_g
    for sat in itertools.product((False, True), repeat=n):
        sat = np.array(sat)
        if np.any(sat & (x >= 1.0)):
            continue
        b = 1.0 - x[sat].sum()
        if b <= 0.0:
            continue
        n_l = (n0 + (w / M)[~sat].sum()) / b
        dis = np.where(sat, x * n_l * M, w)
        # самосогласованность: насыщенные не превышают своего количества, ненасыщенные ниже предела
        if np.all(dis[sat] <= w[sat] + 1e-15) and np.all((w / M)[~sat] <= x[~sat] * n_l + 1e-15):
            return dis
    raise AssertionError('нет самосогласованного разбиения')


def test_single_group_is_legacy_formula():
    """Одна группа с прежними MW, Tm, alpha - ровно прежняя растворимость (6.1)-(6.2)."""
    worst = 0.0
    for w in np.linspace(0.0, 0.5, 26):
        for t in np.linspace(-20.0, 90.0, 45):
            legacy = _wp_saturated(w, t)
            dis = _split_numpy(np.array([w]), np.array([MW]), np.array([Tm_K]), np.array([alpha]),
                               np.array([0.0]), t)[0]
            worst = max(worst, abs(dis - legacy) / max(legacy, 1e-300))
    assert worst < 1e-12, worst


def test_kernel_matches_enumeration():
    """njit-ядро `sle_split` с группами из oil_composition совпадает с перебором активных множеств."""
    sus = np.zeros(N_W)
    for scale in (0.2, 1.0, 1.6):
        w = WAX_W0 * scale
        for t in np.linspace(-10.0, 120.0, 27):
            w_dis, w_sus, _ = sle_split(w, t, 1e5, sus)
            ref = _split_numpy(w, WAX_M, WAX_TM_K, WAX_DH, WAX_DV * 0.0, t)
            assert np.allclose(w - sus, ref, rtol=1e-12, atol=1e-15)
            assert np.isclose(w_dis + w_sus, w.sum(), rtol=1e-14)
            assert np.all(sus >= -1e-15)


def test_enumeration_many_groups():
    """Жадный набор насыщенных групп точен и при многих группах с произвольным распределением."""
    rng = np.random.default_rng(1)
    n, M, w = scn_distribution(0.05, 17, 60, 0.3)
    for bounds in ((20, 24, 28, 34, 42), (22, 30, 40)):
        wk, Mk, tmk, _ = lump_groups(n, M, w, bounds, 20.0)
        dh = np.full(wk.size, 50e3)
        for t in rng.uniform(-10.0, 110.0, 15):
            _split_numpy(wk, Mk, tmk, dh, np.zeros(wk.size), t)  # должна найтись самосогласованная точка


def test_wat_is_onset_of_precipitation():
    """WAT в замкнутой форме - ровно граница: чуть выше все растворено, чуть ниже выпадает."""
    sus = np.zeros(N_W)
    w = WAX_W0.copy()
    wat = wat_cell(w, 1e5)
    assert sle_split(w, wat + 1e-6, 1e5, sus)[1] < 1e-12
    assert sle_split(w, wat - 1e-3, 1e5, sus)[1] > 0.0


def test_calibrated_curve_matches_li2024():
    """Группы по умолчанию воспроизводят кривую выпадения Li et al. (2024) в интервале 10-45 C."""
    li = [(40.0, 1.5), (35.0, 5.0), (30.0, 9.5), (25.0, 13.0), (20.0, 16.0), (10.0, 20.5)]
    sus = np.zeros(N_W)
    w = WAX_W0 * 0.2493 / (init_Wp + init_Wps)
    worst = max(abs(100.0 * sle_split(w, t, 1e5, sus)[1] - prec) for t, prec in li)
    assert worst < 1.0, worst
    assert abs(wat_cell(w, 1e5) - 45.65) < 1.5


def test_clausius_clapeyron():
    """Пойнтинг в (1) дает dWAT/dP = WAT*dv/dH для группы, первой достигающей насыщения."""
    w = np.array([0.05, 0.08])
    M = np.array([300.0, 450.0])
    tm = np.array([330.0, 360.0])
    dh = np.array([50e3, 60e3])
    dv = np.array([2e-5, 3e-5])
    n_l = (1.0 - w.sum()) / M_o + (w / M).sum()

    def wat(dp):
        num = dh / R + dv / R * dp
        t_k = num / (dh / R / tm - np.log(w / M / n_l))
        return t_k.max(), t_k.argmax()

    t0, k = wat(0.0)
    slope = (wat(1e5)[0] - wat(-1e5)[0]) / 2e5
    assert math.isclose(slope, t0 * dv[k] / dh[k], rel_tol=1e-6)
    # тот же ответ дает сама формула растворимости: при WAT(dp) группа k ровно насыщена
    t1, _ = wat(1e7)
    x = math.exp(-dh[k] / R * (1 / t1 - 1 / tm[k]) - dv[k] * 1e7 / (R * t1))
    assert math.isclose(x, w[k] / M[k] / n_l, rel_tol=1e-10)


def test_gas_lowers_wat():
    """Растворенный газ добавляет моли в раствор и понижает WAT (эффект состава, Pan et al., 1997)."""
    w = WAX_W0
    wats = []
    for n_g in (0.0, 1e-3, 2e-3):
        n_l = (1.0 - w.sum()) / M_o + n_g + (w / WAX_M).sum()
        t_k = (WAX_DH / R) / (WAX_DH / R / WAX_TM_K - np.log(w / WAX_M / n_l))
        wats.append(t_k.max())
    assert wats[0] > wats[1] > wats[2]


def test_x_saturation_above_melting():
    """Выше эффективной температуры плавления группа растворима неограниченно."""
    for k in range(N_W):
        assert x_saturation(k, WAX_TM_K[k] - 273.15 + 1.0, 1e5) == 1.0


def test_won_correlation():
    """Корреляция Won для н-алканов против измерений Broadhurst (1962): C20 36.8 C, C30 65.8 C, C40 81.5 C."""
    for n, tm_c in ((20, 36.8), (30, 65.8), (40, 81.5)):
        assert abs(won_tm(14.027 * n + 2.016) - 273.15 - tm_c) < 2.0
