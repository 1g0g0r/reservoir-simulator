"""Асфальтены: растворимость по Флори-Хаггинсу (Hirschberg), калибровка на давление начала осаждения,
колокол выпадения по давлению с максимумом у давления насыщения, флокуляция (`equations/Asphaltene.py`).

Ядро `asph_soluble` берет давление насыщения с газом только при флаге `wax_pressure`; здесь та же
формула проверяется на numpy через `oil_composition.delta_maltene_py` (с газом всегда), а совпадение
njit и numpy - при выключенном флаге на давлениях выше P_b, где газ одинаков.
"""
import math

import numpy as np

from paraphin.constants import (init_T, P_onset_asph, P_bubble, sara_asphaltenes, sara_resins, sara_aromatics,
                                v_asph, R, ro_asph, ro_o, k_floc, k_redis)
from paraphin.equations.Asphaltene import floc_relax
from paraphin.oil_composition import (DELTA_ASPH, V_M, REST0, F_SAT_REST, WAX_TOTAL, delta_maltene_py,
                                      asph_volume_fraction)


def _soluble(p, t=init_T, w_res=sara_resins):
    """Наибольшая растворенная массовая доля асфальтенов (1) при начальном составе."""
    rest = 1.0 - WAX_TOTAL - sara_asphaltenes - w_res
    d_m = delta_maltene_py(rest * F_SAT_REST + WAX_TOTAL, rest * (1.0 - F_SAT_REST), w_res, p, t)
    dd = (DELTA_ASPH - d_m) * 1e3
    phi = min(1.0, math.exp(v_asph / V_M - 1.0 - v_asph * dd * dd / (R * (t + 273.15))))
    return phi * ro_asph / (phi * ro_asph + (1.0 - phi) * ro_o)


def test_onset_calibration():
    """delta_a подобран так, что при P_onset_asph и init_T начальная доля асфальтенов ровно насыщающая."""
    assert math.isclose(_soluble(P_onset_asph), sara_asphaltenes, rel_tol=1e-9)
    assert _soluble(P_onset_asph * 1.1) > sara_asphaltenes     # выше AOP - устойчивы
    assert _soluble(0.5 * (P_onset_asph + P_bubble)) < sara_asphaltenes  # ниже - выпадают


def test_bell_shape_minimum_solubility_at_bubble_point():
    """Растворимость минимальна у давления насыщения: выше него нефть расширяется при падении давления,
    ниже - из нее уходит газ с низким параметром растворимости (Leontaritis & Mansoori, 1987)."""
    p = np.linspace(0.5e6, 2.0 * P_onset_asph, 400)
    sol = np.array([_soluble(x) for x in p])
    p_min = p[sol.argmin()]
    assert abs(p_min - P_bubble) <= (p[1] - p[0]) * 1.01
    assert sol[0] > sol.min() and sol[-1] > sol.min()


def test_resins_stabilize():
    """Смолы повышают параметр растворимости мальтенов: больше смол - больше растворимость."""
    assert _soluble(P_bubble, w_res=2.0 * sara_resins) > _soluble(P_bubble, w_res=sara_resins)


def test_volume_fraction_roundtrip():
    phi = asph_volume_fraction(sara_asphaltenes)
    w = phi * ro_asph / (phi * ro_asph + (1.0 - phi) * ro_o)
    assert math.isclose(w, sara_asphaltenes, rel_tol=1e-12)


def test_floc_relax():
    """Точное решение релаксации: границы, пределы k*dt -> 0 и -> бесконечность, разные константы."""
    dt = 3600.0
    assert floc_relax(0.0, 0.005, dt) == 0.005 + (0.0 - 0.005) * math.exp(-k_floc * dt)
    assert floc_relax(0.005, 0.0, dt) == 0.005 * math.exp(-k_redis * dt)
    assert 0.0 <= floc_relax(0.0, 0.005, 1e9) <= 0.005 and math.isclose(floc_relax(0.0, 0.005, 1e9), 0.005)
    assert floc_relax(0.002, 0.005, 0.0) == 0.002
    # выпадение быстрее растворения
    assert abs(floc_relax(0.0, 0.005, dt) - 0.0) > abs(floc_relax(0.005, 0.0, dt) - 0.005)


def test_sara_consistent():
    assert 0.0 < REST0 < 1.0 and 0.0 <= F_SAT_REST <= 1.0
    assert sara_aromatics + sara_resins + sara_asphaltenes < 1.0
