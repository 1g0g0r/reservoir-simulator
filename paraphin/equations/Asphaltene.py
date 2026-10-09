"""Асфальтены: растворимость (Флори-Хаггинс), флокуляция и скорость сужения капилляров флокулами.

Асфальтены в нефти - коллоид: наноагрегаты и их кластеры, пептизированные смолами (модель Йена-Маллинса,
Mullins et al., Energy Fuels 2012, 26:3986; Leontaritis & Mansoori, SPE 16258, 1987). Устойчивость
определяется сродством к растворителю - параметром растворимости мальтенов delta_m. В полимерной теории
Флори-Хаггинса предельная объемная доля растворенных асфальтенов (Hirschberg et al., SPEJ 1984, 24:283)

    phi_a^max = exp[v_a/v_m - 1 - v_a*(delta_a - delta_m)^2/(R*T)].                     (1)

delta_m падает при снижении давления к P_b (нефть расширяется) и растет ниже P_b (уходит газ с низким
delta), поэтому растворимость минимальна у давления насыщения - отсюда колоколообразная зависимость
выпадения от давления и зона осаждения не на забое, а там, где P ~ P_b (Darabi et al., SPE 169121, 2014;
Tabzar et al., Oil Gas Sci. Technol. 2018, 73:51). Смолы с высоким delta повышают delta_m: чем их больше, тем ниже давление
начала осаждения.

Выпадение - не мгновенное: флокулы растут за часы-сутки (Maqbool, Balgoa & Fogler, Energy Fuels 2009,
23:3681), поэтому доля флокул релаксирует к равновесной с константой k_floc, а обратное растворение
медленнее (k_redis): частичная необратимость, отмеченная Leontaritis & Mansoori (1987).
Осаждение флокул в порах - `calc_velocity_asph` и `Qp_m_k_fi.calc_qp_m_k_fi`.

С флагом `asph_nghiem` предел растворимости берется из таблицы модели твердой фазы Nghiem на уравнении
состояния Пенга-Робинсона (`thermo/tables.py`), откалиброванной по тому же давлению начала осаждения.
Если задана кривая выпавших `asph_curve`, по ней подбираются v_a в (1) или V_s и kij асфальтены-газ Нгхайема.
"""
import math

import numpy as np
from numba import njit

from paraphin.constants import (R, ro_o, ro_asph, delta_sat, delta_aro, delta_res, delta_gas, v_gas,
                                c_oil_comp, beta_oil, P_ref_wax, k_floc, k_redis, k_B, D_asph, Lk, S_max,
                                min_Wps_bound, asph_nghiem)
from paraphin.layout import IA_F
from paraphin.oil_composition import DELTA_ASPH, V_ASPH, V_M, V_LIQ
from paraphin.thermo.tables import LNWAMAX, eos_interp
from .Thermo_wax import n_gas

# v_a - v_asph или подобранный по кривой выпавших asph_curve (`oil_composition._fit_v_asph`)
_VA_VM_1 = V_ASPH / V_M - 1.0
_VA_R = V_ASPH / R
# Броуновская диффузия флокулы по Стоксу-Эйнштейну: D_a = k_B*T/(3*pi*mu*d_a), как для кристаллов парафина
_DIFF_A = k_B / (3.0 * np.pi * D_asph)
_So_max = 1.0 - S_max


@njit(cache=True)
def delta_maltene(w_sat, w_aro, w_res, p, T):
    """Параметр растворимости мальтенов, [МПа^0.5]. Та же формула, что `oil_composition.delta_maltene_py`.

    Жидкие псевдокомпоненты смешиваются по массовым долям, результат пропорционален плотности (правило
    «одной трети»: Buckley et al., Pet. Sci. Technol. 1998, 16:251; Wang & Buckley, Energy Fuels 2001,
    15:1004) - сжатие поднимает delta, нагрев снижает. Растворенный газ смешивается по объемной доле.
    """
    w_liq = w_sat + w_aro + w_res
    d_liq = (w_sat * delta_sat + w_aro * delta_aro + w_res * delta_res) / w_liq
    d_liq *= 1.0 + c_oil_comp * (p - P_ref_wax) - beta_oil * (T - 25.0)
    vg = n_gas(T, p) * v_gas
    vl = w_liq * V_LIQ
    phi_g = vg / (vg + vl)
    return (1.0 - phi_g) * d_liq + phi_g * delta_gas


@njit(cache=True)
def asph_soluble(w_sat, w_aro, w_res, p, T):
    """Наибольшая массовая доля растворенных асфальтенов в нефтяной фазе по (1), [-]."""
    if asph_nghiem:
        # ponytail: состав на исходной нефти - смолы и насыщенные ячейки (аргументы) на предел не влияют
        return min(1.0, math.exp(eos_interp(LNWAMAX, T, p)))
    dd = (DELTA_ASPH - delta_maltene(w_sat, w_aro, w_res, p, T)) * 1e3  # МПа^0.5 -> Па^0.5
    arg = _VA_VM_1 - _VA_R * dd * dd / (T + 273.15)
    phi = 1.0 if arg >= 0.0 else math.exp(arg)
    return phi * ro_asph / (phi * ro_asph + (1.0 - phi) * ro_o)


@njit(cache=True)
def floc_relax(w_af, w_af_eq, dt):
    """Доля флокул через шаг dt: точное решение dw/dt = -k*(w - w_eq), безусловно устойчиво.
    Выпадение идет с k_floc, растворение флокул - с меньшей k_redis."""
    k = k_floc if w_af < w_af_eq else k_redis
    return w_af_eq + (w_af - w_af_eq) * math.exp(-k * dt)


@njit(cache=True)
def calc_velocity_asph(i, j, S, T, Um_r2, Wc, mu_p, new_kx):
    """Коэффициент скорости сужения капилляров флокулами асфальтенов: u_a(r) = Ua*r^(1/3), [м^(2/3)/с].

    Та же формула типа Левека, что для кристаллов парафина (`Velocity_h.calc_velocities_h`):
    ua = -So*w_af*(2*um*D_a^2/(r*Lk))^(1/3), um = um_r2*r^2, - с броуновской диффузией флокулы D_a и
    вязкостью жидкой основы `mu_p` (гель на подвижность частиц не влияет). Это поверхностное осаждение
    модели Wang & Civan (SPE 64991, 2001; JERT 2005, 127:318), выведенное из течения в капилляре.
    Как и Ur, считается на новый слой (`new_kx['ua']`) и в `calc_qp_m_k_fi` используется на следующем шаге.
    """
    w_af = Wc[i, j, IA_F]
    if w_af > min_Wps_bound:
        So = max(0.0, 1.0 - S[i, j] - _So_max)
        d_a = _DIFF_A * (T[i, j] + 273.15) / mu_p[i, j]
        new_kx[i, j].ua = -So * w_af * (Um_r2[i, j] * 2.0 * d_a * d_a / Lk) ** (1.0 / 3.0)
    else:
        new_kx[i, j].ua = 0.0
