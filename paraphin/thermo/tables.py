r"""Флюид модели нефти для уравнения состояния, калибровки и таблицы по (T, P), которые читают njit-ядра.

Нефть - псевдокомпоненты PR EOS на грамм дегазированной нефти: газ (при `wax_pressure`), растворитель с молярной
массой M_o (вся не парафиновая часть, вместе со смолами), группы парафина `oil_composition` и, для асфальтенов,
отдельный компонент M_a = v_asph*ro_asph. Пока асфальтены не выделены, число молей раствора совпадает с n_L ядра
`Thermo_wax.sle_split`, поэтому таблица ниже подставляется в него без пересчета.

В каждом узле сетки равновесие последовательное (V-L-S): flash живой нефти дает жидкость x_L и растворенный газ,
multi-solid на x_L - равновесную жидкость x_ref, и из f_k^L = f_k^{0,S}
    ln x_k^sat = ln f_k^{0,S}(T, p) - ln p - ln phi_k^L(x_ref, T, p)                                   (1)
- предел растворимости группы k в той же форме, что ln x_k^sat ядра (идеальный раствор, Пойнтинг), но с
неидеальностью из уравнения состояния, калориметрическими T_m, dH Вона без эффективного сдвига (или Coutinho с
твердо-твердым переходом, `eos_melting`) и физическим скачком объема dv (`solids.MultiSolidWax`). Асфальтены -
предельная массовая доля по модели твердой фазы (`solids.NghiemAsphaltene.solubility`); если задана кривая
выпавших `asph_curve`, V_s и kij асфальтены-газ подбираются по ней (`fit_nghiem_curve`).

ponytail: состав в phi заморожен на исходной нефти (в ячейке состав другой); равномерная сетка размазывает
излом у давления насыщения на один шаг по P. Точнее - flash в ячейке, на порядки дороже.

`python -m paraphin.thermo.tables` - свойства псевдокомпонентов, калибровки и время сборки таблиц.
"""
import hashlib
import os
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from numba import njit

from paraphin.constants import (data_type, wax_eos, asph_nghiem, wax_pressure, wax_characterization, eos_T_grid,
                                eos_P_grid, eos_melting, P_ref_wax, init_T, P_bubble, P_onset_asph, asph_curve, R,
                                M_o, ro_o, ro_p,
                                ro_wax_liq,
                                v_asph, ro_asph, sara_asphaltenes, latent_heat, wax_Tm_shift, eos_gas_M, eos_gas_Tc,
                                eos_gas_Pc, eos_gas_omega, eos_asph_Tc, eos_asph_Pc, eos_asph_omega, eos_kij_asph_gas)
from paraphin.layout import N_W
from paraphin.oil_composition import WAX_W0, WAX_M, WAX_TM_K, WAX_L_REL, N_GAS_B, won_tm

T_LO, T_HI, NT = eos_T_grid
T_STEP = (T_HI - T_LO) / (NT - 1)
if wax_pressure:
    P_LO, P_HI, NP = eos_P_grid
else:  # давление в равновесие не входит: одна точка, при которой снята кривая выпадения
    P_LO, P_HI, NP = P_ref_wax, P_ref_wax, 1
P_STEP = (P_HI - P_LO) / (NP - 1) if NP > 1 else 1.0
T_GRID = np.linspace(T_LO, T_HI, NT)
P_GRID = np.linspace(P_LO, P_HI, NP)

# Плавление групп для уравнения состояния - калориметрическое по Won, без эффективного сдвига wax_Tm_shift
WAX_TM_WON = won_tm(WAX_M) if wax_characterization == 'single' else WAX_TM_K - wax_Tm_shift  # [K]
WAX_DH_WON = WAX_L_REL * latent_heat * WAX_M * 1e-3  # [Дж/моль]


def wax_melting(m_wax=WAX_M, how=eos_melting):
    """Плавление групп для multi-solid: (T_m, dH_m, T_tr, dH_tr), у 'won' переход None. 'coutinho' - н-алкан с
    числом атомов по средней молярной массе группы (`characterization.coutinho_nalkane`).

    ponytail: в уравнение энергии по-прежнему идет теплота Вона (носитель Hl); с переходом Coutinho полная теплота
    больше - согласовать, если вариант 'coutinho' станет основным."""
    if how == 'won':
        return WAX_TM_WON, WAX_DH_WON, None, None
    from .characterization import coutinho_nalkane
    return coutinho_nalkane((np.asarray(m_wax) - 2.016) / 14.027)
# Скачок объема при кристаллизации - физический, по плотностям расплава и кристаллов: (v_L - v_S)/v_L = 0.13.
# У эффективной модели свой wax_dv_frac (0.028): он подобран вместе с заниженной эффективной теплотой
WAX_DV_EOS = WAX_M * 1e-3 * (1.0 / ro_wax_liq - 1.0 / ro_p)  # [м^3/моль]
M_ASPH = v_asph * ro_asph * 1e3  # [г/моль]


def oil_fluid(kij_gas=0.0, with_asph=False, gas=wax_pressure, w_wax=WAX_W0, kij_asph=None, m_wax=WAX_M):
    """Псевдокомпоненты нефти для PR EOS: [газ], растворитель, группы парафина (массовые доли w_wax, молярные массы
    m_wax), [асфальтены]; kij_asph - асфальтены-газ (None - `eos_kij_asph_gas`).

    Returns
    -------
    eos, n, mw, ig, iw, ia: PengRobinson, numpy.ndarray, numpy.ndarray, int, int, int
        Уравнение состояния, моли на грамм дегазированной нефти, молярные массы [г/моль], индексы газа
        (-1 - нет), первой группы парафина и асфальтенов (-1 - нет)
    """
    from .characterization import critical_props, fraction_tb, nalkane_sg, nalkane_tb
    from .eos import PengRobinson

    comps = []  # (Tc, Pc, omega, M, моль/г)
    ig = ia = -1
    if gas:
        ig = 0
        comps.append((eos_gas_Tc, eos_gas_Pc, eos_gas_omega, eos_gas_M, N_GAS_B))
    w_a = sara_asphaltenes if with_asph else 0.0
    sg = ro_o / 999.0
    t, p, o = critical_props(fraction_tb(M_o, sg), sg)
    comps.append((float(t), float(p), float(o), M_o, (1.0 - np.sum(w_wax) - w_a) / M_o))
    iw = len(comps)
    comps += list(zip(*critical_props(nalkane_tb(m_wax), nalkane_sg(m_wax)), m_wax, np.asarray(w_wax) / m_wax))
    if with_asph:
        ia = len(comps)
        comps.append((eos_asph_Tc, eos_asph_Pc, eos_asph_omega, M_ASPH, w_a / M_ASPH))
    tc, pc, om, mw, n = np.array(comps, float).T
    kij = np.zeros((tc.size, tc.size))
    if ig >= 0:
        kij[ig, 1:] = kij[1:, ig] = kij_gas
        if ia >= 0:
            kij[ig, ia] = kij[ia, ig] = eos_kij_asph_gas if kij_asph is None else kij_asph
    return PengRobinson(tc, pc, om, kij), n, mw, ig, iw, ia


def calibrate_kij_gas(k_lo=-0.3, k_hi=0.3):
    """kij газ-тяжелые, при котором давление насыщения нефти при init_T равно P_bubble."""
    from scipy.optimize import brentq

    from .eos import bubble_pressure

    t = init_T + 273.15

    def f(k):
        eos, n = oil_fluid(k, gas=True)[:2]
        return np.log(bubble_pressure(eos, t, n / n.sum()) / P_bubble)

    f_lo, f_hi = f(k_lo), f(k_hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(f'kij газ-нефть вне [{k_lo}, {k_hi}]: давление насыщения при init_T от '
                         f'{P_bubble * np.exp(f_lo) / 1e6:.2f} до {P_bubble * np.exp(f_hi) / 1e6:.2f} МПа '
                         f'против P_bubble = {P_bubble / 1e6:.2f} МПа - проверьте Rs_bubble и eos_gas_*')
    return float(brentq(f, k_lo, k_hi, xtol=1e-7))


def live_liquid(eos, n, ig, T, p):
    """Жидкость живой нефти при (T [K], p [Па]): состав и растворенный газ, [моль/г дегазированной нефти].

    Однофазное состояние - всегда жидкость: метка «пар» у flash по корню Z для нефти под давлением неверна."""
    z = n / n.sum()
    if ig < 0:
        return z, 0.0
    from .eos import flash

    beta, x, _ = flash(eos, T, p, z, vapor_only=True)
    if not 0.0 < beta < 1.0:
        return z, float(n[ig])
    return x, (1.0 - beta) * x[ig] * n.sum()


def build_tables(t_grid, p_grid, wax=wax_eos, asph=asph_nghiem, gas=wax_pressure, curve=asph_curve):
    """Таблицы на сетке t_grid [C] x p_grid [Па]; curve - кривая выпавших асфальтенов (`asph_curve`).

    Returns
    -------
    lnxsat, ng, lnwamax, kij_gas: numpy.ndarray(N_W, NT, NP), numpy.ndarray(NT, NP), numpy.ndarray(NT, NP), float
        ln x_k^sat по (1), растворенный газ [моль/г], ln предельной массовой доли растворенных асфальтенов
        (на дегазированную нефть) и подобранный kij газ-тяжелые
    """
    from .solids import MultiSolidWax

    kij = calibrate_kij_gas() if gas else 0.0
    shape = (t_grid.size, p_grid.size)
    lnxsat, ng, lnwamax = np.zeros((N_W,) + shape), np.zeros(shape), np.zeros(shape)
    if wax:
        eos, n, _, ig, iw, _ = oil_fluid(kij, gas=gas)
        groups = np.arange(iw, iw + N_W)
        t_m, dh_m, t_tr, dh_tr = wax_melting()
        ms = MultiSolidWax(eos, groups, t_m, dh_m, WAX_DV_EOS, P_ref_wax, t_tr, dh_tr)
        for a, t_c in enumerate(t_grid):
            t = t_c + 273.15
            for b, p in enumerate(p_grid):
                x, ng[a, b] = live_liquid(eos, n, ig, t, p)
                x_ref = ms.precipitate(t, p, x)[2]
                lnxsat[:, a, b] = (ms.solid_ln_fugacity(t, p) - np.log(p) - eos.ln_phi(t, p, x_ref, 'liquid'))[groups]
    if asph:
        v_s, kij_a = (v_asph, None) if curve is None else fit_nghiem_curve(curve, kij, gas)
        lnwamax = nghiem_lnwamax(t_grid, p_grid, kij, gas, v_s, kij_a)
    return lnxsat, ng, lnwamax, kij


def nghiem_lnwamax(t_grid, p_grid, kij_gas, gas, v_s=v_asph, kij_asph=None):
    """ln предельной массовой доли растворенных асфальтенов (на дегазированную нефть) по модели твердой фазы
    Нгхайема с мольным объемом твердого v_s [м^3/моль]; f_s* - по P_onset_asph при init_T."""
    from .solids import NghiemAsphaltene

    eos, n, mw, ig, _, ia = oil_fluid(kij_gas, with_asph=True, gas=gas, kij_asph=kij_asph)
    model = NghiemAsphaltene(eos, ia, v_s)
    t0 = init_T + 273.15
    model.calibrate_onset(t0, P_onset_asph, live_liquid(eos, n, ig, t0, P_onset_asph)[0])
    dead = np.arange(eos.nc) != ig
    out = np.zeros((t_grid.size, p_grid.size))
    for a, t_c in enumerate(t_grid):
        t = t_c + 273.15
        for b, p in enumerate(p_grid):
            xa, xs = model.solubility(t, p, live_liquid(eos, n, ig, t, p)[0])
            out[a, b] = np.log(xa * M_ASPH / np.sum(xs[dead] * mw[dead]))
    return out


def fit_nghiem_curve(curve, kij_gas, gas):
    """V_s [м^3/моль] и kij асфальтены-газ (None без газа) по кривой выпавших при init_T: МНК по
    w_a0 - w_a^max(p), как выпадение считает ядро (`Components.components_equation`).

    Выпадение задает разность парциального объема асфальтенов в нефти и V_s: подъем давления от P_onset
    пересыщает нефть на exp[(v_a - V_s) dP/RT], и при v_a - V_s в доли процента от v_a выпадает уже половина
    асфальтенов. Поэтому старт - V_s = v_a при P_onset, а kij (форма колокола ниже P_b) - перебором с подбором
    V_s на каждом значении, затем совместное уточнение.
    """
    from scipy.optimize import least_squares

    p, w = np.asarray(curve, float).T
    t0 = init_T + 273.15
    tg = np.array([float(init_T)])

    def resid(v_s, kij_a):
        return np.maximum(sara_asphaltenes - np.exp(nghiem_lnwamax(tg, p, kij_gas, gas, v_s, kij_a)[0]), 0.0) - w

    def fit_vs(kij_a):
        eos, n, _, ig, _, ia = oil_fluid(kij_gas, with_asph=True, gas=gas, kij_asph=kij_a)
        x = live_liquid(eos, n, ig, t0, P_onset_asph)[0]
        h = 1e-3 * P_onset_asph  # парциальный мольный объем v_a = RT d(ln phi_a)/dp + RT/p
        v_a = (R * t0 * (eos.ln_phi(t0, P_onset_asph + h, x, 'liquid')[ia]
                         - eos.ln_phi(t0, P_onset_asph - h, x, 'liquid')[ia]) / (2.0 * h) + R * t0 / P_onset_asph)
        f = least_squares(lambda q: resid(q[0] * v_a, kij_a), [1.01], bounds=([0.9], [1.2]), diff_step=1e-5)
        return f.x[0] * v_a, float(np.sum(f.fun ** 2))

    if not gas:
        return fit_vs(None)[0], None
    kij_best = min(np.linspace(0.0, 0.5, 51), key=lambda k: fit_vs(k)[1])  # минимумов несколько, узкие
    v_s = fit_vs(kij_best)[0]
    f = least_squares(lambda q: resid(q[0] * v_s, q[1]), [1.0, kij_best], bounds=([0.9, 0.0], [1.1, 0.8]),
                      diff_step=1e-5)
    return f.x[0] * v_s, float(f.x[1])


def _cached_tables():
    """Таблицы и kij газ-нефть из дискового кеша, иначе - сборка и запись в кеш.

    Сборка с газом и подбором по кривой асфальтенов - 10-20 с на каждый процесс, а зависят таблицы только от
    исходников пакета (constants.py, состав, thermo/) и версий numpy/scipy. Поэтому ключ кеша - штамп хеша
    исходников, который пишет `paraphin/__init__.py`, плюс версии; файл лежит рядом с кешем numba."""
    pycache = Path(__file__).resolve().parents[1] / '__pycache__'
    stamp = (pycache / 'constants_hash.txt').read_text(encoding='ascii')
    key = hashlib.md5(f'{stamp} {np.__version__} {version("scipy")}'.encode()).hexdigest()[:16]
    path = pycache / f'eos_tables_{key}.npz'
    if path.is_file():
        with np.load(path) as f:
            return f['lnxsat'], f['ng'], f['lnwamax'], float(f['kij_gas']), f['gasv']
    lnxsat, ng, lnwamax, kij_gas = build_tables(T_GRID, P_GRID)
    gasv = np.zeros((NT, NP))  # объем свободного газа на объем нефти V_g/V_L - диагностика выгрузки (`thermo.pvt`)
    if wax_eos and wax_pressure:
        from .pvt import free_gas_table
        gasv = free_gas_table(kij_gas, T_GRID, P_GRID)
    for stale in pycache.glob('eos_tables_*.npz'):
        stale.unlink(missing_ok=True)
    tmp = pycache / f'eos_tmp_{os.getpid()}.npz'  # параллельные процессы копии не видят недописанный файл
    np.savez(tmp, lnxsat=lnxsat, ng=ng, lnwamax=lnwamax, kij_gas=kij_gas, gasv=gasv)
    os.replace(tmp, path)
    return lnxsat, ng, lnwamax, kij_gas, gasv


BUILD_TIME = 0.0  # сборка таблиц или чтение из кеша, [с]
if wax_eos or asph_nghiem:
    _t0 = time.perf_counter()
    LNXSAT, NG, LNWAMAX, KIJ_GAS, GASV = _cached_tables()
    BUILD_TIME = time.perf_counter() - _t0
else:  # ядра таблиц не читают: ветки свернуты на компиляции
    LNXSAT, NG, LNWAMAX, KIJ_GAS = np.zeros((N_W, NT, NP)), np.zeros((NT, NP)), np.zeros((NT, NP)), 0.0
    GASV = np.zeros((NT, NP))
LNXSAT, NG, LNWAMAX, GASV = (a.astype(data_type) for a in (LNXSAT, NG, LNWAMAX, GASV))


@njit(cache=True)
def eos_interp(tab, T, p):
    """Билинейная интерполяция таблицы tab[NT, NP] по температуре T [C] и давлению p [Па].

    ponytail: за пределами сетки - значение на краю; сетку задают eos_T_grid, eos_P_grid."""
    ft = min(max((T - T_LO) / T_STEP, 0.0), NT - 1.0)
    i = min(int(ft), NT - 2)
    a = ft - i
    if NP == 1:
        return tab[i, 0] + a * (tab[i + 1, 0] - tab[i, 0])
    fp = min(max((p - P_LO) / P_STEP, 0.0), NP - 1.0)
    j = min(int(fp), NP - 2)
    b = fp - j
    return ((1.0 - a) * ((1.0 - b) * tab[i, j] + b * tab[i, j + 1])
            + a * ((1.0 - b) * tab[i + 1, j] + b * tab[i + 1, j + 1]))


if __name__ == '__main__':
    from .characterization import fraction_tb, nalkane_sg, nalkane_tb
    from .eos import bubble_pressure

    eos, n, mw, ig, iw, ia = oil_fluid(0.0, with_asph=True, gas=True)
    tb = np.r_[np.nan, fraction_tb(M_o, ro_o / 999.0), nalkane_tb(WAX_M), np.nan]
    sg = np.r_[np.nan, ro_o / 999.0, nalkane_sg(WAX_M), ro_asph / 999.0]
    names = ['газ', 'растворитель'] + [f'группа {k + 1}' for k in range(N_W)] + ['асфальтены']
    print('| компонент | M, г/моль | моль/кг | Tb, K | SG | Tc, K | Pc, МПа | omega |')
    for k, name in enumerate(names):
        print(f'| {name} | {mw[k]:.1f} | {n[k] * 1e3:.4f} | {tb[k]:.1f} | {sg[k]:.4f} | {eos.tc[k]:.1f} | '
              f'{eos.pc[k] / 1e6:.3f} | {eos.omega[k]:.3f} |')
    t0 = time.perf_counter()
    kij = calibrate_kij_gas()
    eos, n = oil_fluid(kij, gas=True)[:2]
    print(f'kij газ-нефть = {kij:.4f}: P_b(init_T) = {bubble_pressure(eos, init_T + 273.15, n / n.sum()) / 1e6:.3f} '
          f'МПа (задано {P_bubble / 1e6:.2f}), {time.perf_counter() - t0:.1f} с')
    t0 = time.perf_counter()
    tg, pg = np.linspace(T_LO, T_HI, eos_T_grid[2]), np.linspace(*eos_P_grid)
    build_tables(tg, pg, wax=True, asph=True, gas=True)
    print(f'таблицы {tg.size}x{pg.size} с газом: {time.perf_counter() - t0:.1f} с')
