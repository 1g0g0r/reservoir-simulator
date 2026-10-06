"""Уравнение состояния (`paraphin/thermo`): порт из пакета research, корреляции, калибровки, таблицы ядер.

Якоря порта сняты кодом `C:\\postgraduate\\research` (PR 1978 для omega > 0.491) на тех же данных NIST.
Флаги не нужны: таблицы строятся прямым вызовом `build_tables` на грубой сетке, а njit-ядра с включенными
флагами проверяет `test_composition_flags_on.py` в копии пакета.
"""
import numpy as np
import pytest

from paraphin.constants import M_o, init_T, P_bubble, P_onset_asph, P_ref_wax, sara_asphaltenes
from paraphin.layout import N_W
from paraphin.oil_composition import WAX_M, WAX_W0, scn_distribution, sle_split_np
from paraphin.thermo import characterization as ch
from paraphin.thermo.eos import PengRobinson, bubble_pressure, flash
from paraphin.thermo.solids import IdealSolidSolutionWax, MultiSolidWax, NghiemAsphaltene
from paraphin.thermo.tables import (M_ASPH, WAX_DH_WON, WAX_DV_EOS, WAX_TM_WON, NT, T_GRID, build_tables, eos_interp,
                                    live_liquid, oil_fluid)

# C1, C3, nC10, nC16, nC20 (NIST): Tc, Pc, omega; у трех н-алканов - T_fus, dH_fus
TC = [190.564, 369.83, 617.70, 723.0, 768.0]
PC = [4.5992e6, 4.2480e6, 2.11e6, 1.40e6, 1.11e6]
OMEGA = [0.0114, 0.1521, 0.4884, 0.7174, 0.9069]
T_FUS = np.array([243.5, 291.3, 309.6])
DH_FUS = np.array([28710.0, 53360.0, 69880.0])


def _model_oil():
    eos = PengRobinson(TC, PC, OMEGA)
    z = np.array([0.3, 0.1, 0.3, 0.2, 0.1])
    is_former = np.array([False, False, True, True, True])
    t_fus, dh = np.r_[np.nan, np.nan, T_FUS], np.r_[np.nan, np.nan, DH_FUS]
    return eos, z, MultiSolidWax(eos, [2, 3, 4], T_FUS, DH_FUS), IdealSolidSolutionWax(is_former, t_fus, dh)


def test_flash_matches_research():
    eos = PengRobinson([TC[0], TC[2]], [PC[0], PC[2]], [OMEGA[0], OMEGA[2]])
    z = np.array([0.4, 0.6])
    beta, x, y = flash(eos, 350.0, 5e6, z)
    assert beta == pytest.approx(0.2508670789, abs=1e-9)
    assert np.abs(beta * y + (1.0 - beta) * x - z).max() < 1e-12
    f_l = np.log(x) + eos.ln_phi(350.0, 5e6, x, 'liquid')
    f_v = np.log(y) + eos.ln_phi(350.0, 5e6, y, 'vapor')
    assert np.abs(f_l - f_v).max() < 1e-10


def test_wax_models_match_research():
    eos, z, ms, ss = _model_oil()
    wat = ms.wat(1e6, z)
    assert wat == pytest.approx(285.615785, abs=1e-5)
    assert ss.wat(z) == pytest.approx(288.593485, abs=1e-5)  # твердый раствор выпадает раньше
    s, L, x = ms.precipitate(wat - 10.0, 1e6, z)
    assert s[4] == pytest.approx(0.06837719, abs=1e-8) and s[:4].max() == 0.0  # выпал один nC20
    assert np.abs(L * x + s - z).max() < 1e-12
    lnf_l = ms.liquid_ln_fugacity(wat - 10.0, 1e6, x)
    assert lnf_l[4] == pytest.approx(ms.solid_ln_fugacity(wat - 10.0, 1e6)[4], abs=1e-9)
    assert ss.precipitate(wat - 10.0, z)[0].sum() >= s.sum()  # переоценка у твердого раствора


@pytest.mark.parametrize('k', [0, 1, 2])
def test_pure_wat_is_t_fus(k):
    """dG_fus = 0 при T_fus: WAT чистого компонента от уравнения состояния и давления не зависит."""
    eos = PengRobinson(TC[2 + k:3 + k], PC[2 + k:3 + k], OMEGA[2 + k:3 + k])
    ms = MultiSolidWax(eos, [0], T_FUS[k:k + 1], DH_FUS[k:k + 1])
    assert ms.wat(1e5, np.ones(1)) == pytest.approx(T_FUS[k], abs=0.01)


@pytest.mark.parametrize('M, tb, sg, tc, pc, omega, tol_omega', [
    (142.285, 447.3, 0.734, 617.7, 2.11e6, 0.4884, 0.10),
    (282.556, 616.9, 0.7886, 768.0, 1.11e6, 0.9069, 0.05),
    # ветвь Kesler-Lee T_br >= 0.8: у тяжелых н-алканов omega занижена (12 % у C30)
    (422.83, 722.9, 0.810, 844.0, 0.80e6, 1.307, 0.15),
])
def test_correlations_nist(M, tb, sg, tc, pc, omega, tol_omega):
    t, g = ch.nalkane_tb(M), ch.nalkane_sg(M)
    assert t == pytest.approx(tb, rel=0.002) and g == pytest.approx(sg, rel=0.003)
    assert ch.fraction_tb(M, sg) == pytest.approx(tb, rel=0.006)
    c_t, c_p, o = ch.critical_props(t, g)
    assert c_t == pytest.approx(tc, rel=0.01) and c_p == pytest.approx(pc, rel=0.05)
    assert float(o) == pytest.approx(omega, rel=tol_omega)


def test_gamma_alpha_1_is_exponential():
    _, _, w_exp = scn_distribution()
    _, _, w_gam = scn_distribution(alpha=1.0)
    assert np.allclose(w_gam, w_exp, rtol=1e-12, atol=0.0)
    _, _, w_2 = scn_distribution(alpha=2.0)
    assert w_2.sum() == pytest.approx(w_exp.sum(), rel=1e-14)
    assert np.argmax(w_2) > 0  # при alpha > 1 максимум внутри, а не на первой группе


def test_bubble_point_calibration():
    from paraphin.thermo.tables import calibrate_kij_gas
    kij = calibrate_kij_gas()
    eos, n = oil_fluid(kij, gas=True)[:2]
    assert bubble_pressure(eos, init_T + 273.15, n / n.sum()) == pytest.approx(P_bubble, rel=1e-4)


def test_nghiem_onset_calibration():
    """При давлении начала осаждения и init_T предел растворимости - ровно содержание асфальтенов в нефти."""
    eos, n, mw, ig, _, ia = oil_fluid(0.0, with_asph=True, gas=False)
    model = NghiemAsphaltene(eos, ia, 1e-3)
    t = init_T + 273.15
    x = live_liquid(eos, n, ig, t, P_onset_asph)[0]
    model.calibrate_onset(t, P_onset_asph, x)
    xa, xs = model.solubility(t, P_onset_asph, x)
    assert xa * M_ASPH / np.sum(xs * mw) == pytest.approx(sara_asphaltenes, rel=1e-10)
    # Пойнтинг: v_s > 0, сжатие повышает фугитивность твердого - выше онсета нефть недосыщена
    assert model.solubility(t, 2.0 * P_onset_asph, x)[0] > xa


@pytest.mark.parametrize('gas', [False, True])
def test_tables_reproduce_multisolid(gas):
    """В узле сетки ядро (`sle_split_np` с x_sat и газом из таблиц) дает те же растворенные доли групп,
    что multi-solid на уравнении состояния: подстановка таблицы в форму идеального раствора точна."""
    t_grid, p_grid = np.array([10.0, 30.0]), np.array([5e6, 15e6])
    lnxsat, ng, _, kij = build_tables(t_grid, p_grid, wax=True, asph=False, gas=gas)
    eos, n, _, ig, iw, _ = oil_fluid(kij, gas=gas)
    ms = MultiSolidWax(eos, np.arange(iw, iw + N_W), WAX_TM_WON, WAX_DH_WON, WAX_DV_EOS, P_ref_wax)
    worst = 0.0
    for a, t_c in enumerate(t_grid):
        for b, p in enumerate(p_grid):
            x, n_g = live_liquid(eos, n, ig, t_c + 273.15, p)
            s = ms.precipitate(t_c + 273.15, p, x)[0]
            n_liq = n_g / x[ig] if gas else n.sum()  # моли жидкости на грамм дегазированной нефти
            eos_dis = (x - s)[iw:iw + N_W] * n_liq * WAX_M
            ker_dis = sle_split_np(WAX_W0, WAX_M, None, None, None, None, n_g=n_g, m_o=M_o,
                                   x=np.minimum(1.0, np.exp(lnxsat[:, a, b])))
            assert (WAX_W0 - eos_dis).sum() > 0.0  # парафин выпадает - проверка не пустая
            worst = max(worst, np.abs(ker_dis - eos_dis).max() / WAX_W0.sum())
    # без газа совпадение до сходимости подстановок; с газом часть растворителя уходит в газ
    assert worst < (1e-10 if not gas else 1e-5), worst


def test_coutinho_matches_broadhurst():
    """C24 (Broadhurst, J. Res. NBS 1962, 66A:241, табл. 3): T_m 323.75 K, плавление 54.9 кДж/моль. Переход - по
    формуле (A4) Coutinho (у Broadhurst 31.3 кДж/моль - сумма нескольких переходов четного н-алкана)."""
    t_m, dh_m, t_tr, dh_tr = ch.coutinho_nalkane(24)
    assert float(t_m) == pytest.approx(323.75, abs=0.5)
    assert float(dh_m) == pytest.approx(54.9e3, rel=0.01)
    assert float(dh_tr) == pytest.approx(19.42e3, rel=1e-3) and float(t_tr) < float(t_m)
    assert float(ch.coutinho_nalkane(50)[3]) == 0.0  # с C42 ротаторной фазы нет


def test_uniquac_limits_and_dasilva_wdt():
    """ln gamma чистого твердого - ноль; WDT непрерывной смеси C18-C36 в н-декане (da Silva et al., 2017, Bim 0)
    по PR + UNIQUAC без подбора - в пределах 1.5 C от опыта (у авторов 1.05 C)."""
    import json
    from pathlib import Path
    from paraphin.thermo.solids import UniquacSolidWax

    data = json.loads((Path(__file__).parents[1] / 'experiments' / 'data' / 'dasilva2017_sle.json')
                      .read_text(encoding='utf-8'))
    mix = data['mixtures']['Bim 0']
    n = np.array(sorted(int(k) for k in mix['wt']))
    w = np.array([mix['wt'][str(k)] for k in n])
    M = 14.027 * n + 2.016
    eos = PengRobinson(*ch.critical_props(ch.nalkane_tb(M), ch.nalkane_sg(M)))
    f = n != data['solvent']
    t_m, dh_m, t_tr, dh_tr = ch.coutinho_nalkane(n)
    ms = MultiSolidWax(eos, np.where(f)[0], t_m[f], dh_m[f], t_tr=t_tr[f], dh_tr=dh_tr[f])
    uq = UniquacSolidWax(ms, n[f])
    one = np.zeros(f.sum())
    one[3] = 1.0
    assert abs(uq.ln_gamma(300.0, one)[3]) < 1e-12
    z = (w / M) / np.sum(w / M)
    assert uq.wat(101325.0, z) - 273.15 == pytest.approx(mix['wdt_exp'], abs=1.5)
    s, L, x, xs = uq.precipitate(uq.wat(101325.0, z) - 5.0, 101325.0, z)
    assert np.abs(L * x + s - z).max() < 1e-10 and s.sum() > 0.0


def test_nghiem_precipitate_leaves_saturated_liquid():
    eos, n, mw, ig, _, ia = oil_fluid(0.0, with_asph=True, gas=False)
    model = NghiemAsphaltene(eos, ia, 1e-3)
    t, x = init_T + 273.15, n / n.sum()
    model.calibrate_onset(t, P_onset_asph, x)
    p = 0.5 * P_onset_asph  # ниже онсета нефть пересыщена (без газа - монотонно)
    s = model.precipitate(t, p, x)
    assert s > 0.0
    assert (x[ia] - s) / (1.0 - s) == pytest.approx(model.solubility(t, p, x)[0], rel=1e-9)


def test_live_oil_flash_ignores_asphaltene_liquid():
    """Флюид Tabzar et al. (2018) с kij асфальтены-легкие 0.4: PR делит его выше P_b на две жидкости. Flash живой
    нефти ищет только газ, поэтому давление насыщения существует и лежит между P_b опыта и давлением онсета."""
    import json
    from pathlib import Path

    d = json.loads((Path(__file__).parents[1] / 'experiments' / 'data' / 'tabzar2018_asphaltene.json')
                   .read_text(encoding='utf-8'))
    c = d['components']
    z = np.array([x['mol'] for x in c])
    z /= z.sum()
    light = np.array([x['light'] for x in c])
    ia = next(i for i, x in enumerate(c) if x.get('asphaltene'))
    kij = np.zeros((len(c), len(c)))
    kij[ia, light] = kij[light, ia] = 0.4
    eos = PengRobinson([x['Tc'] for x in c], np.array([x['Pc_atm'] for x in c]) * 101325.0,
                       [x['omega'] for x in c], kij)
    t = (d['T_F'] - 32.0) / 1.8 + 273.15
    assert 0.0 < flash(eos, t, 30e6, z)[0] < 1.0  # без ограничения - две жидкости
    p_b = bubble_pressure(eos, t, z) / 6894.757
    assert d['P_bubble_psia'] < p_b < d['P_onset_psia']


def test_pvt_at_bubble_point():
    from paraphin.constants import Rs_bubble
    from paraphin.thermo.pvt import pvt_point
    from paraphin.thermo.tables import calibrate_kij_gas

    eos, n, mw, ig = oil_fluid(calibrate_kij_gas(), gas=True)[:4]
    t = init_T + 273.15
    below, above = pvt_point(eos, n, mw, ig, t, 0.999 * P_bubble), pvt_point(eos, n, mw, ig, t, 1.2 * P_bubble)
    assert below['Rs'] == pytest.approx(Rs_bubble, rel=2e-3) and above['Rs'] == pytest.approx(Rs_bubble, rel=1e-6)
    assert above['free_gas'] == 0.0 and np.isnan(above['Bg'])
    assert 0.0 < below['free_gas'] < 0.01 and below['Bo'] > above['Bo'] > 1.0  # выше P_b нефть сжимается
    assert below['mu_g'] == pytest.approx(14.5e-6, rel=0.1)  # метан, 70 C, 9.5 МПа (NIST - около 14.5 мкПа*с)


def test_asph_curve_recovers_parameters(monkeypatch):
    """Кривая выпавших, посчитанная моделью, возвращает ее параметры: v_a у Хиршберга, V_s и kij у Нгхайема."""
    import paraphin.oil_composition as oc
    from paraphin.constants import R
    from paraphin.thermo.tables import calibrate_kij_gas, fit_nghiem_curve, nghiem_lnwamax

    p = np.array([4e6, 7e6, 9.5e6, 10.5e6])
    monkeypatch.setattr(oc, 'asph_curve', list(zip(p, oc.hirschberg_precipitated(1.6e-3, p))))
    assert oc._fit_v_asph() == pytest.approx(1.6e-3, rel=1e-4)

    kij_gas, t, tg = calibrate_kij_gas(), init_T + 273.15, np.array([float(init_T)])
    eos, n, _, ig, _, ia = oil_fluid(kij_gas, with_asph=True, gas=True, kij_asph=0.2)
    x, h = live_liquid(eos, n, ig, t, P_onset_asph)[0], 1e4
    v_a = (R * t * (eos.ln_phi(t, P_onset_asph + h, x, 'liquid')[ia] - eos.ln_phi(t, P_onset_asph - h, x, 'liquid')[ia])
           / (2.0 * h) + R * t / P_onset_asph)
    w = np.maximum(sara_asphaltenes - np.exp(nghiem_lnwamax(tg, p, kij_gas, True, 1.005 * v_a, 0.2)[0]), 0.0)
    assert w.max() > 0.0
    v_s, kij = fit_nghiem_curve(list(zip(p, w)), kij_gas, True)
    assert v_s == pytest.approx(1.005 * v_a, rel=1e-3) and kij == pytest.approx(0.2, abs=0.01)


def test_eos_interp_nodes():
    tab = np.random.default_rng(0).random((NT, 1))
    for i in range(NT):
        assert eos_interp(tab, T_GRID[i], 1e5) == pytest.approx(tab[i, 0], abs=1e-14)
    assert eos_interp(tab, T_GRID[0] - 50.0, 1e5) == tab[0, 0]  # за сеткой - край
    mid = 0.5 * (T_GRID[3] + T_GRID[4])
    assert eos_interp(tab, mid, 1e5) == pytest.approx(0.5 * (tab[3, 0] + tab[4, 0]), abs=1e-14)
