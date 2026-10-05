r"""PVT-свойства нефти модели по уравнению состояния: R_s, B_o, B_g, плотность и вязкость газа, объем свободного газа.

Нефть - флюид `tables.oil_fluid` (газ, растворитель, группы парафина), моли на грамм дегазированной нефти. В каждой
точке (T, p) - flash пар-жидкость (`eos.flash`, только паровая пробная фаза), затем
    R_s = n_g^L R T_sc/P_sc rho_o,     B_o = V_L(T, p)/V_dead(T_sc, P_sc),     B_g = Z_V T P_sc/(p T_sc),
n_g^L - растворенный газ на грамм дегазированной нефти, V - объемы PR (B_o - отношение объемов одного уравнения:
без сдвига объема Peneloux абсолютные плотности PR занижены на 10-15 %, отношение - меньше). Вязкость газа - Lee,
Gonzalez & Eakin (JPT 1966, 18:997) по плотности газа из PR.

Таблицы диагностические: уравнения фильтрации их не читают (газ растворен в нефти, свободной газовой фазы в модели
нет). Объем свободного газа на объем нефти при пластовых условиях V_g/V_L = beta Z_V/((1 - beta) Z_L) выгружается
полем 'Free gas' (`utils/save_data_fields.py`) при wax_eos и wax_pressure: где он больше нуля, ниже давления
насыщения модель держит газ растворенным, хотя он выделился бы.

`python -m paraphin.thermo.pvt` - таблица свойств при init_T.
"""
import numpy as np

from paraphin.constants import T_sc_gas, P_sc_gas, ro_o


def lee_gonzalez_eakin(T, rho, M):
    """Вязкость природного газа, [Па*с]: T [K], плотность [кг/м^3], молярная масса [г/моль]."""
    t_r = 1.8 * T
    k = (9.4 + 0.02 * M) * t_r ** 1.5 / (209.0 + 19.0 * M + t_r)
    x = 3.5 + 986.0 / t_r + 0.01 * M
    return 1e-7 * k * np.exp(x * (rho * 1e-3) ** (2.4 - 0.2 * x))  # 1e-4 сП = 1e-7 Па*с


def pvt_point(eos, n, mw, ig, T, p):
    """PVT-свойства при (T [K], p [Па]). Однофазная нефть - газовые свойства NaN, свободного газа нет.

    Returns
    -------
    dict: Rs [м^3/м^3], Bo, Bg [-], rho_g [кг/м^3], mu_g [Па*с], Z_g, free_gas = V_g/V_L [-]
    """
    from .eos import R_GAS, flash

    z = n / n.sum()
    beta, x, y = flash(eos, T, p, z, vapor_only=True)
    if not 0.0 < beta < 1.0:
        beta, x = 0.0, z
    n_liq = (1.0 - beta) * n.sum()
    v_liq = n_liq * eos.z_factor(T, p, x, 'liquid') * R_GAS * T / p  # [м^3/г дегазированной нефти]
    dead = np.arange(eos.nc) != ig
    zd = n[dead] / n[dead].sum()
    eos_d = eos if ig < 0 else type(eos)(eos.tc[dead], eos.pc[dead], eos.omega[dead], eos.kij[np.ix_(dead, dead)])
    v_dead = n[dead].sum() * eos_d.z_factor(T_sc_gas, P_sc_gas, zd, 'liquid') * R_GAS * T_sc_gas / P_sc_gas
    out = {'Rs': n_liq * x[ig] * R_GAS * T_sc_gas / P_sc_gas * ro_o * 1e3 if ig >= 0 else 0.0,
           'Bo': v_liq / v_dead, 'Bg': np.nan, 'rho_g': np.nan, 'mu_g': np.nan, 'Z_g': np.nan, 'free_gas': 0.0}
    if beta > 0.0:
        zv = eos.z_factor(T, p, y, 'vapor')
        m_g = float(y @ mw)
        rho = p * m_g * 1e-3 / (zv * R_GAS * T)
        out.update(Bg=zv * T * P_sc_gas / (p * T_sc_gas), rho_g=rho, mu_g=lee_gonzalez_eakin(T, rho, m_g), Z_g=zv,
                   free_gas=beta * zv / ((1.0 - beta) * eos.z_factor(T, p, x, 'liquid')))
    return out


def free_gas_table(kij_gas, t_grid, p_grid):
    """V_g/V_L на сетке t_grid [C] x p_grid [Па] - таблица поля 'Free gas'."""
    from .tables import oil_fluid

    eos, n, mw, ig = oil_fluid(kij_gas, gas=True)[:4]
    return np.array([[pvt_point(eos, n, mw, ig, t + 273.15, p)['free_gas'] for p in p_grid] for t in t_grid])


if __name__ == '__main__':
    from paraphin.constants import init_T, P_bubble, Rs_bubble
    from .tables import calibrate_kij_gas, oil_fluid

    kij = calibrate_kij_gas()
    eos, n, mw, ig = oil_fluid(kij, gas=True)[:4]
    print(f'Нефть модели при {init_T} C, kij газ-нефть = {kij:.4f}; задано P_b = {P_bubble / 1e6:.2f} МПа, '
          f'R_s = {Rs_bubble} м^3/м^3')
    print('| P, МПа | R_s, м^3/м^3 | B_o | B_g | rho_g, кг/м^3 | mu_g, мкПа*с | Z_g | V_g/V_L |')
    for p in np.r_[np.linspace(0.5e6, P_bubble, 8), 12e6, 20e6]:
        r = pvt_point(eos, n, mw, ig, init_T + 273.15, p)
        print(f"| {p / 1e6:.2f} | {r['Rs']:.1f} | {r['Bo']:.4f} | {r['Bg']:.5f} | {r['rho_g']:.1f} | "
              f"{r['mu_g'] * 1e6:.2f} | {r['Z_g']:.4f} | {r['free_gas']:.3f} |")
