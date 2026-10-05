r"""Свойства псевдокомпонентов для уравнения состояния и гамма-распределение н-алканов.

Н-алканы по молярной массе M [г/моль] - корреляции гомологического ряда (Riazi, Characterization and
Properties of Petroleum Fractions, ASTM 2005, разд. 2.3.3; Riazi & Al-Sahhaf, FPE 1996, 117:217):
    T_b = 1070 - exp(6.98291 - 0.02013 M^(2/3)) [K],     SG = 0.85 - exp(92.22793 - 89.82447 M^0.01).
Нефтяная фракция с известными M и SG - температура кипения по Riazi & Daubert (IECR 1987, 26:755):
    T_b = 9.3369 exp(1.6514e-4 M + 1.4103 SG - 7.5152e-4 M SG) M^0.5369 SG^(-0.7276) [K].
Критические свойства - Riazi & Daubert (Hydrocarbon Process. 1980, 59:115), T_b в градусах Ренкина:
    T_c[R] = 24.2787 T_b^0.58848 SG^0.3596,     P_c[psia] = 3.12281e9 T_b^(-2.3125) SG^2.3201.
Ацентрический фактор - Kesler & Lee (Hydrocarbon Process. 1976, 55:153), две ветви по T_br = T_b/T_c:
при T_br < 0.8 - формула через P_c, при T_br >= 0.8 - через фактор Уотсона K_w = (1.8 T_b)^(1/3)/SG
(в нее попадают группы тяжелее C20).

Гамма-распределение плюс-фракции по молярной массе (Whitson, SPEJ 1983, 23:683):
    p(M) = (M - eta)^(alpha - 1) exp[-(M - eta)/beta] / (beta^alpha Gamma(alpha)).
Мольная доля SCN-группы - интеграл по ее границам M_n -+ 7.0135; при alpha = 1 это экспонента
z_n ~ exp(-14.027 n/beta), то есть прежнее распределение `scn_slope` при beta = 14.027/scn_slope.
"""
import numpy as np

_PSI_TO_PA = 6894.757293168


def nalkane_tb(M):
    """Нормальная температура кипения н-алкана, [K]."""
    return 1070.0 - np.exp(6.98291 - 0.02013 * np.asarray(M, float) ** (2.0 / 3.0))


def nalkane_sg(M):
    """Относительная плотность н-алкана (60/60 F), [-]."""
    return 0.85 - np.exp(92.22793 - 89.82447 * np.asarray(M, float) ** 0.01)


def fraction_tb(M, sg):
    """Нормальная температура кипения нефтяной фракции по M [г/моль] и SG (Riazi-Daubert, 1987), [K]."""
    return (9.3369 * np.exp(1.6514e-4 * M + 1.4103 * sg - 7.5152e-4 * M * sg) * M ** 0.5369 * sg ** -0.7276)


def riazi_daubert_tc_pc(tb, sg):
    """Критические температура [K] и давление [Па] по T_b [K] и SG (Riazi-Daubert, 1980)."""
    tb_r = 1.8 * np.asarray(tb, float)
    tc = 24.2787 * tb_r ** 0.58848 * sg ** 0.3596 / 1.8
    pc = 3.12281e9 * tb_r ** -2.3125 * sg ** 2.3201 * _PSI_TO_PA
    return tc, pc


def kesler_lee_acentric(tb, tc, pc, sg):
    """Ацентрический фактор по Kesler-Lee (1976), обе ветви по T_br."""
    tb, tc, pc, sg = (np.asarray(v, float) for v in (tb, tc, pc, sg))
    tbr = tb / tc
    low = ((-np.log(pc / 101325.0) - 5.92714 + 6.09648 / tbr + 1.28862 * np.log(tbr) - 0.169347 * tbr**6)
           / (15.2518 - 15.6875 / tbr - 13.4721 * np.log(tbr) + 0.43577 * tbr**6))
    kw = (1.8 * tb) ** (1.0 / 3.0) / sg
    high = -7.904 + 0.1352 * kw - 0.007465 * kw**2 + 8.359 * tbr + (1.408 - 0.01063 * kw) / tbr
    return np.where(tbr < 0.8, low, high)


def critical_props(tb, sg):
    """(Tc [K], Pc [Па], omega) псевдокомпонента по T_b и SG."""
    tc, pc = riazi_daubert_tc_pc(tb, sg)
    return tc, pc, kesler_lee_acentric(tb, tc, pc, sg)


def coutinho_nalkane(n):
    """Плавление н-алкана с числом атомов n по Coutinho (SPE 78324, 2002, приложение): (T_m [K], dH_m [Дж/моль],
    T_tr [K], dH_tr [Дж/моль]) - температуры и теплоты плавления и перехода ротаторной фазы в орторомбическую.

    T_m = 421.63 - 1936112.63 exp[-7.8945 (n - 1)^0.07194] (A1), T_tr = 420.42 - 134784.42 exp[-4.344 (n + 6.592)^0.14627]
    (A5). При 8 < n < 42 теплоты - полиномы (A4), (A6), при n >= 42 ротаторной фазы нет и dH_m = 3.7791 n - 12.654
    кДж/моль (A2). В статье подписи (A4) и (A6) перепутаны: сверка с Broadhurst (J. Res. NBS 1962, 66A:241, табл. 3)
    у C23 и C25 дает плавление 54.0 и 57.7 кДж/моль при (A6) 52.9 и 57.2, переход 21.8 и 26.1 при (A4) 17.8 и 21.1, -
    поэтому здесь (A6) - плавление, (A4) - переход.
    """
    n = np.asarray(n, float)
    t_m = 421.63 - 1936112.63 * np.exp(-7.8945 * (n - 1.0) ** 0.07194)
    t_tr = 420.42 - 134784.42 * np.exp(-4.344 * (n + 6.592) ** 0.14627)
    dh_m = 1e3 * np.where(n < 42, 0.00355 * n ** 3 - 0.2376 * n ** 2 + 7.4 * n - 34.814, 3.7791 * n - 12.654)
    dh_tr = 1e3 * np.where(n < 42, -0.00355 * n ** 3 + 0.2376 * n ** 2 - 3.6209 * n + 18.5391, 0.0)
    return t_m, dh_m, t_tr, np.maximum(dh_tr, 0.0)


def pedersen_dcp(M, T):
    """Скачок теплоемкости при плавлении C_p^L - C_p^S, [Дж/(моль*К)]: 0.3033 M - 4.635e-4 M T, кал/(моль*К), M в
    г/моль (Pedersen et al., Energy Fuels 1991, 5:924; так же у Lira-Galeana et al., 1996)."""
    M = np.asarray(M, float)
    return 4.184 * (0.3033 * M - 4.635e-4 * M * T)


def nalkane_dh_vap(n):
    """Теплота испарения н-алкана при 298 К, [Дж/моль]: 5.0 кДж/моль на группу CH2 по данным NIST для н-декана и
    н-гексадекана (51.4 и 81.4 кДж/моль). Нужна только разностями - в энергиях взаимодействия UNIQUAC.

    ponytail: без зависимости от температуры; Морган-Кобаяши (FPE 1994, 94:51) - если понадобится точнее."""
    return 1e3 * (5.0 * np.asarray(n, float) + 1.4)


def gamma_scn_fractions(M, alpha, eta, beta, half=7.0135):
    """Мольные доли SCN-групп со средними массами M по гамма-распределению: F(M + half) - F(M - half).

    Разность верхних неполных гамма-функций: у хвоста распределения она не теряет точность, как
    разность функций распределения, близких к единице.
    """
    from scipy.special import gammaincc

    def tail(m):
        return gammaincc(alpha, np.maximum(m - eta, 0.0) / beta)

    return tail(M - half) - tail(M + half)
