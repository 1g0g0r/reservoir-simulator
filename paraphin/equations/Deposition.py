"""Кинетика осаждения в пористой среде: ядро кольматации при флагах кинетики (`deposition_kinetics`).

Механизмы (обзор `твт_статья_АСПО/материалы/full_review.pdf`, разд. 2-5; описание с формулами - `docs/модель_АСПО.md`, разд. 13):

  перенос частиц к стенке капилляра (2.2)   броуновская диффузия (как в прежней модели) + сдвиговая дисперсия
                                            + гравитационное оседание (`wall_transport`);
  захват частиц                             сужение широких капилляров и блокирование узких - кристаллы парафина
                                            и флокулы асфальтенов; размер флокулы - из агрегации
                                            (`asph_aggregation`), рост захвата с осадком (`snowball`);
  кристаллизация на стенках (2.3)            пересыщение растворенного парафина уходит на стенки пор
                                            (`wax_kinetics`; при `thermal_nonequilibrium` - и за счет более
                                            холодной породы, диффузия по Фику);
  адсорбция асфальтенов и смол               слой на стенках всех каналов (`adsorption`);
  вынос (2.2.5, 3.6)                         сдвиговое сдирание осадка при tau_w > tau_кр (`entrainment`);
  старение гель-отложения (2.4)              отложение - гель с долей парафина C, сужает каналы по объему геля;
                                            C растет к C_max за счет растворенного парафина (`deposit_aging`).

Все механизмы собираются в одну скорость изменения радиуса u(r) и коэффициент блокирования b(r), и функция
пор fi переносится по оси радиусов той же неявной схемой, что в `Qp_m_k_fi._update_fi` (`Pore_bundle.update_fi_rows`). Скорости переноса
считаются по текущему слою, а не берутся с прошлого шага, как `Ur`, `Ua` в прежних ядрах: запаздывание на
шаг с ограничителем подводом давало колебания доли флокул.

Выходной контракт тот же, что у `calc_qp_m_k_fi_2`: скорости потери порового объема по чистым материалам
(`new_qp1` - осадок парафина на стенках, `new_qp2` - пробки парафина, `new_kx[KX_QPA]` - осадок асфальтенов и смол,
плюс новые каналы в `new_kx`: стеночная кристаллизация, старение, адсорбция асфальтенов и смол), и
тождество  (q_p1 + q_p2 + q_pa + q_w + q_g + q_ada + q_adr)*dt = m - m^new  выполняется по построению.
Отрицательная скорость - вынос: осадок возвращается в нефть (`components_equation`).

Режим `deposition_model = 'filtration'` - континуальная модель глубинной фильтрации (`calc_filtration`) с тем
же контрактом: fi не меняется, проницаемость - из корреляции k(phi, sigma).
"""
import math

from numba import njit

from paraphin.geometry import r1, r2, r4, w2_cv, plug_cv, dr_cv, eta, surf_0, fi_0
from paraphin.constants import (Nr, init_m, init_k, min_Wps_bound, ro_o, ro_p, ro_asph, ro_asph_dep,
                                resin_in_deposit, volume, D, D_asph, betta, gamma, diff_mult, S_max, R,
                                E_activation,
                                asphaltenes, wax_kinetics, wall_transport, entrainment, asph_aggregation,
                                snowball, adsorption, deposit_aging, thermal_nonequilibrium, perm_model, pore_network)
from paraphin.kinetics_params import (K_CRYST, K_WALL, SHEAR_DISP, GRAV_EFF, DIFF_MULT, D_CRYST, ENT_RATE, ENT_TAU, AGG_D0,
                                      AGG_DF, SNOW_A, ADS_GMAX, ADS_K, ADS_DH, ADS_T_REF, ADS_RATE, ADS_RESIN,
                                      ADS_FILM, AGE_C0, AGE_CMAX, AGE_RATE, FILT_KD, FILT_KPL, FILT_KE, FILT_UCR,
                                      PERM_N, PERM_BETA, PERM_SMAX, PERM_GAMMA, PERM_ALPHA, LTNE_DG, LTNE_DM, NET_Z,
                                      NET_GAMMA)
from paraphin.layout import (N_W, IA_D, IA_F, I_R, IN_F, KX_WEQ, KX_WSH, KX_GSH, KX_QW, KX_QG, KX_QADA, KX_QADR, KX_GA,
                             KX_GR, KX_VGEL, KX_TS, KX_GMAX, KX_SIG0, KX_QPA, ROW_U, ROW_UW, ROW_UA, ROW_BW, ROW_BA, ROW_UE,
                             ROW_TMP, ROW_A, ROW_B)
from paraphin.utils import crystal_volume_fraction
from .Thermo_wax import sle_split
from .Pore_bundle import moments, int_r_u_fi, weighted, update_fi_rows
from .Kinetics_math import (brownian_diffusivity, shear_diffusivity, stokes_velocity, leveque_velocity, floc_size,
                            wall_fraction, langmuir_constant, langmuir_ldf_step, langmuir_film_step, perm_kozeny_carman,
                            perm_power, perm_damage, perm_surface, ema_conductance)

_SO_MAX = 1.0 - S_max
_RO_P_O = ro_p / ro_o
_RO_AD_O = ro_asph_dep / ro_o


_K_FLOOR = 1e-8  # остаточная проницаемость, доли init_k


# --- Скорости переноса частиц к стенке ------------------------------------------------------------------------

@njit(cache=True)
def _particle_rows(u, b, so_eff, w, d_p, rho_p, um_r2, T_K, mu, kin, snow, phi_vol, n_block_max):
    """Профили сужения u(r) и коэффициента блокирования b(r) для частиц диаметра d_p с массовой долей w.

    Узкие каналы (r < d_p/(2*gamma)) частица затыкает целиком: b = So*w*6*betta/d_p^3*um_r2*r^4 (как у
    кристаллов в `Velocity_h`). В широких оседает на стенке: коэффициент массоотдачи типа Левека
    k = (2*um*D^2/(r*Lk))^(1/3), um = um_r2*r^2, с суммарным коэффициентом переноса частиц
        D = k_B*T/(3*pi*mu*d_p) + shear_disp*phi^2*gamma_w*a^2,   gamma_w = 4*um/r  (Leighton & Acrivos 1987)
    плюс гравитационное оседание grav_eff*u_s, u_s = 2*a^2*(rho_p - rho_o)*g/(9*mu) (обзор 2.2.3-2.2.4):
        u(r) = -So*w*(k(r) + grav_eff*u_s)*snow.
    Без `wall_transport` остается только броуновская часть - ровно формула прежних ядер. Броуновский
    коэффициент умножается на runtime-множитель kin[DIFF_MULT] (подбор по опыту без перекомпиляции)."""
    a = 0.5 * d_p
    r_pass = a / gamma
    d_b = diff_mult * kin[DIFF_MULT] * brownian_diffusivity(T_K, mu, d_p)
    ug = kin[GRAV_EFF] * stokes_velocity(a, rho_p - ro_o, mu) if wall_transport else 0.0
    b_coef = so_eff * w * 6.0 * betta / (d_p * d_p * d_p) * um_r2 * snow
    for ij in range(Nr):
        if r1[ij] < r_pass and ij < n_block_max:
            b[ij] = b_coef * r4[ij]
            u[ij] = 0.0
        else:
            b[ij] = 0.0
            d_tot = d_b
            if wall_transport:  # скорость сдвига у стенки 4*um/r = 4*um_r2*r
                d_tot += shear_diffusivity(kin[SHEAR_DISP], phi_vol, 4.0 * um_r2 * r1[ij], a)
            u[ij] = -so_eff * w * (leveque_velocity(um_r2 * r2[ij], r1[ij], d_tot) + ug) * snow if ij > 0 else 0.0


@njit(cache=True)
def floc_diameter(w_af, n_f, kin):
    """Диаметр флокулы из массы и числа флокул: d_f = d_0*(m_f/m_0)^(1/D_f), фрактальный агрегат (обзор 5.2).
    Без агрегации - постоянный `D_asph`."""
    if not asph_aggregation:
        return D_asph
    if n_f <= 0.0 or w_af <= 0.0:
        return kin[AGG_D0]
    return floc_size(w_af / n_f, kin[AGG_D0], kin[AGG_DF], ro_asph)


# --- Ядро: пучок капилляров -------------------------------------------------------------------------------------

@njit(cache=True)
def calc_deposition(i, j, S, T, p, m, k, fi, h, Wc, Ws, Wps, Dep, um_r2, grad_p, mu_p, kx, kin,
                    integr_r2_fi0, integr_r4_fi0, rows, net_g0,
                    new_qp1, new_qp2, new_fi, new_h, new_k, new_m, new_kx, out_o, dt) -> None:
    """Кольматация ячейки (i, j) всеми включенными механизмами в модели пучка капилляров.

    rows: numpy.ndarray(NROWS, Nr) - строки скретча `Solver.rows[i]` (свои на каждый i, иначе гонка в prange).
    kx, new_kx - поля кинетических моделей (`kinetics_params`), kin - вектор параметров `solver.kin`.
    out_o - отток нефти из ячейки за шаг через грани и добывающую скважину, [м^3/с]: запас на осаждение -
    то, что останется после оттока (как в `calc_qp_m_k_fi_2`). net_g0 - проводимость исходной сети пор
    (`network_g0`, при `pore_network`): одна на все ячейки, поэтому считается раз за шаг до цикла по ячейкам.
    """
    u, uw, ua, bw, ba, ue = rows[ROW_U], rows[ROW_UW], rows[ROW_UA], rows[ROW_BW], rows[ROW_BA], rows[ROW_UE]
    tmp = rows[ROW_TMP]
    to_m = init_m / integr_r2_fi0
    so = 1.0 - S[i, j]
    so_eff = max(0.0, so - _SO_MAX)
    mso = m[i, j] * so
    mso_dt = max(mso / dt - out_o / volume, 0.0)  # запас на осаждение за шаг, [1/с] в долях нефти
    T_K = T[i, j] + 273.15
    mu = mu_p[i, j]
    rfi, r2fi, r4fi = moments(fi, i, j)
    surf = 2.0 * to_m * rfi  # удельная поверхность проводящих каналов, [1/м]

    snow = 1.0
    if snowball:
        snow = 1.0 + kin[SNOW_A] * max(init_m - m[i, j], 0.0) / init_m
    c0 = kin[AGE_C0] if deposit_aging else 1.0  # доля парафина в новом отложении (1 - сплошной кристалл)

    for ij in range(Nr):
        uw[ij] = 0.0
        ua[ij] = 0.0
        bw[ij] = 0.0
        ba[ij] = 0.0
        ue[ij] = 0.0

    # --- кристаллы парафина из взвеси
    # Диаметр кристалла - runtime (kin[D_CRYST], по умолчанию D): порог блокирования d/(2*gamma) - по нему, объем
    # пробки - plug_cv с множителем (d/D)^3, как у флокул
    wps = Wps[i, j]
    d_w = kin[D_CRYST]
    plug_w = (d_w / D) ** 3
    lim_w, i_w, p_w = 0.0, 0.0, 0.0
    if wps > min_Wps_bound:
        _particle_rows(uw, bw, so_eff, wps, d_w, ro_p, um_r2[i, j], T_K, mu, kin, snow,
                       crystal_volume_fraction(wps), Nr)
        i_w = max(-2.0 * to_m * int_r_u_fi(fi, i, j, uw), 0.0)  # кристаллы, [1/с]
        p_w = to_m * weighted(fi, i, j, bw, plug_cv, plug_w)
        sink = _RO_P_O * (i_w + p_w)
        avail = mso_dt * wps
        lim_w = avail / sink if sink > avail else 1.0

    # --- флокулы асфальтенов
    lim_a, i_a, p_a, plug_scale = 0.0, 0.0, 0.0, 1.0
    if asphaltenes:
        w_af = Wc[i, j, IA_F]
        if w_af > min_Wps_bound:
            n_f = Wc[i, j, IN_F] if asph_aggregation else 0.0
            d_f = floc_diameter(w_af, n_f, kin)
            phi_f = w_af / ro_asph / (w_af / ro_asph + (1.0 - w_af) / ro_o)
            _particle_rows(ua, ba, so_eff, w_af, d_f, ro_asph, um_r2[i, j], T_K, mu, kin, snow, phi_f,
                           Nr if asph_aggregation else 0)
            plug_scale = (d_f / D) ** 3
            i_a = max(-2.0 * to_m * int_r_u_fi(fi, i, j, ua), 0.0)
            p_a = to_m * weighted(fi, i, j, ba, plug_cv, plug_scale)
            sink_a = _RO_AD_O * (i_a + p_a)
            avail_a = mso_dt * min(Wc[i, j, IA_F] / (1.0 - resin_in_deposit),
                                   Wc[i, j, I_R] / resin_in_deposit if resin_in_deposit > 0.0 else 1e300)
            lim_a = avail_a / sink_a if sink_a > avail_a else 1.0

    # --- кристаллизация на стенках, адсорбция асфальтенов и смол, старение гель-отложения
    q_wall, q_ada, q_adr, q_age = _wall_sinks(i, j, T, p, m, S, Wc, Ws, Dep, kx, kin, mso, mso_dt, dt, tmp, new_kx)

    # --- вынос осадка потоком
    if entrainment:
        tau_coef = grad_p[i, j] * 0.5 / eta  # tau_w = r*|grad p|/(2*eta)
        for ij in range(Nr):
            tw = r1[ij] * tau_coef
            if tw > kin[ENT_TAU] and h[i, j, ij] > 0.0:
                ue[ij] = kin[ENT_RATE] * (tw / kin[ENT_TAU] - 1.0) * h[i, j, ij]

    # Однородная по r скорость - стеночная кристаллизация (через удельную поверхность). Удержанные асфальтены и
    # смолы в сужение каналов не входят: они сидят в горлах, и их вклад в проводимость - функция повреждения ниже
    u_uni = 0.0
    if surf > 0.0:
        u_uni = -(q_wall / c0) / surf
    active = lim_w > 0.0 or lim_a > 0.0 or u_uni != 0.0 or entrainment or adsorption

    if not active and q_age == 0.0:
        for ij in range(Nr):
            new_fi[i, j, ij] = fi[i, j, ij]
            new_h[i, j, ij] = h[i, j, ij]
        new_qp1[i, j] = 0.0
        new_qp2[i, j] = 0.0
        new_kx[i, j, KX_QPA] = 0.0
        new_kx[i, j, KX_QW] = 0.0
        new_kx[i, j, KX_QG] = 0.0
        new_kx[i, j, KX_QADA] = 0.0
        new_kx[i, j, KX_QADR] = 0.0
        new_kx[i, j, KX_VGEL] = kx[i, j, KX_VGEL]
        new_m[i, j] = m[i, j]
        new_k[i, j] = k[i, j]
        return

    # Суммарный профиль: осадок парафина - гель, его толщина по объему геля (/c0)
    for ij in range(Nr):
        u[ij] = lim_w * uw[ij] / c0 + lim_a * ua[ij] + u_uni + ue[ij]
        tmp[ij] = lim_w * bw[ij] + lim_a * ba[ij]
    update_fi_rows(new_fi, fi, i, j, u, tmp, rows[ROW_A], rows[ROW_B], dt)
    for ij in range(Nr):
        new_h[i, j, ij] = max(h[i, j, ij] - u[ij] * dt, 0.0)
    rfi_n, r2fi_n, r4fi_n = moments(new_fi, i, j)

    # Разбор фактической убыли проводящих каналов по механизмам
    blocked = to_m * (lim_w * weighted(new_fi, i, j, bw, w2_cv, 1.0)
                      + lim_a * weighted(new_fi, i, j, ba, w2_cv, 1.0))       # каналы -> тупиковые, [1/с]
    qp2 = lim_w * to_m * weighted(new_fi, i, j, bw, plug_cv, plug_w)         # пробки парафина
    qpa_plug = lim_a * to_m * weighted(new_fi, i, j, ba, plug_cv, plug_scale)  # пробки флокул
    narrow = to_m * (r2fi - r2fi_n) / dt - blocked                            # сужение + вынос, [1/с]
    e_w = lim_w * i_w / c0          # оценки до прогонки: объем геля парафина, флокул, стеночный
    e_a = lim_a * i_a
    e_wall = q_wall / c0
    e_ent = 0.0
    if entrainment:
        e_ent = 2.0 * to_m * int_r_u_fi(fi, i, j, ue)  # объем, освобожденный выносом, [1/с]
    pos = e_w + e_a + e_wall
    scale = (narrow + e_ent) / pos if pos > 1e-300 else 0.0
    if scale < 0.0:
        scale = 0.0

    # Вынос возвращает осадок пропорционально его составу: объем геля парафина и осадок асфальтенов
    ent_wax, ent_asph = 0.0, 0.0
    if entrainment and e_ent > 0.0:
        dep_wax = 0.0
        for kk in range(N_W):
            dep_wax += Dep[i, j, kk]
        dep_wax_v = dep_wax / ro_p
        dep_asph_v = (Dep[i, j, IA_F] + Dep[i, j, I_R]) / ro_asph_dep
        c_dep = c0
        if deposit_aging and kx[i, j, KX_VGEL] > 0.0:
            c_dep = min(dep_wax_v / kx[i, j, KX_VGEL], 1.0)
        tot = dep_wax_v / c_dep + dep_asph_v if c_dep > 0.0 else dep_asph_v
        if tot > 0.0:
            ent_gel = e_ent * dep_wax_v / c_dep / tot if c_dep > 0.0 else 0.0
            ent_wax = min(ent_gel * c_dep, dep_wax_v / dt)                  # кристаллы, [1/с]
            ent_asph = min(e_ent * dep_asph_v / tot, dep_asph_v / dt)

    qp1 = c0 * e_w * scale - ent_wax
    q_w = c0 * e_wall * scale
    qpa = e_a * scale + qpa_plug - ent_asph

    new_qp1[i, j] = qp1
    new_qp2[i, j] = qp2
    new_kx[i, j, KX_QPA] = qpa
    new_kx[i, j, KX_QW] = q_w
    new_kx[i, j, KX_QG] = q_age
    new_kx[i, j, KX_QADA] = q_ada
    new_kx[i, j, KX_QADR] = q_adr
    gel = kx[i, j, KX_VGEL] + (e_w + e_wall) * scale * dt
    if entrainment and ent_wax > 0.0:
        c_dep = c0
        if kx[i, j, KX_VGEL] > 0.0:
            dw = 0.0
            for kk in range(N_W):
                dw += Dep[i, j, kk]
            c_dep = max(min(dw / ro_p / kx[i, j, KX_VGEL], 1.0), 1e-12)
        gel -= ent_wax / c_dep * dt
    new_kx[i, j, KX_VGEL] = max(gel, 0.0)

    new_m[i, j] = m[i, j] - (qp1 + qp2 + qpa + q_w + q_age + q_ada + q_adr) * dt
    if pore_network:
        new_k[i, j] = init_k * _network_ratio(new_fi, new_h, i, j, kin, net_g0, tmp, ue)  # скратчи уже отработали
    else:
        new_k[i, j] = init_k * r4fi_n / integr_r4_fi0
    if adsorption:
        # Удержание в горлах: малый объем - большая потеря проводимости (обзор 4.1, структурный эффект), функция
        # повреждения Civan (2015) по объему удержанного, sigma = (G_a + G_r)/ro_ad
        sigma_v = (kx[i, j, KX_GA] + kx[i, j, KX_GR]) / ro_asph_dep + (q_ada + q_adr) * dt
        # повреждение - от начального удержания (`ads_init_equilibrium`; у чистого керна sigma_0 = 0, D = 1)
        d_0 = perm_damage(kx[i, j, KX_SIG0] / (kin[PERM_SMAX] * init_m), kin[PERM_BETA], kin[PERM_GAMMA])
        new_k[i, j] *= perm_damage(sigma_v / (kin[PERM_SMAX] * init_m), kin[PERM_BETA], kin[PERM_GAMMA]) / d_0
    # Остаточная проницаемость: функция повреждения при sigma >= sigma_max и сеть ниже порога протекания дают
    # ноль, а нулевая подвижность делает матрицу давления вырожденной
    new_k[i, j] = max(new_k[i, j], _K_FLOOR * init_k)


@njit(cache=True)
def _wall_sinks(i, j, T, p, m, S, Wc, Ws, Dep, kx, kin, mso, mso_dt, dt, tmp, new_kx):
    """Стоки на стенках, общие для обоих ядер, - объемы за единицу времени, [1/с]; выключенный флагом механизм дает
    ноль: кристаллизация пересыщенного растворенного парафина (`wax_kinetics`, `thermal_nonequilibrium`),
    адсорбция асфальтенов и смол (`adsorption`, отрицательная - десорбция), старение гель-отложения
    (`deposit_aging`: парафин входит в гель при постоянном его объеме). tmp - строка скретча длины Nr."""
    q_wall = 0.0
    if wax_kinetics or thermal_nonequilibrium:
        q_wall = _wall_crystallization(i, j, T, p, Wc, Ws, kx, kin, m, S, mso_dt, dt, tmp, new_kx)
    q_ada, q_adr = 0.0, 0.0
    if adsorption:
        q_ada, q_adr = _adsorption(i, j, T[i, j] + 273.15, Wc, kx, kin, mso, mso_dt, dt, new_kx)
    q_age = 0.0
    if deposit_aging:
        q_age = _aging(i, j, Wc, Ws, Dep, kx, kin, mso_dt, tmp, new_kx)
    return q_wall, q_ada, q_adr, q_age


@njit(cache=True)
def network_g0(kin, g, w):
    """Эффективная проводимость исходной сети (fi_0, h = 0) - знаменатель k/k0 в `_network_ratio`. Зависит только
    от NET_Z, NET_GAMMA, поэтому одна на все ячейки. g, w - скратч-строки длины Nr."""
    scale = 1.0 / r1[Nr - 1]
    for ij in range(Nr):
        g[ij] = (kin[NET_GAMMA] * r1[ij] * scale) ** 4
        w[ij] = fi_0[ij] * dr_cv[ij] if ij > 0 else 0.0
    w_open = 0.0
    for ij in range(Nr):
        w_open += w[ij]
    return ema_conductance(g, w, max(1.0 - w_open, 0.0), kin[NET_Z])


@njit(cache=True)
def _network_ratio(fi, h, i, j, kin, gm0, g, w):
    """k/k0 сети пор и горл (`pore_network`, docs/кинетика_осаждения.md, разд. 13.13).

    Каналы fi(r) - поры, их горла - net_gamma от исходного радиуса поры: у поры радиуса r с отложением толщины h
    горло net_gamma*(r + h) - h (тот же слой h). Проводимость горла ~ r_t^4, доля горл класса - fi*dr_cv
    (sum(fi_0*dr_cv) = 1). Закрытые горла сети - блокированные каналы (ушли из fi) и горла, закрытые слоем.
    Эффективная проводимость - `ema_conductance` с координационным числом net_z, нормированная на исходную сеть
    (fi_0, h = 0) - она приходит готовой, gm0 (`network_g0`). g, w - скратч-строки длины Nr (нарезаны по i, как все
    скратчи ядра)."""
    z, gam = kin[NET_Z], kin[NET_GAMMA]
    scale = 1.0 / r1[Nr - 1]
    w_open = 0.0
    for ij in range(Nr):
        rt = gam * (r1[ij] + h[i, j, ij]) - h[i, j, ij]
        if ij > 0 and rt > 0.0 and fi[i, j, ij] > 0.0:
            g[ij] = (rt * scale) ** 4
            w[ij] = fi[i, j, ij] * dr_cv[ij]
            w_open += w[ij]
        else:
            g[ij] = 0.0
            w[ij] = 0.0
    gm = ema_conductance(g, w, max(1.0 - w_open, 0.0), z)
    return max(gm / gm0, 1e-8) if gm0 > 0.0 else 1.0


@njit(cache=True)
def _wall_crystallization(i, j, T, p, Wc, Ws, kx, kin, m, S, mso_dt, dt, tmp, new_kx):
    """Кристаллизация пересыщенного растворенного парафина прямо на стенках пор (Huang et al., 2011; обзор 2.3).

    `wax_kinetics`: взвесь в нефти неравновесна, пересыщение группы k - разность равновесной и фактической
    взвеси, Delta_k = w_s,k^eq - w_s,k > 0 (растворено больше предела). Оно расходуется двумя конкурирующими путями:
    рост кристаллов в объеме (k_cryst, `components_equation`) и на стенках (k_wall). За шаг на стенки уходит
    доля k_wall/(k_wall + k_cryst)*(1 - exp(-(k_wall + k_cryst)*dt)) пересыщения.
    `thermal_nonequilibrium`: порода холоднее нефти (T_s < T), у стенки нефть пересыщена относительно равновесия
    при T_s, и растворенный парафин диффундирует к стенке по Фику (обзор 2.2.1) с коэффициентом массоотдачи
    Sh*D_m/d_g на удельной поверхности зерен 6*(1 - m)/d_g (Sh = 2 при малых числах Рейнольдса).
    Возвращает объем кристаллов за единицу времени, [1/с]; доли групп - в new_kx[KX_WSH:].
    """
    total = 0.0
    for kk in range(N_W):
        tmp[kk] = 0.0
    mso = m[i, j] * (1.0 - S[i, j])
    if wax_kinetics:
        frac = wall_fraction(kin[K_WALL], kin[K_CRYST], dt)
        for kk in range(N_W):
            delta = kx[i, j, KX_WEQ + kk] - Ws[i, j, kk]
            if delta > 0.0:
                tmp[kk] += frac * delta
    if thermal_nonequilibrium:
        t_s = kx[i, j, KX_TS]
        if t_s < T[i, j]:
            # Равновесие при температуре породы: сколько растворенного лишнее у стенки
            eq_s = tmp[N_W:2 * N_W]
            w_dis, _, _ = sle_split(Wc[i, j, :N_W], t_s, p[i, j], eq_s)
            d_g = kin[LTNE_DG]
            k_f = 2.0 * kin[LTNE_DM] / d_g * 6.0 * (1.0 - m[i, j]) / d_g / max(mso, 1e-12)
            frac_f = 1.0 - math.exp(-k_f * dt)
            for kk in range(N_W):
                dis_now = Wc[i, j, kk] - Ws[i, j, kk]
                dis_eq_s = Wc[i, j, kk] - eq_s[kk]
                excess = dis_now - dis_eq_s
                if excess > 0.0:
                    tmp[kk] += frac_f * excess
    for kk in range(N_W):
        total += tmp[kk]
    if total <= 0.0:
        for kk in range(N_W):
            new_kx[i, j, KX_WSH + kk] = 0.0
        return 0.0
    for kk in range(N_W):
        new_kx[i, j, KX_WSH + kk] = tmp[kk] / total
    # За шаг - не больше пересыщения с учетом оттока; в объем кристаллов
    return min(total * mso / dt, total * mso_dt) * ro_o / ro_p


@njit(cache=True)
def _adsorption(i, j, T_K, Wc, kx, kin, mso, mso_dt, dt, new_kx):
    """Адсорбция растворенных асфальтенов и смол на стенках: кинетический Ленгмюр (линейная движущая сила).

        G_eq = G_max*K*c/(1 + K*c),  K(T) = ads_K*exp(-ads_dH/R*(1/T - 1/T_ref)),  G_max = ads_gmax*a_v,
        ads_film = 0:  dG/dt = k_ads*(G_eq - G)  ->  за шаг точно  dG = (G_eq - G)*(1 - exp(-k_ads*dt)),
        ads_film = 1:  dG/dt = k_ads*m*S_o*ro_o*(c - c*(G))  (пленочная, `langmuir_film_step`),
        k_ads = ads_rate*(T/T_ref)*mu(T_ref)/mu(T).
    Скорость - массообмен к стенке (линейная движущая сила Глюкауфа), k_ads ~ D_m/delta^2, а коэффициент
    молекулярной диффузии по Уилки-Чангу D_m ~ T/mu: от 90 до 45 C в нефти Li et al. (2024) он падает втрое.
    mu - вязкость жидкой основы по Аррениусу с E_activation (гель и кристаллы на диффузию молекул не влияют).
    a_v = surf_0 - удельная поверхность породы (пучок fi_0), а не текущих проводящих каналов: емкость Ленгмюра
    измеряют на единицу массы породы, и адсорбированное в канале, который потом заткнул парафин, остается на месте.
    С поверхностью проводящих каналов емкость падала вместе с закупоркой, и смолы десорбировались - на ступени
    45 C опыта Li et al. (2024) проницаемость от этого росла к концу ступени. Возвращает объем слоя за единицу времени для асфальтенов
    и смол, [1/с] (отрицательный - десорбция); сами количества G обновляет `components_equation`."""
    k_l = langmuir_constant(kin[ADS_K], kin[ADS_DH], T_K, kin[ADS_T_REF] + 273.15, R)
    t_ref = kin[ADS_T_REF] + 273.15
    k_ads = kin[ADS_RATE] * T_K / t_ref * math.exp(E_activation / R * (1.0 / t_ref - 1.0 / T_K))
    film = kin[ADS_FILM] > 0.5
    frac = k_ads * mso * ro_o * dt if film else 1.0 - math.exp(-k_ads * dt)
    g_max = kin[ADS_GMAX] * surf_0
    new_kx[i, j, KX_GMAX] = g_max
    q_a = _langmuir_step(Wc[i, j, IA_D], kx[i, j, KX_GA], g_max, k_l, frac, film, mso_dt, dt)
    q_r = _langmuir_step(Wc[i, j, I_R], kx[i, j, KX_GR], g_max * kin[ADS_RESIN], k_l, frac, film, mso_dt, dt)
    return q_a, q_r


@njit(cache=True)
def _langmuir_step(c, g_now, g_max, k_l, frac, film, mso_dt, dt):
    """Объем слоя, адсорбируемого за единицу времени, [1/с]: шаг к изотерме Ленгмюра, не больше растворенного.
    frac - доля пути к равновесию за шаг (кинетика твердой фазы) или A*dt (пленочная). Кинетика твердой фазы -
    неявная по концентрации в нефти ячейки (`langmuir_ldf_step`): остаточной нефти в заводненной зоне мало, и
    явный шаг колебался между полной выборкой растворенного и десорбцией."""
    if film:
        dg = langmuir_film_step(g_now, g_max, k_l, c, frac) - g_now
        if dg > 0.0:
            dg = min(dg, mso_dt * dt * c * ro_o)
    else:
        dg = langmuir_ldf_step(g_now, g_max, k_l, c, mso_dt * dt * ro_o, frac) - g_now  # [кг/м^3 породы]
    return dg / ro_asph_dep / dt


@njit(cache=True)
def _aging(i, j, Wc, Ws, Dep, kx, kin, mso_dt, tmp, new_kx):
    """Старение гель-отложения встречной диффузией (Singh et al., 2000; обзор 2.4).

    Доля парафина в геле C = M_парафин/(ro_p*V_геля) растет к C_max: dC/dt = age_rate*(C_max - C), объем геля
    постоянен - парафин вытесняет из него захваченную нефть. Парафин приходит из растворенного в нефти групп,
    насыщенных при местной температуре (тяжелее критического углеродного числа: у них есть кристаллы).
    Возвращает объем кристаллов за единицу времени, [1/с]; доли групп - в new_kx[KX_GSH:]."""
    v_gel = kx[i, j, KX_VGEL]
    for kk in range(N_W):
        new_kx[i, j, KX_GSH + kk] = 0.0
    if v_gel <= 0.0:
        return 0.0
    m_wax = 0.0
    for kk in range(N_W):
        m_wax += Dep[i, j, kk]
    c_dep = m_wax / ro_p / v_gel
    rate = v_gel * kin[AGE_RATE] * max(kin[AGE_CMAX] - c_dep, 0.0)  # объем кристаллов, [1/с]
    if rate <= 0.0:
        return 0.0
    dis_sat = 0.0
    for kk in range(N_W):
        eq_sus = kx[i, j, KX_WEQ + kk] if wax_kinetics else Ws[i, j, kk]
        tmp[kk] = Wc[i, j, kk] - Ws[i, j, kk] if eq_sus > 0.0 else 0.0
        dis_sat += tmp[kk]
    if dis_sat <= 0.0:
        return 0.0
    for kk in range(N_W):
        new_kx[i, j, KX_GSH + kk] = tmp[kk] / dis_sat
    return min(rate, mso_dt * dis_sat * ro_o / ro_p)


# --- Ядро: глубинная фильтрация ---------------------------------------------------------------------------------

@njit(cache=True)
def _perm(m_new, sigma_v, kin):
    """k/k0 по корреляции `perm_model` (обзор 4.2); sigma_v - объем отложений на единицу объема породы."""
    if perm_model == 'kozeny_carman':
        return perm_kozeny_carman(m_new, init_m)
    elif perm_model == 'power':
        return perm_power(m_new, init_m, kin[PERM_N])
    elif perm_model == 'damage':
        return perm_damage(sigma_v / (kin[PERM_SMAX] * init_m), kin[PERM_BETA], kin[PERM_GAMMA])
    else:
        return perm_surface(m_new, init_m, sigma_v / init_m, kin[PERM_ALPHA])


@njit(cache=True)
def calc_filtration(i, j, S, T, p, m, k, fi, Wc, Ws, Wps, Dep, grad_p, lam_o, kx, kin, integr_r2_fi0, rows,
                    new_qp1, new_qp2, new_fi, new_k, new_m, new_kx, out_o, dt) -> None:
    """Глубинная фильтрация Ивса-Цивана (обзор 3.6, 5.4; Civan, Transp Porous Media 2015; Wang & Civan 2001):

        d sigma/dt = kd*m*S_o*c + kpl*|u_o|*c - ke*sigma*(|u_o| - u_cr)+,

    c - масса взвеси (кристаллов парафина, флокул) в 1 м^3 нефти, sigma - масса отложений в 1 м^3 породы,
    |u_o| - скорость фильтрации нефти. Пористость - за вычетом объема отложений (и слоя адсорбции, стеночной
    кристаллизации, старения), проницаемость - корреляция k(phi, sigma) `perm_model`. Функция пор fi здесь
    не меняется (гель считается по начальному распределению). Выходной контракт - как у `calc_deposition`."""
    tmp = rows[ROW_TMP]
    so = 1.0 - S[i, j]
    mso = m[i, j] * so
    mso_dt = max(mso / dt - out_o / volume, 0.0)
    u_o = lam_o[i, j] * grad_p[i, j]
    ex = max(u_o - kin[FILT_UCR], 0.0)
    for ij in range(Nr):
        new_fi[i, j, ij] = fi[i, j, ij]

    def_rate = kin[FILT_KD] * mso + kin[FILT_KPL] * u_o  # [1/с] на единицу доли взвеси (в долях нефти)
    # парафин: сток по массе взвеси, вынос - по осадку
    wps = Wps[i, j]
    dep_w = 0.0
    for kk in range(N_W):
        dep_w += Dep[i, j, kk]
    dep_rate_w = def_rate * wps * ro_o / ro_p                      # объем кристаллов, [1/с]
    dep_rate_w = min(dep_rate_w, mso_dt * wps * ro_o / ro_p)
    ent_w = min(kin[FILT_KE] * dep_w / ro_p * ex, dep_w / ro_p / dt)
    qp1 = dep_rate_w - ent_w
    qpa = 0.0
    if asphaltenes:
        w_af = Wc[i, j, IA_F]
        dep_a = Dep[i, j, IA_F] + Dep[i, j, I_R]
        rate_a = def_rate * w_af / (1.0 - resin_in_deposit) * ro_o / ro_asph_dep
        avail_a = mso_dt * min(w_af / (1.0 - resin_in_deposit),
                               Wc[i, j, I_R] / resin_in_deposit if resin_in_deposit > 0.0 else 1e300) / _RO_AD_O
        qpa = min(rate_a, avail_a) - min(kin[FILT_KE] * dep_a / ro_asph_dep * ex, dep_a / ro_asph_dep / dt)

    q_wall, q_ada, q_adr, q_age = _wall_sinks(i, j, T, p, m, S, Wc, Ws, Dep, kx, kin, mso, mso_dt, dt, tmp, new_kx)

    new_qp1[i, j] = qp1
    new_qp2[i, j] = 0.0
    new_kx[i, j, KX_QPA] = qpa
    new_kx[i, j, KX_QW] = q_wall
    new_kx[i, j, KX_QG] = q_age
    new_kx[i, j, KX_QADA] = q_ada
    new_kx[i, j, KX_QADR] = q_adr
    new_kx[i, j, KX_VGEL] = kx[i, j, KX_VGEL]
    new_m[i, j] = m[i, j] - (qp1 + qpa + q_wall + q_age + q_ada + q_adr) * dt
    sigma_v = init_m - new_m[i, j]
    new_k[i, j] = init_k * _perm(new_m[i, j], sigma_v, kin)
