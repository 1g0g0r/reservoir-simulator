"""Формулы кинетики осаждения без привязки к флагам: их проверяют юнит-тесты (`tests/test_kinetics_math.py`)
в основном процессе, а ядра `Deposition.py`, `Components.py` вызывают их из скомпилированного кода.

Источники - обзор `твт_статья/full_review.pdf` и работы, указанные у каждой функции; вывод и допущения -
`docs/модель_АСПО.md`, разд. 13.
"""
import math

from numba import njit

from paraphin.constants import k_B, g, Lk


# --- Релаксация и кристаллизация -------------------------------------------------------------------------------

@njit(cache=True)
def relax_exp(x, x_eq, rate, dt):
    """Точное решение dx/dt = rate*(x_eq - x) за шаг dt при постоянных x_eq и rate (кинетика кристаллизации
    Huang et al., AIChE J 2011, 57:2955: dC_S/dt = k*(C_eq - C_S); флокуляция; адсорбция с линейной движущей силой)."""
    return x_eq + (x - x_eq) * math.exp(-rate * dt)


@njit(cache=True)
def wall_fraction(k_wall, k_bulk, dt):
    """Доля пересыщения, ушедшая за шаг на стенки, когда его расходуют два параллельных пути первого порядка:
    на стенки (k_wall) и в объем (k_bulk). Пересыщение убывает как exp(-(k_wall + k_bulk)*t), стенкам достается
    k_wall/(k_wall + k_bulk) убыли (Huang et al., 2011: конкуренция осаждения на стенке и выпадения в объеме)."""
    k_sum = k_wall + k_bulk
    if k_sum <= 0.0:
        return 0.0
    return k_wall / k_sum * (1.0 - math.exp(-k_sum * dt))


# --- Частицы: диффузия, оседание, перенос к стенке ---------------------------------------------------------------

@njit(cache=True)
def brownian_diffusivity(T_K, mu, d_p):
    """Коэффициент броуновской диффузии частицы по Стоксу-Эйнштейну k_B*T/(3*pi*mu*d), [м^2/с] (обзор 2.2.3)."""
    return k_B * T_K / (3.0 * math.pi * mu * d_p)


@njit(cache=True)
def shear_diffusivity(coef, phi, gamma_w, a):
    """Коэффициент сдвиговой дисперсии частиц coef*phi^2*gamma*a^2, [м^2/с] (Leighton & Acrivos, J Fluid Mech 1987,
    181:415: самодиффузия частиц радиуса a в сдвиговом течении при малой объемной доле phi; обзор 2.2.2)."""
    return coef * phi * phi * gamma_w * a * a


@njit(cache=True)
def stokes_velocity(a, d_rho, mu):
    """Скорость гравитационного оседания частицы радиуса a по Стоксу 2*a^2*|d_rho|*g/(9*mu), [м/с] (обзор 2.2.4)."""
    return 2.0 * a * a * abs(d_rho) * g / (9.0 * mu)


@njit(cache=True)
def leveque_velocity(um, r, d_eff):
    """Коэффициент массоотдачи частиц к стенке капилляра радиуса r при средней скорости um (решение Левека для
    развитого ламинарного течения): (2*um*D^2/(r*Lk))^(1/3), [м/с]. Это формула прежней модели
    (`Velocity_h.calc_velocities_h`), записанная через D_eff - сумму броуновской и сдвиговой диффузии."""
    return (2.0 * um * d_eff * d_eff / (r * Lk)) ** (1.0 / 3.0)


# --- Агрегация флокул --------------------------------------------------------------------------------------------

@njit(cache=True)
def coagulation_kernel(T_K, mu, stability):
    """Броуновское ядро Смолуховского для одинаковых частиц 8*k_B*T/(3*mu*W), [м^3/с]; W - коэффициент
    устойчивости Фукса (1 - диффузионно-лимитированная агрегация, DLCA; W >> 1 - реакционно-лимитированная, RLCA;
    обзор 2.3.3, 5.2)."""
    return 8.0 * k_B * T_K / (3.0 * mu * stability)


@njit(cache=True)
def smoluchowski_step(n_vol, k_coag, dt):
    """Число частиц в единице объема через dt по уравнению Смолуховского с постоянным ядром dN/dt = -K*N^2/2:
    N(t + dt) = N/(1 + K*N*dt/2) - точно, поэтому два полушага дают то же, что один шаг."""
    return n_vol / (1.0 + 0.5 * k_coag * n_vol * dt)


@njit(cache=True)
def floc_size(m_floc, d0, df, rho):
    """Диаметр фрактальной флокулы массы m_floc из первичных частиц диаметра d0: d = d0*(m/m0)^(1/D_f), [м];
    не меньше первичной частицы (обзор 5.2: DLCA D_f ~ 1.8, RLCA ~ 2.1)."""
    m0 = rho * math.pi * d0 * d0 * d0 / 6.0
    return d0 * max(m_floc / m0, 1.0) ** (1.0 / df)


# --- Адсорбция ---------------------------------------------------------------------------------------------------

@njit(cache=True)
def langmuir_constant(k_ref, d_h, T_K, T_ref_K, R):
    """Константа Ленгмюра по Вант-Гоффу K(T) = K_ref*exp(-dH/R*(1/T - 1/T_ref)); при dH < 0 (адсорбция
    экзотермична) K растет с охлаждением."""
    return k_ref * math.exp(-d_h / R * (1.0 / T_K - 1.0 / T_ref_K))


@njit(cache=True)
def langmuir_eq(g_max, k_l, c):
    """Равновесная адсорбция G_max*K*c/(1 + K*c) (изотерма Ленгмюра)."""
    return g_max * k_l * c / (1.0 + k_l * c)


@njit(cache=True)
def langmuir_film_step(g, g_max, k_l, c, a_dt):
    """Шаг пленочной кинетики адсорбции (массообмен со стороны нефти, линейная движущая сила Глюкауфа):

        dG/dt = A*(c - c*(G)),   c*(G) = G/(K*(G_max - G))  - обратная изотерма Ленгмюра,

    неявным Эйлером: G_new - меньший корень K*x^2 - (K*(G_max + g) + a_dt*(K*c + 1))*x + K*G_max*(g + a_dt*c) = 0,
    он всегда лежит в [0, G_max). a_dt = A*dt, [кг/м^3 породы на единицу массовой доли]. В отличие от кинетики
    со стороны твердой фазы dG/dt = k*(G_eq - G) скорость здесь не падает при подходе к равновесию, пока изотерма
    выпуклая (K*c >> 1): удержание идет с постоянной скоростью и резко останавливается при насыщении. При линейной
    изотерме (K*c << 1) обе формы совпадают с k = A/(G_max*K)."""
    b = k_l * (g_max + g) + a_dt * (k_l * c + 1.0)
    cc = k_l * g_max * (g + a_dt * c)
    disc = max(b * b - 4.0 * k_l * cc, 0.0)
    return 2.0 * cc / (b + math.sqrt(disc)) if b > 0.0 else g


# --- Проницаемость от отложений (обзор 4.2) ---------------------------------------------------------------------

@njit(cache=True)
def perm_kozeny_carman(m, m0):
    """k/k0 = (m/m0)^3*((1 - m0)/(1 - m))^2 - относительная форма Козени-Кармана."""
    x = m / m0
    return x * x * x * ((1.0 - m0) / (1.0 - m)) ** 2


@njit(cache=True)
def perm_power(m, m0, n):
    """k/k0 = (m/m0)^n - степенная поправка (Krauss & Mays, 2013; n = 8-19 по обзору 4.2)."""
    return (m / m0) ** n


@njit(cache=True)
def perm_damage(sigma_rel, beta, gam):
    """k/k0 = (1 - beta*sigma/sigma_max)^gamma - функция повреждения (Civan, Transp Porous Media 2015);
    sigma_rel = sigma/sigma_max."""
    return max(1.0 - beta * sigma_rel, 0.0) ** gam


@njit(cache=True)
def perm_surface(m, m0, sigma_rel0, alpha):
    """Козени-Карман с ростом удельной поверхности отложениями S_v = S_v0*(1 + alpha*sigma/m0) (обзор 4.2:
    S_v^eff = S_v0 + sum alpha_j*sigma_j); sigma_rel0 = sigma/m0."""
    sv = 1.0 + alpha * sigma_rel0
    return perm_kozeny_carman(m, m0) / (sv * sv)


# --- Скважина ----------------------------------------------------------------------------------------------------

@njit(cache=True)
def hawkins_skin(k0, k_s, r_s, r_w):
    """Скин-фактор поврежденной зоны радиуса r_s вокруг скважины радиуса r_w (Hawkins, 1956; обзор 4.3):
    S = (k0/k_s - 1)*ln(r_s/r_w)."""
    return (k0 / k_s - 1.0) * math.log(r_s / r_w)
