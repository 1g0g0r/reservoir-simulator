"""Вычисление скоростей и толщины осадочного слоя в ячейке."""
from numba import njit

from paraphin import r1, r4, cbrt_r1, n_block, n_narrow
from paraphin.constants import (data_type, Nr, D, g, betta, Diff, Lk, Cf, S_max, Delta, ro_p,
                                min_Wps_bound, suffusion)


b_D_3 = 6.0 * betta / D / D / D
cf_D2 = Cf * D * D * g / 18.0
Diff_2 = 2.0 * Diff * Diff / Lk
So_max = 1.0 - S_max


@njit(cache=True)
def calc_velocities_h(i, j, S, Um_r2, Wps, mu_o, fi, h_sloy, Ur, h_sloy_new, Ur_new, Ub_new, dt) -> None:
    """Скорости блокирования и сужения капилляров и толщина осадочного слоя в ячейке.

    Узкие капилляры частица затыкает целиком (Ub), в широкие проходит и оседает на стенке,
    сужая их (Ur < 0). Граница - радиус, при котором частица проходит горло; критерий и деление
    сетки радиусов на два диапазона - в `paraphin/__init__.py` (`r_pass`, `n_block`, `n_narrow`).
    При Wps ниже порога кольматации цикл не имеет смысла и не выполняется.

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    # Тк при Wps=0 цикл не имеет смысла
    if Wps[i, j] > min_Wps_bound:
        So = 1.0 - S[i, j] - So_max
        wps = Wps[i, j]
        um_r2 = Um_r2[i, j]

        # Оба множителя зависят только от ячейки, зависимость от радиуса вынесена в константные
        # массивы. Скорость блокирования: So*wps*b_D_3 * um*r^2 * fi, где um = um_r2*r^2, то есть
        # um*r^2 = um_r2 * r^4. Скорость сужения: -So*wps * (um*Diff_2/r)^(1/3), где
        # um*Diff_2/r = um_r2*Diff_2*r, то есть корень распадается на cbrt(um_r2*Diff_2)*cbrt(r).
        # Было Nr вызовов pow на ячейку, стало один.
        ub_coef = So * wps * b_D_3 * um_r2
        ur_coef = -So * wps * (um_r2 * Diff_2) ** (1.0 / 3.0)

        # Блокирование - только узкие капилляры (r <= r_pass), сужение - только широкие
        for ij in range(n_block):
            Ub_new[i, j, ij] = ub_coef * r4[ij] * fi[i, j, ij]
        for ij in range(n_block, Nr):
            Ub_new[i, j, ij] = 0.0
        for ij in range(n_narrow):
            Ur_new[i, j, ij] = 0.0
        for ij in range(n_narrow, Nr):
            Ur_new[i, j, ij] = u_r(ur_coef * cbrt_r1[ij], So, um_r2, r1[ij],
                                   h_sloy[i, j, ij], mu_o[i, j])

        for ij in range(Nr):
            h_sloy_new[i, j, ij] = sed_h(h0=h_sloy[i, j, ij], ur=Ur[i, j, ij], r=r1[ij], dt=dt)


@njit(cache=True)
def u_r(ur_narrowing: data_type, So: data_type, um_r2: data_type, r: data_type,
        h: data_type, mu: data_type) -> data_type:
    """Скорость изменения радиуса капилляра, [м/с]: сужение минус вынос.

    Сужение (кольматация) приходит готовым в `ur_narrowing`: множитель -So*wps*cbrt(um_r2*Diff_2)
    от радиуса не зависит и считается один раз на ячейку, здесь остается только умножение на
    cbrt(r) - см. `calc_velocities_h`.

    Расширение (суффозия, вынос осевших частиц потоком) включается флагом `suffusion` в
    `constants.py`. Флаг - константа времени компиляции, поэтому при выключенной суффозии numba
    выкидывает всю ветку целиком: ни `u_c`, ни `um` не считаются. Раньше `u_c` вызывалась
    безусловно на каждой паре (ячейка, радиус), а результат никуда не шел.

    Parameters
    ----------
    ur_narrowing: float
        Готовая скорость сужения для этого радиуса, [м/с]
    So: float
        Нефтенасыщенность, [-]
    um_r2: float
        Средняя скорость в капилляре без множителя r^2, [1/(м*с)]
    r: float
        Радиус капилляра, [м]
    h: float
        Толщина осадочного слоя, [м]
    mu: float
        Вязкость нефти, [Па*с]

    Returns
    -------
    ur: float
        Скорость изменения радиуса капилляра, [м/с]
    """
    ur = ur_narrowing

    if suffusion:
        um = um_r2 * r * r
        uc = u_c(r=r, mu=mu, ro=ro_p)
        if um > uc and h > 0.0:
            ur += So * Delta * (um - uc) * h * (r + h * 0.5) / r

    return ur


@njit(cache=True)
def u_c(r: data_type, mu: data_type, ro: data_type) -> data_type:
    """Критическая скорость потока в капилляре, [м/с]. Нужна только суффозии (`u_r`).

    Parameters
    ----------
    r: float
        Радиус капилляра, [м]
    mu: float
        Вязкость нефти, [Па*с]
    ro: float
        Плотность парафина, [кг/м^3]

    Return
    ------
    uc: float
        Критическая скорость, [м/с]
    """
    uc = 0.0
    if r > 0.0:
        x0 = 0.5 * D / r
        if x0 < 1.0:
            x = 1.0 - x0
            uc = cf_D2 * ro / (mu * (1.0 - x * x))

    return uc


@njit(cache=True)
def sed_h(h0: data_type, ur: data_type, r: data_type, dt: data_type) -> data_type:
    """Вычисление толщины осадочного слоя.

    Parameters
    ----------
    h0: float
        Толщина осадочного слоя, [м]
    ur: float
        Скорость изменения радиуса капилляра, [м/с]
    r: float
        Радиус капилляра, [м]
    dt: float
        Текущий шаг по времени, [с]

    Returns
    -------
    hr: float
        Толщина осадочного слоя на новом временном слое, [м]
    """
    hr = h0 - dt * ur
    hr = max(0.0, min(hr, r - 1e-7))

    return hr
