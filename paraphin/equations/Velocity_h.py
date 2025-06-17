"""Вычисление скоростей и толщины осадочного слоя в ячейке."""
import taichi as ti

from paraphin import r1, r2
from paraphin.constants import data_type, Nr, dt, ro_p, D, g, gamma, betta, Diff, Lk, Cf, S_max, Delta

"""
Lk: float
    средняя длина капилляра, [м]
gamma: float
    Отношение радиуса горла к радиусу канала
delta: float
    Кинетическая константа суффозии, [1/м]
Diff: float
    Коэффициент диффузионного осаждения частиц, [м2/сек]
D: float
    Размер частицы(диаметр частицы), [м]
Cf: float
    Коэффициент сопротивления частицы в нефти, [-]
Delta: float
    Кинетическая константа суффозии, [1/м]
eta: float
    Коэффициент извилистости, [-]
"""

b_D_3 = 6.0 * betta / D / D / D
cf_D2 = Cf * D * D * g / 18.0
Diff_2 = 2.0 * Diff * Diff / Lk
D_2_gamma = D * 0.5 / gamma
So_max = 1.0 - S_max


@ti.func
def calc_velocitys_h(i, j, S, Um_r2, Wps, mu_o, fi, h_sloy, Ur, h_sloy_new, Ur_new, Ub_new) -> None:
    """Вычисление скоростей и толщины осадочного слоя в ячейке.

    Parameters
    ----------
    i, j : int
        Индексы текущей ячейки, [-]
    S: taichi.field(Nx, Ny)
        Водонасыщенность, [-]
    Um_r2: taichi.field(Nx, Ny)
         Средняя скорость в капилляре без множителя r^2, [1/(с*м)]
    Wps: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина, [-]
    mu_o: taichi.field(Nx, Ny)
        Вязкость нефти, [Па*с]
    fi: taichi.field(Nx, Ny, Nr)
        Функция распределения пор по размеру, [-]
    h_sloy: : taichi.field(Nx, Ny, Nr)
         Толщина осадочного слоя, [m]
    Ur: taichi.field(Nx, Ny, Nr)
        Скорость изменения радиуса капилляра, [м/с]
    h_sloy_new: : taichi.field(Nx, Ny, Nr)
         Толщина осадочного слоя на новом временном слое, [m]
    Ub_new: taichi.field(Nx, Ny, Nr)
        Скорость блокирования капилляра на новом временном слое, [м/с]
    Ur_new: taichi.field(Nx, Ny, Nr)
        Скорость изменения радиуса капилляра на новом временном слое, [м/с]
    """
    # Тк при Wps=0 цикл не имеет смысла
    if Wps[i, j] > 1e-6:
        So = 1.0 - S[i, j] - So_max
        for ij in ti.ndrange(Nr):
            um = Um_r2[i, j] * r2[ij]
            uc = u_c(r=r1[ij], mu=mu_o[i, j], ro=ro_p)
            Ub_new[i, j, ij] = u_b(So=So, um=um, wps=Wps[i, j], fi=fi[i, j, ij], r=r1[ij])
            Ur_new[i, j, ij] = u_r(So=So, wps=Wps[i, j], um=um, uc=uc, r=r1[ij], h=h_sloy[i, j, ij])
            h_sloy_new[i, j, ij] = sed_h(h0=h_sloy[i, j, ij], ur=Ur[i, j, ij], r=r1[ij])


@ti.func
def u_r(So: data_type, wps: data_type, um: data_type, uc: data_type, r: data_type, h: data_type) -> data_type:
    """Скорость изменения радиуса капилляра.

    Parameters
    ----------
    So: float
        Нефтенасыщенность, [-]
    wps: float
        Объемная концентрация взвешенных частиц парафина, [м3/м3]
    um: float
        Среднее значение скорости жидкости в канале, [м/c]
    uc: float
        Критическая скорость жидкости в канале, [м/c]
    r: float
        Радиус капилляра, [м]
    h: float
        Толщина осадочного слоя, [м]

    Returns
    -------
    ur: float
        Скорость изменения радиуса капилляра, [м/с]
    """
    ur = 0.0
    if r >= D_2_gamma:
        # Сужение (кольматация) каналов
        ur = -So * wps * (um * Diff_2 / r) ** (1.0/3.0)

        # Расширение (суффозия) каналов
        # if um > uc and h > 0.0:
        #     ur += So * Delta * (um - uc) * h * (r + h * 0.5) / r

    return ur


@ti.func
def u_b(So: data_type, um: data_type, wps: data_type, fi: data_type, r: data_type) -> data_type:
    """Скорость блокирования капилляров.

    Parameters
    ----------
    So: float
        Нефтенасыщенность, [-]
    um: float
        Средняя скорость жидкости в капилляре, [м/с]
    wps: float
        Концентрация частиц в потоке, [м3/м3]
    fi: float
        Значение функции распределения пор по размерам
    r: float
        Радиус капилляра, [м]

    Returns
    -------
    ub: float
        Скорость блокирования капилляров, [м/с]
    """
    ub = 0.0
    if r <= D_2_gamma:
        ub = So * wps * r * r * fi * um * b_D_3

    return ub


@ti.func
def u_c(r: data_type, mu: data_type, ro: data_type) -> data_type:
    """Критическая скорость.

    Parameters
    ----------
    r: float
        Радиус капилляра, [м]
    mu: float
        Вязкость нефти, [Па/с]
    ro: float
        Плотность парафина, [Кг/м^3]

    Return
    ------
    uc: float
        Критическая скорость, [м/с]
    """
    uc = 0.0
    x0 = 0.5 * D / r
    if x0 < 1.0:
        x = 1.0 - x0
        uc =  cf_D2 * ro / (mu * (1.0 - x * x))

    return uc


@ti.func
def sed_h(h0: data_type, ur: data_type, r: data_type) -> data_type:
    """
    Вычисление толщины осадочного слоя.

    Parameters
    ----------
    h0: float
        Толщина осадочного слоя, [m].
    ur: float
        Скорость изменения радиуса капилляра, [m/c]
    r: float
        Радиус капилляра, [m].

    Returns
    -------
    hr: float
        Толщина осадочного слоя, [m].
    """
    hr = h0 - dt * ur
    hr = ti.max(0.0, ti.min(hr, r - 1e-7))
    return hr
