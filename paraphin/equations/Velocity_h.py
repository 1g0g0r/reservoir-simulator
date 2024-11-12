import numpy as np
import taichi as ti

from paraphin.constants import default_type, Nx, Ny, Nr, dt, ro_p, D, gamma, betta, Diff, Lk, Cf, Delta, eta

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
    Коэффициент сопротивления частицы в нефти
Delta: float
    Кинетическая константа суффозии, [1/м]
eta: float
    Коэффициент извилистости, [-]
"""

# Операции с константными величинами (вычисляются один раз только при импорте модуля)
b_D_3 = 6.0 * betta / D / D / D
cf_D2 = Cf * D * D * 9.81 / 18.0
Diff_2 = 2.0 * Diff * Diff / Lk


def calc_velocitys_h(p, Wps, mu_o, fi, r, h_sloy, Ur, Ub) -> (ti.field(dtype=default_type, shape=(Nx, Ny, Nr)),
                                                              ti.field(dtype=default_type, shape=(Nx, Ny, Nr)),
                                                              ti.field(dtype=default_type, shape=(Nx, Ny, Nr))):
    """Вычисление скоростей и толщины осадочного слоя.

    Parameters
    ----------
    p: taichi.field(Nx, Ny)
        Давление, [Па]
    Wps: taichi.field(Nx, Ny)
        Концентрация взвешенных частиц парафина, [-]
    mu_o: taichi.field(Nx, Ny)
        Вязкость нефти, [Па*с]
    fi: taichi.field(Nx, Ny, Nr)
        Функция распределения пор по размеру, [-]
    r: taichi.field(Nr)
        Радиусы пор, [m]
    h_sloy: : taichi.field(Nx, Ny, Nr)
         Толщина осадочного слоя, [m]
    Ub: taichi.field(Nx, Ny, Nr)
        Скорость блокирования капилляра, [м/с]
    Ur: taichi.field(Nx, Ny, Nr)
        Скорость изменения радиуса капилляра, [м/с]

    Returns
    -------
    h_sloy: : taichi.field(Nx, Ny, Nr)
         Толщина осадочного слоя на новом временном слое, [m]
    Ub: taichi.field(Nx, Ny, Nr)
        Скорость блокирования капилляра на новом временном слое, [м/с]
    Ur: taichi.field(Nx, Ny, Nr)
        Скорость изменения радиуса капилляра на новом временном слое, [м/с]
    """
    Um_r2 = ti.field(dtype=default_type, shape=(Nx, Ny))
    Um_r2.from_numpy(np.linalg.norm(np.gradient(p.to_numpy()), axis=0) * 0.125 / eta / mu_o.to_numpy())
    r2 = ti.field(dtype=default_type, shape=Nr)
    r2.from_numpy(r.to_numpy() * r.to_numpy())

    @ti.kernel
    def calc_velocitys_h_loop():
        for i, j, ij in ti.ndrange(Nx, Ny, Nr):
            um = Um_r2[i, j] * r2[ij]
            uc = u_c(r[ij], mu_o[i, j], ro_p)
            Ub[i, j, ij] = u_b(um, Wps[i, j], fi[i, j, ij], r[ij])
            Ur[i, j, ij] = u_r(Wps[i, j], um, uc, r[ij], h_sloy[i, j, ij])
            h_sloy[i, j, ij] = sed_h(h_sloy[i, j, ij], Ur[i, j, ij], r[ij])

    calc_velocitys_h_loop()

    return h_sloy, Ur, Ub


@ti.func
def u_r(wps: default_type, um: default_type, uc: default_type, r: default_type, h: default_type) -> default_type:
    """Скорость изменения радиуса капилляра.

    Parameters
    ----------
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
    if 2.0 * r * gamma > D:
        # Сужение (кольматация) каналов
        # Ur = -wps * (Um * 2 * Diff ** 2 / (r * Lk)) ** (1/3)
        ur = -wps * (um * Diff_2 / r) ** (1/3)

        # Расширение (суффозия) каналов
        if um > uc and h > 0:
            ur += Delta * (um - uc) * h * (r + h * 0.5) / r

    return ur


@ti.func
def u_b(um: default_type, wps: default_type, fi: default_type, r: default_type) -> default_type:
    """Скорость блокирования капилляров.

    Parameters
    ----------
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
    if 2.0 * r * gamma <= D:
        # 6.0 * betta * wps * r * r * fi * Um / D**3
        ub = wps * r * r * fi * um * b_D_3

    return ub


@ti.func
def u_c(r: default_type, mu: default_type, ro: default_type) -> default_type:
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
    if x0 <= 1.0:
        x = 1.0 - x0
        # uc = Cf * D * D * 9.81 / 18.0 * ro / (mu * (1.0 - x * x))
        uc =  cf_D2 * ro / (mu * (1.0 - x * x))

    return uc


@ti.func
def sed_h(h0: default_type, ur: default_type, r: default_type) -> default_type:
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
    hr = max(0.0, min(hr, r - 1e-7))
    return hr
