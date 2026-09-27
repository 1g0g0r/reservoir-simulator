"""Локальное тепловое неравновесие флюидов и породы (`thermal_nonequilibrium`; обзор 4.5).

В основной модели флюиды и порода в каждой точке имеют одну температуру (LTE, обзор 3.7). Здесь у породы своя
температура T_s (поле `kx[..., KX_TS]`), а теплообмен между ними идет с объемным коэффициентом h_v:

    C_f*dT/dt   = (конвекция, теплопроводность, скрытая теплота, потери) - h_v*(T - T_s),
    C_s*dT_s/dt = h_v*(T - T_s),

C_f - теплоемкость флюидов и отложений в единице объема пласта (psi без породы), C_s = (1 - m0)*ro_f*C_f.
Теплопроводность и потери через кровлю и подошву отнесены к уравнению флюидов (при h_v -> inf это ровно LTE).
Обмен жесткий: время релаксации C/h_v - доли секунды при зернах 0.1-1 мм, поэтому он берется точным
решением за шаг, сохраняющим энергию:

    T_eq = (C_f*T + C_s*T_s)/(C_f + C_s),  (T - T_s)(t + dt) = (T - T_s)(t)*exp(-h_v*(1/C_f + 1/C_s)*dt).

Коэффициент теплообмена - корреляция Wakao & Kaguei (Heat and Mass Transfer in Packed Beds, 1982):
    Nu = 2 + 1.1*Re^0.6*Pr^(1/3),  h_v = a_v*Nu*lam_f/d_g,  a_v = 6*(1 - m)/d_g,
Re = ro*|u|*d_g/mu по скорости фильтрации. Молекулярная диффузия парафина к более холодной породе (Фик,
обзор 2.2.1) - в `Deposition._wall_crystallization` с той же корреляцией для числа Шервуда.

Проверка - аналитическое решение Шумана (Schumann, J Franklin Inst 1929, 208:405) для слоя с теплообменом
без теплопроводности: `experiments/`.
"""
import math

from numba import njit

from paraphin.constants import init_m, ro_f, ro_o, ro_w, K_o, K_w, heat_losses, wax_components, volume
from paraphin.kinetics_params import LTNE_DG, KX_TS
from .Temperature import psi_cell, LATENT, _heat_losses_lauwerier, _heat_losses_vw


@njit(cache=True)
def interphase_h(m, S, u, mu, c_f, kin):
    """Объемный коэффициент теплообмена флюид - порода h_v по Wakao & Kaguei (1982), [Вт/(м^3*C)].
    u - модуль скорости фильтрации, [м/с]; mu, c_f - вязкость [Па*с] и удельная теплоемкость [Дж/(кг*C)] флюида."""
    d_g = kin[LTNE_DG]
    lam_f = S * K_w + (1.0 - S) * K_o
    rho = S * ro_w + (1.0 - S) * ro_o
    re = rho * u * d_g / mu
    pr = c_f * mu / lam_f
    nu = 2.0 + 1.1 * re ** 0.6 * pr ** (1.0 / 3.0)
    return 6.0 * (1.0 - m) / d_g * nu * lam_f / d_g


@njit(cache=True)
def exchange_step(t_f, t_s, c_f, c_s, h_v, dt):
    """Точное решение теплообмена двух емкостей за шаг: энергия C_f*T + C_s*T_s сохраняется, разность
    температур убывает экспоненциально. Возвращает (T, T_s) в конце шага."""
    if c_f <= 0.0 or c_s <= 0.0:
        return t_f, t_s
    t_eq = (c_f * t_f + c_s * t_s) / (c_f + c_s)
    diff = (t_f - t_s) * math.exp(-h_v * (1.0 / c_f + 1.0 / c_s) * dt)
    return t_eq + c_s / (c_f + c_s) * diff, t_eq - c_f / (c_f + c_s) * diff


@njit(cache=True)
def temperature_equation_ltne(i, j, T, T_0, m, S, C_o, C_w, C_f, C_p, Wo, Wp, Wps, new_Wp, new_Wps, Hl, new_Hl,
                              cells_T_eq, t, E_ff, new_T, new_m, new_S, kx, new_kx, u_abs, mu_o, kin, dt):
    """Уравнение энергии флюидов (как `temperature_equation`, но без теплоемкости породы) и обмен с породой.

    Возвращает psi - полную теплоемкость ячейки текущего слоя (флюиды + порода): ее, как и в `temperature_equation`,
    берет ограничение Куранта по температуре."""
    c_rock = (1.0 - init_m) * ro_f * C_f[i, j]
    psi = psi_cell(i, j, m, S, Wo[i, j], Wp[i, j], Wps[i, j], C_w, C_o, C_p, C_f)
    psi_next = psi_cell(i, j, new_m, new_S, 1.0 - new_Wp[i, j] - new_Wps[i, j],
                        new_Wp[i, j], new_Wps[i, j], C_w, C_o, C_p, C_f)
    cf_now, cf_next = psi - c_rock, psi_next - c_rock
    derivative_add = T[i, j] * (cf_next - cf_now) / dt

    carrier, carrier_new = (Hl[i, j], new_Hl[i, j]) if wax_components else (Wp[i, j], new_Wp[i, j])
    latent = LATENT * (m[i, j] * (1.0 - S[i, j]) * carrier - new_m[i, j] * (1.0 - new_S[i, j]) * carrier_new) / dt

    if heat_losses == 0:
        t_losses = 0.0
    elif heat_losses == 1:
        t_losses = _heat_losses_lauwerier(i, j, t, T)
    else:
        t_losses = _heat_losses_vw(i, j, t, T, T_0, E_ff, dt)

    t_f = T[i, j] + dt / cf_next * (cells_T_eq[i, j] / volume - derivative_add - t_losses + latent)
    h_v = interphase_h(new_m[i, j], new_S[i, j], u_abs, mu_o[i, j], C_o[i, j], kin)
    t_f, t_s = exchange_step(t_f, kx[i, j, KX_TS], cf_next, c_rock, h_v, dt)
    new_T[i, j] = t_f
    new_kx[i, j, KX_TS] = t_s
    return psi
