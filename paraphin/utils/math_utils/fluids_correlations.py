"""Зависимости псевдокомпонентов задачи от температуры."""
import numpy as np
from numba import njit

from paraphin.constants import data_type, R, phi_max, E_activation, mu_o_ref, T_mu_ref

# Показатель в формуле Кригера-Догерти. Константа уровня модуля: numba вшивает ее литералом,
# а не считает произведение на каждой ячейке каждый шаг.
_KD_EXPONENT = -2.5 * phi_max
_E_OVER_R = E_activation / R
_INV_T_REF = 1.0 / (T_mu_ref + 273.15)


@njit(cache=True)
def calc_mu_o(t: data_type, w_ps: data_type) -> data_type:
    """Вязкость нефтяной фазы, [Па*с].

    Жидкая основа - уравнение Аррениуса, записанное через опорную точку:

        mu_L(T) = mu_o_ref * exp[(E_a/R) * (1/T - 1/T_ref)].

    Такая запись разделяет два независимых параметра: `mu_o_ref` задает уровень вязкости при
    пластовой температуре, а `E_activation` - только крутизну зависимости от температуры.

    Выпавшие кристаллы парафина образуют в жидкой основе суспензию и дополнительно повышают
    вязкость; это учитывается множителем Кригера-Догерти

        mu = mu_L(T) * (1 - phi/phi_max)^(-2.5*phi_max),

    где phi - объемная доля кристаллов. Без этого множителя кристаллизация влияла бы только на
    проницаемость через кольматацию, хотя экспериментально рост вязкости - основной эффект.
    При содержании парафина 5% масс. множитель не превышает 1.13: суспензия разбавленная.

    w_ps - массовая доля взвешенного парафина в нефтяной фазе. Точный пересчет в объемную долю
    твердой фазы дал бы phi = ro_o*w_ps/ro_p, но при ro_p ~= ro_o разница пренебрежимо мала (то
    же приближение, что и для объемной концентрации R в однофазной формуле суффозии, см.
    `equations/Velocity_h.py`), поэтому w_ps подставляется в phi напрямую.
    """
    mu_liquid = mu_o_ref * np.exp(_E_OVER_R * (1.0 / (t + 273.15) - _INV_T_REF))

    if w_ps <= 0.0:
        return mu_liquid

    # Кригер-Догерти расходится при phi -> phi_max, поэтому долю подпираем снизу предела
    phi = min(w_ps, 0.99 * phi_max)

    return mu_liquid * (1.0 - phi / phi_max) ** _KD_EXPONENT


@njit(cache=True)
def calc_mu_w(t: data_type) -> data_type:
    """Вязкость воды, [Pa*c]  уравнение Андраде"""
    return 2.414 * 10 ** -5 * 10 ** (247.8 / (t + 133.15))


@njit(cache=True)
def calc_c_w(t: data_type) -> data_type:
    """"Теплоемкость воды, [Дж/(кг*C)]"""
    return 4217 - 2.15 * t + 0.002 * t ** 2


@njit(cache=True)
def calc_c_o(t: data_type) -> data_type:
    """"Теплоемкость нефти, [Дж/(кг*C)]"""
    return 1800 + 4 * t + 0.01 * t ** 2


@njit(cache=True)
def calc_c_f(t: data_type) -> data_type:
    """"Теплоемкость пласта, [Дж/(кг*C)]"""
    return 800 + 0.75 * t


@njit(cache=True)
def calc_c_p(t: data_type) -> data_type:
    """"Теплоемкость парафина, [Дж/(кг*C)]"""
    return 1840 + 3.56 * t  # (t + 273.15)


# теплопроводности
# oil 0.13 + 0.0005(t-20)
# water 0.58 + 0.001(t-20)
# paraphin 0.25 + 0.0007(t-20)
# formation 2.5 + 0.0008(t-20)

if __name__ == '__main__':
    import plotly.graph_objects as go
    n = 100
    t_0 = 20
    t_n = 70
    t_arr = np.zeros(dtype=data_type, shape=n)
    mu_o = np.zeros(dtype=data_type, shape=n)
    mu_w = np.zeros(dtype=data_type, shape=n)

    for i in range(0, n):
        t = t_0 + i * (t_n - t_0) / (n - 1)
        t_arr[i] = t
        mu_o[i] = calc_mu_o(t, 0.0)
        mu_w[i] = calc_mu_w(t)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_arr, y=mu_w, mode='lines', name='mu_w',
                             line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=t_arr, y=mu_o, mode='lines', name='mu_o',
                             line=dict(color='red', width=3)))

    fig.update_layout(plot_bgcolor='white', width=750, height=350, margin=dict(t=0, b=0),
                      xaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1),
                      yaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1, type='log'),
                      xaxis_title="T, °C",
                      yaxis_title="μ, Па∙с"
                      )
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="black", width=1)
    )

    # fig.write_image("mu.svg", width=750, height=350)
    fig.show()
