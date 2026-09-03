"""Функции относительных фазовы проницаемостей флюидов."""
import numpy as np
from numba import njit

from paraphin.constants import S_min, S_max, n_power, data_type, mu_w, mu_o


@njit(cache=True)
def pf_o(s: data_type) -> data_type:
    """
    Функция отностельной фазовой проницаемости нефти:
               ⎡   Smax - S    ⎤^ n
        pf_o = ⎢———————————————⎢
               ⎣  Smax - Smin  ⎦
    """
    ret = 0.0
    if s < S_min:
        ret = 1.0  # однофазная фильтрация нефти
    elif s > S_max:
        ret = 0.0  # однофазная фильтрация воды
    else:
        ret = ((S_max - s) / (S_max-S_min)) ** n_power

    return ret


@njit(cache=True)
def pf_w(s: data_type) -> data_type:
    """
    Функция отностельной фазовой проницаемости воды:
               ⎡   S - Smin    ⎤^ n
        pf_w = ⎢———————————————⎢
               ⎣  Smax - Smin  ⎦
    """
    ret = 0.0
    if s < S_min:
        ret = 0.0   # однофазная фильтрация нефти
    elif s > S_max:
        ret = 1.0    # однофазная фильтрация воды
    else:
        ret = ((s - S_min) / (S_max-S_min)) ** n_power

    return ret


@njit(cache=True)
def Buckley_Leverett(s: data_type, mu_w: data_type, mu_o: data_type) -> data_type:
    """Функция Баклея-Леверетта:
                      pf_w
        f = —————————————————————————
            pf_w + pf_o * mu_w / mu_o
    """
    return pf_w(s) / (pf_w(s) + pf_o(s) * mu_w / mu_o)


if __name__ == '__main__':
    import plotly.graph_objects as go
    n = 100
    s_arr = np.zeros(dtype=data_type, shape=n)
    f_o = np.zeros(dtype=data_type, shape=n)
    f_w = np.zeros(dtype=data_type, shape=n)
    buck_lev = np.zeros(dtype=data_type, shape=n)

    for i in range(n):
        s = i / n
        s_arr[i] = s
        f_o[i] = pf_o(s)
        f_w[i] = pf_w(s)
        buck_lev[i] = Buckley_Leverett(s, mu_w, mu_o)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_arr, y=f_o, mode='lines', name='ОФП нефти',
                             line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=s_arr, y=f_w, mode='lines', name='ОФП воды',
                             line=dict(color='blue', width=3)))
    # fig.add_trace(go.Scatter(x=s_arr.to_numpy(), y=buck_lev.to_numpy(),
    #                          mode='lines', name='БЛ', line=dict(color='black', width=3)))

    fig.update_layout(plot_bgcolor='white', width=750, height=350, margin=dict(t=0, b=0),
                      xaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1),
                      yaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1),
                      xaxis_title="S", yaxis_title="f")
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="black", width=1)
    )

    fig.update_yaxes(range=[-0.005, 1.01])
    # fig.write_image("ofp.svg", width=750, height=350)
    fig.show()
