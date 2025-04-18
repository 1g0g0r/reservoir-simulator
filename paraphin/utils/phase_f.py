import taichi as ti

from paraphin.constants import S_min, S_max, n_power, data_type


@ti.func
def pf_o(s: data_type) -> data_type:
    """
    Функция отностельной фазовой проницаемости нефти
        [(Smax-S)/(Smax-Smin)]^2
    """
    ret = 0.0
    if s < S_min:
        ret = 1.0  # однофазная фильтрация нефти
    elif s > S_max:
        ret = 0.0  # однофазная фильтрация воды
    else:
        ret = ((S_max - s) / (S_max-S_min)) ** n_power

    return ret


@ti.func
def pf_w(s: data_type) -> data_type:
    """
    Функция отностельной фазовой проницаемости воды
        [(S-Smin)/(Smax-Smin)]^2
    """
    ret = 0.0
    if s < S_min:
        ret = 0.0   # однофазная фильтрация нефти
    elif s > S_max:
        ret = 1.0    # однофазная фильтрация воды
    else:
        ret = ((s - S_min) / (S_max-S_min)) ** n_power

    return ret


def _pf_o(s: data_type) -> data_type:
    """
    Функция отностельной фазовой проницаемости нефти
        [(Smax-S)/(Smax-Smin)]^2
    """
    ret = 0.0
    if s < S_min:
        ret = 1.0  # однофазная фильтрация нефти
    elif s > S_max:
        ret = 0.0  # однофазная фильтрация воды
    else:
        ret = ((S_max - s) / (S_max-S_min)) ** n_power

    return ret


def _pf_w(s: data_type) -> data_type:
    """
    Функция отностельной фазовой проницаемости воды
        [(S-Smin)/(Smax-Smin)]^2
    """
    ret = 0.0
    if s < S_min:
        ret = 0.0   # однофазная фильтрация нефти
    elif s > S_max:
        ret = 1.0    # однофазная фильтрация воды
    else:
        ret = ((s - S_min) / (S_max-S_min)) ** n_power

    return ret


if __name__ == '__main__':
    import numpy as np
    import plotly.graph_objects as go
    s_arr = np.linspace(0, 1, 101)
    f_o = np.array([_pf_o(i) for i in s_arr])
    f_w = np.array([_pf_w(i) for i in s_arr])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_arr, y=f_o, mode='lines', name='нефть', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=s_arr, y=f_w, mode='lines', name='вода', line=dict(color='blue')))

    fig.show()
