import taichi as ti

from paraphin.constants import S_min, S_max, n_power, data_type


@ti.func
def pf_o(s: data_type) -> data_type:
    """
    Функция отностельной фазовой проницаемости нефти
        [(Smax-S)/(Smax-Smin)]^n
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
        [(S-Smin)/(Smax-Smin)]^n
    """
    ret = 0.0
    if s < S_min:
        ret = 0.0   # однофазная фильтрация нефти
    elif s > S_max:
        ret = 1.0    # однофазная фильтрация воды
    else:
        ret = ((s - S_min) / (S_max-S_min)) ** n_power

    return ret


@ti.func
def Buckley_Leverett(s: data_type, mu_w: data_type, mu_o: data_type) -> data_type:
    """Функция Баклея-Леверетта
                      pf_w
        f = -------------------------
            pf_w + pf_o * mu_w / mu_o
    """
    return pf_w(s) / (pf_w(s) + pf_o(s) * mu_w / mu_o)


if __name__ == '__main__':
    import plotly.graph_objects as go
    n = 100
    s_arr = ti.field(dtype=data_type, shape=n)
    f_o = ti.field(dtype=data_type, shape=n)
    f_w = ti.field(dtype=data_type, shape=n)

    @ti.kernel
    def calc_data():
        for i in range(n):
            s = i / n
            s_arr[i] = s
            f_o[i] = pf_o(s)
            f_w[i] = pf_w(s)

    calc_data()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_arr.to_numpy(), y=f_o.to_numpy(), mode='lines', name='нефть', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=s_arr.to_numpy(), y=f_w.to_numpy(), mode='lines', name='вода', line=dict(color='blue')))
    fig.show()
