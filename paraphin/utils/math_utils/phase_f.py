import taichi as ti

from paraphin.constants import S_min, S_max, n_power, data_type, mu_w, mu_o


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
    fig.add_trace(go.Scatter(x=s_arr.to_numpy(), y=f_o.to_numpy(), mode='lines', name='ОФП нефти',
                             line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=s_arr.to_numpy(), y=f_w.to_numpy(), mode='lines', name='ОФП воды',
                             line=dict(color='blue', width=3)))
    # fig.add_trace(go.Scatter(x=s_arr.to_numpy(), y=f_w.to_numpy()/(f_w.to_numpy()+mu_w/mu_o*f_o.to_numpy()),
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
    fig.write_image("ofp.svg", width=750, height=350)
    fig.show()
