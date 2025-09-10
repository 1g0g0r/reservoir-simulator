"""Зависимости псевдокомпонентов задачи от температуры."""
import taichi as ti

from paraphin.constants import data_type, R


@ti.func
def calc_mu_o(t: data_type) -> data_type:
    """Вязкость нефти, [Pa*c] Уравнение Аррениуса."""
    return 0.001 * ti.exp(5000 / R / (t + 273.15))


@ti.func
def calc_mu_w(t: data_type) -> data_type:
    """Вязкость воды, [Pa*c]  уравнение Андраде"""
    return 2.414 * 10 ** -5 * 10 ** (247.8 / (t + 133.15))


@ti.func
def calc_c_w(t: data_type) -> data_type:
    """"Теплоемкость воды, [Дж/(кг*C)]"""
    return 4217 - 2.15 * t + 0.002 * t ** 2


@ti.func
def calc_c_o(t: data_type) -> data_type:
    """"Теплоемкость нефти, [Дж/(кг*C)]"""
    return 1800 + 4 * t + 0.01 * t ** 2


@ti.func
def calc_c_f(t: data_type) -> data_type:
    """"Теплоемкость пласта, [Дж/(кг*C)]"""
    return 800 + 0.75 * t


@ti.func
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
    t_arr = ti.field(dtype=data_type, shape=n)
    mu_o = ti.field(dtype=data_type, shape=n)
    mu_w = ti.field(dtype=data_type, shape=n)

    @ti.kernel
    def calc_data():
        for i in range(0, n):
            t = t_0 + i * (t_n - t_0) / (n - 1)
            t_arr[i] = t
            mu_o[i] = calc_mu_o(t)
            mu_w[i] = calc_mu_w(t)

    calc_data()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_arr.to_numpy(), y=mu_w.to_numpy(), mode='lines', name='mu_w',
                             line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=t_arr.to_numpy(), y=mu_o.to_numpy(), mode='lines', name='mu_o',
                             line=dict(color='red', width=3)))

    fig.update_layout(plot_bgcolor='white', width=750, height=350, margin=dict(t=0, b=0),
                      xaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1),
                      yaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1),
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
