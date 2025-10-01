"""Создание графиков и двумерных полей в формате .svg из предобработанных данных расчета."""
import numpy as np
import plotly.graph_objects as go

from .read_data_files import read_solution_data
from paraphin.constants import Nx, Ny, X_max, X_min, Y_min, Y_max, outputs_path, bar_to_pa, day_to_sec, S_min, init_T

x_mesh = np.linspace(X_min, X_max, Nx)
y_mesh = np.linspace(Y_min, Y_max, Ny)


def create_graphs_and_maps():
    path = outputs_path / 'pictures'
    path.mkdir(parents=True, exist_ok=True)

    _, data = read_solution_data('Wp=0.0_processed_data.pkl')
    _, data_wp = read_solution_data('Wp=0.05_processed_data.pkl')

    data['Pressure'] /= bar_to_pa
    data_wp['Pressure'] /= bar_to_pa
    data['Time'] /= day_to_sec
    data_wp['Time'] /= day_to_sec

    idx_end_wp = len(data_wp['Time']) - 1
    idx_sat_wp = np.argwhere(data_wp['Wells']['Producer_eta'] != 0)[0][0]
    idx_end = len(data['Time']) - 1
    idx_sat = np.argwhere(data['Wells']['Producer_eta'] != 0)[0][0]
    idx_wp = idx_sat_wp
    idx = idx_sat

    fields_settings = [['Pressure', 50, 150, 2], ['Saturation', S_min*1.02, 1, 0.03], ['Temperature', 25*1.001, init_T*0.99, 10]]
    for _setings in fields_settings:
        _field_vis(idx, idx_wp, data, data_wp, path, *_setings)

    plots_settings = [['Producer_oil', 'Producer_water', 'qв, м³/сут'],
                      # ['Producer_eta','Injector_water', '$$q,\\: \\frac{м^3}{сут}$$'],
                      ['Producer_Q_oil','Producer_Q_water', 'Qв, м³']
                      ]
    for _settings in plots_settings:
        _plot_vis(data_wp['Time'], data['Wells'], data_wp['Wells'], path, *_settings)

    # maps = [['m mult', 0, 1, 0.03], ['k mult', 0, 1, 0.03], ['Wps dep', 0, 0.05, 0.03],
    #         ['Wps', 0, 0.05, 0.03], ['Wp', 0, 0.05, 0.03]]
    # for _maps_setings in maps:
    #     _create_map(idx_wp, data_wp, path, *_maps_setings)

    print('Done!')


def _plot_vis(time, data, data_wp, path, name_plot1, name_plot2, y_axis_title):
    """Процедура строит векторные графики показателей работы скважин."""
    fig = go.Figure()

    if 'eta' in name_plot1 or 'Q' in name_plot1:
        plot_data1 = np.abs(data[name_plot1])
        plot_data_wp1 = np.abs(data_wp[name_plot1])
    else:
        plot_data1 = np.abs(data[name_plot1]) * day_to_sec
        plot_data_wp1 = np.abs(data_wp[name_plot1]) * day_to_sec

    if 'Q' not in name_plot1:
        plot_data2 = np.abs(data[name_plot2]) * day_to_sec
        plot_data_wp2 = np.abs(data_wp[name_plot2]) * day_to_sec
    else:
        plot_data2 = np.abs(data[name_plot2])
        plot_data_wp2 = np.abs(data_wp[name_plot2])

    fig.add_trace(go.Scatter(
        x=time, y=plot_data_wp1, yaxis='y2',
        mode='lines', name='Wp=5%',
        line=dict(color='black', width=3), showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=time, y=plot_data1, yaxis='y2',
        mode='lines', name='Wp=0%',
        line=dict(color='red', width=3, dash='dash'), showlegend=True
    ))
    fig.update_layout(yaxis2 = dict(side="right", overlaying="y", title='qн, м³/сут', domain=[0.0, 0.5], title_font=dict(size=24)))

    fig.add_trace(go.Scatter(
        x=time, y=plot_data_wp2,
        mode='lines', line=dict(color='black', width=3), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=time, y=plot_data2,
        mode='lines', line=dict(color='red', width=3, dash='dash'), showlegend=False
    ))
    fig = _plots_params(fig, 't, сут', y_axis_title)
    fig.write_image(path / f"{name_plot1}_{name_plot2}.svg", width=700, height=600)


def _field_vis(idx, idx_wp, data, data_wp, path, field_name, start, end, step):
    """Процедура строит векторные графики изолиний полей данных."""
    fig = go.Figure()

    field_wp = data_wp[field_name][idx_wp]
    field = data[field_name][idx]

    fig.add_trace(go.Contour(
        x=x_mesh, y=y_mesh, z=field_wp, name='Wp=5%',
        contours=dict(
            coloring='lines', showlabels=True,
            start=start, end=end, size=step,
        ),
        line=dict(width=3), colorscale=[[0, 'black'], [1, 'black']],
        showscale=False, showlegend=True
    ))
    fig.add_trace(go.Contour(
        x=x_mesh, y=y_mesh, z=field, name='Wp=0%',
        contours=dict(
            coloring='lines', showlabels=True,
            start=start, end=end, size=step,
        ),
        line=dict(width=3, dash='dash'), colorscale=[[0, 'red'], [1, 'red']],
        showscale=False, showlegend=True
    ))

    fig = _plots_params(fig, 'X', 'Y')
    fig.write_image(path / f"{field_name}_{round(data_wp['Time'][idx], 2)}.svg", width=700, height=600)


def _create_map(idx_wp, data_wp, path, field_name, start, end, step):
    fig = go.Figure()

    field = data_wp[field_name][idx_wp]

    fig.add_trace(go.Contour(
        x=x_mesh, y=y_mesh, z=field,
        contours=dict(
            coloring='fill', showlabels=True,
            # start=start, end=end, size=step,
        ),
        colorscale='Jet',
        showscale=True, showlegend=False
    ))
    fig.update_layout(
        plot_bgcolor='white', xaxis_title='X', yaxis_title='Y',
        height=500, width=600, showlegend=True,
        margin=dict(t=0, b=0),
        xaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1),
        yaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1),
    )

    fig.write_image(path / f"{field_name}_{round(data_wp['Time'][idx_wp], 2)}.svg", width=700, height=600)


def _plots_params(fig, x_axis_title, y_axis_title):
    """Настройки внешнего вида графиков."""
    fig.update_layout(
        plot_bgcolor='white', xaxis_title=x_axis_title, yaxis_title=y_axis_title,
        height=500, width=600, showlegend=False,
        # legend=dict(x=1.05, y=0.5, bgcolor='rgba(255,255,255,0.7)'),
        margin=dict(t=0, b=0),
        xaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1, title_font=dict(size=24)),
        yaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1, title_font=dict(size=24)),
    )

    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="black", width=1)
    )
    fig.add_hline( y=0, line=dict(color='black', width=1))

    return fig
