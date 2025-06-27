from pickle import load

import numpy as np
import plotly.graph_objects as go

from paraphin.constants import Nx, Ny, X_max, X_min, Y_min, Y_max, hx, hy, results_path, bar_to_pa, day_to_sec

x_mesh = np.linspace(X_min + hx / 2, X_max - hx / 2, Nx)
y_mesh = np.linspace(Y_min + hy / 2, Y_max - hy / 2, Ny)


def create_graphs_and_maps():
    path = results_path.parent / 'pictures'

    with open(results_path / 'processed_data.pkl', 'rb') as file:
        _, data = load(file)

    with open(results_path / 'wp_processed_data.pkl', 'rb') as file:
        _, data_wp = load(file)

    data['Pressure'] /= bar_to_pa
    data_wp['Pressure'] /= bar_to_pa
    data_wp['Time'] /=day_to_sec

    idx_end = len(data_wp['Time']) - 1
    idx_sat = 94
    idx = idx_sat

    fields_settings = [['Pressure', 50, 150, 2], ['Saturation', 0, 1, 0.03], ['Temperature', 25, 70, 5]]
    for _setings in fields_settings:
        _field_vis(idx, data, data_wp, path, *_setings)

    plots_names = ['Producer_oil', 'Producer_water', 'Producer_eta', 'Injector_total']
    for name in plots_names:
        _plot_vis(data_wp['Time'], data['Wells'], data_wp['Wells'], path, name)

    maps = ['m mult', 'k mult']  # , 'Wps_dep'
    for name in  maps:
        _create_map(idx, data_wp, path, name)

    print('Done!')


def _plot_vis(time, data, data_wp, path, name_plot):
    """Процедура строит векторные графики показателей работы скважин."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time, y=np.abs(data[name_plot]),
        mode='lines', name='Wp=0%',
        line=dict(color='black', width=3, dash='dash'), showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=time, y=np.abs(data_wp[name_plot]),
        mode='lines', name='Wp=5%',
        line=dict(color='red', width=3), showlegend=True
    ))
    fig.update_layout(
        xaxis_title='t, сут', yaxis_title='q, m^3 / сут', height=500, width=600, showlegend=True,
        legend=dict(
            x=1.05, y=0.5,
            bgcolor='rgba(255,255,255,0.7)'
        ),
        margin=dict(t=0, b=0), xaxis=dict(gridcolor='black', gridwidth=1), yaxis=dict(gridcolor='black', gridwidth=1),
    )
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="black", width=1)
    )

    fig.write_image(path / f"{name_plot}.svg", width=700, height=600)


def _field_vis(idx, data, data_wp, path, field_name, start, end, step):
    """Процедура строит векторные графики изолиний полей данных."""
    fig = go.Figure()

    field_wp = data_wp[field_name][idx]
    field = data[field_name][idx]

    fig.add_trace(go.Contour(
        x=x_mesh, y=y_mesh, z=field, name='Wp=0%',
        contours=dict(
            coloring='lines', showlabels=True,
            start=start, end=end, size=step,
        ),
        line=dict(width=3, dash='dash'), colorscale=[[0, 'black'], [1, 'black']],
        showscale=False, showlegend=True
    ))

    fig.add_trace(go.Contour(
        x=x_mesh, y=y_mesh, z=field_wp, name='Wp=5%',
        contours=dict(
            coloring='lines', showlabels=True,
            start=start, end=end, size=step,
        ),
        line=dict(width=3), colorscale=[[0, 'red'], [1, 'red']],
        showscale=False, showlegend=True
    ))
    fig.update_layout(
        xaxis_title='X', yaxis_title='Y', height=500, width=600, showlegend=True,
        legend=dict(
            x=1.05, y=0.5, bgcolor='rgba(255,255,255,0.7)'
        ),
        margin=dict(t=0, b=0), xaxis=dict(gridcolor='black', gridwidth=1), yaxis=dict(gridcolor='black', gridwidth=0.5),
    )
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1, line=dict(color="black", width=1)
    )

    time = round(data_wp['Time'][idx], 2)
    fig.write_image(path / f"{field_name}_{time}.svg", width=700, height=600)


def _create_map(idx, data_wp, path, field_name):
    fig = go.Figure()

    field = data_wp[field_name][idx]

    fig.add_trace(go.Contour(x=x_mesh, y=y_mesh, z=field, colorscale='Jet', name=field_name,
                             hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>",
                             contours=dict(
                                 coloring='fill',  # 'lines',
                                 # showlabels=True,
                                 labelfont=dict(size=12, color='black')
                                 ))
                  )

    fig.update_layout(
        xaxis_title='X', yaxis_title='Y', height=500, width=600, showlegend=True,
        legend=dict(
            x=1.05, y=0.5, bgcolor='rgba(255,255,255,0.7)'
        ),
        margin=dict(t=0, b=0), xaxis=dict(gridcolor='black', gridwidth=1),
        yaxis=dict(gridcolor='black', gridwidth=0.5)
    )
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1, line=dict(color="black", width=1)
    )

    time = round(data_wp['Time'][idx], 2)
    fig.write_image(path / f"{field_name}_{time}.svg", width=700, height=600)
