from pickle import load

import numpy as np
import plotly.graph_objects as go

from paraphin.constants import Nx, Ny, X_max, X_min, Y_min, Y_max, hx, hy, results_path

x_mesh = np.linspace(X_min + hx / 2, X_max - hx / 2, Nx)
y_mesh = np.linspace(Y_min + hy / 2, Y_max - hy / 2, Ny)


def create_diplom_graphs():
    with open(results_path / 'processed_data.pkl', 'rb') as file:
        _, data = load(file)

    with open(results_path / 'wp_processed_data.pkl', 'rb') as file:
        _, data_wp = load(file)

    field_fig = fields_vis(data, data_wp)
    # plot_fig = plots_vis(data, data_wp)

    field_fig.show()
    # plot_fig .show()


def fields_vis(data, data_wp):
    fig = go.Figure()

    field_name = 'Saturation'
    idx_end = len(data_wp['Time']) - 1
    idx_sat = 94
    field_wp = data_wp[field_name][idx_sat]
    field = data[field_name][idx_sat]

    start_val = 0
    end_val = 1
    step = 0.05

    width_line = 2

    fig.add_trace(go.Contour(
        x=x_mesh,
        y=y_mesh,
        z=field,
        name='Wp=0%',
        contours=dict(
            coloring='lines',
            showlabels=True,
            start=start_val,
            end=end_val,
            size=step,
        ),
        line=dict(width=width_line, dash='dash'),
        colorscale=[[0, 'black'], [1, 'black']],
        showscale=False,
        showlegend=True
    ))

    fig.add_trace(go.Contour(
        x=x_mesh,
        y=y_mesh,
        z=field_wp,
        name='Wp=5%',
        contours=dict(
            coloring='lines',
            showlabels=True,
            start=start_val,
            end=end_val,
            size=step,
        ),
        line=dict(width=width_line),
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False,
        showlegend=True
    ))

    fig.update_layout(
        xaxis_title='X', yaxis_title='Y',
        height=500, width=600,
        showlegend=True,
        legend=dict(
            x=1.05,
            y=0.5,
            bgcolor='rgba(255,255,255,0.7)'
        ),
        margin=dict(t=0, b=0),
        xaxis=dict(gridcolor='black', gridwidth=1),
        yaxis=dict(gridcolor='black', gridwidth=1),
    )
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="black", width=1)
    )

    fig.write_image(results_path.parent / f"{field_name}.svg", width=700, height=600)

    return fig


def plots_vis(data, data_wp):
    fig = go.Figure()

    time = data


    start_val = 0
    end_val = 1
    step = 0.05

    width_line = 2

    fig.add_trace(go.Scatter(
        x=x_data,
        y=y1_data,
        mode='lines',
        name='Линия 1',
        line=dict(color='black', width=width_line, dash='dash'),
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=x_data,
        y=y2_data,
        mode='lines',
        name='Линия 2',
        line=dict(color='red', width=width_line),
        showlegend=True
    ))

    fig.update_layout(
        xaxis_title='X', yaxis_title='Y',
        height=500, width=600,
        showlegend=True,
        legend=dict(
            x=1.05,
            y=0.5,
            bgcolor='rgba(255,255,255,0.7)'
        ),
        margin=dict(t=0, b=0),
        xaxis=dict(gridcolor='black', gridwidth=1),
        yaxis=dict(gridcolor='black', gridwidth=1),
    )
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="black", width=1)
    )

    fig.write_image(results_path.parent / f"{field_name}.svg", width=700, height=600)

    return fig
