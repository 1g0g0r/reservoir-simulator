from pickle import load

import numpy as np
import plotly.graph_objects as go

from paraphin.constants import Nx, Ny, X_max, X_min, Y_min, Y_max, hx, hy, results_path

x_mesh = np.linspace(X_min + hx / 2, X_max - hx / 2, Nx)
y_mesh = np.linspace(Y_min + hy / 2, Y_max - hy / 2, Ny)


def create_diplom_graphs():
    with open(results_path / 'processed_data.pkl', 'rb') as file:
        _, data = load(file)

    fig = go.Figure()

    Z1 = data['Saturation'][-1]
    Z2 = data['Saturation'][-1] * 1.2

    fig.add_trace(go.Contour(
        x=x_mesh,
        y=y_mesh,
        z=Z1,
        name='Первый',
        contours=dict(
            coloring='lines',
            showlabels=True,
            size=0.1,  # Шаг изолиний
        ),
        line=dict(width=2, dash='dash'),
        colorscale=[[0, 'black'], [1, 'black']],
        showscale=False,
        showlegend=True
    ))

    fig.add_trace(go.Contour(
        x=x_mesh,
        y=y_mesh,
        z=Z2,
        name='Второй',
        contours=dict(
            coloring='lines',
            showlabels=True,
            size=0.1,  # Шаг изолиний
        ),
        line=dict(width=2),
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False,
        showlegend=True
    ))

    fig.update_layout(
        xaxis_title='X',
        yaxis_title='Y',
        height=600,
        width=600,
        showlegend=True,
        legend=dict(
            x=1.05,
            y=0.5,
            bgcolor='rgba(255,255,255,0.7)'
        )
    )

    fig.show()
