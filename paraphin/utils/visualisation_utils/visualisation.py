"""Модуль визуализации решения."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from paraphin import r
from paraphin.constants import (Nx, Ny, X_min, X_max, hx, hy, Y_max, Y_min, results_path, js_path, bar_to_pa,
                                day_to_sec, CONTOUR_PLOT)
from .read_data_files import read_pkl_files

x_mesh = np.linspace(X_min + hx / 2, X_max - hx / 2, Nx)
y_mesh = np.linspace(Y_min + hy / 2, Y_max - hy / 2, Ny)


def visualize_solution():
    """Визуализация данных расчета."""
    _n_times, input_data = read_pkl_files()
    print('Временных слоев:', _n_times)

    # fig_plots = _visualize_plots(input_data)
    fig_fields = _visualize_fields(input_data)

    # fig_plots.show()
    fig_fields.show()


def _visualize_fields(input_data):
    """Создание анимации полей данных и параметров скважин."""
    wells_plots = 0
    aver_param_plots = 0

    time = input_data['Time'] / day_to_sec
    n_times = len(time)
    del input_data['Time']
    del input_data['plots']

    # Создаем графики
    n_fields = len(input_data)
    data_fields = []
    for name, field in input_data.items():
        trace = []
        if name == 'Pressure':
            if CONTOUR_PLOT:
                trace = [go.Contour(x=x_mesh, y=y_mesh, z=field / bar_to_pa, colorscale='Jet', name=name,
                                    zmin=np.min(field) / bar_to_pa, zmax=np.max(field) / bar_to_pa,
                                    hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z} Bar<extra></extra>",
                                    contours=dict(
                                        coloring='fill',
                                        showlabels=True,
                                        labelfont=dict(size=12, color='black')
                                    ))]
            else:
                trace = [go.Heatmap(x=x_mesh, y=y_mesh, z=field / bar_to_pa, colorscale='Jet', name=name,
                                    zmin=np.min(field) / bar_to_pa, zmax=np.max(field) / bar_to_pa,
                                    hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z} Bar<extra></extra>")]
        elif name == 'Wells':
            for _name, _val in field.items():
                if np.all(np.isclose(_val, 0.0)) or np.all(np.isclose(_val, 1.0)):
                    continue
                if 'eta' in _name:
                    trace += [go.Scatter(x=time, y=abs(_val), mode='lines', name=_name, yaxis='y2',
                                         hovertemplate="x: %{x} день<br>y: %{y}<br>")]
                else:
                    if 'Q' in _name:
                        trace += [go.Scatter(x=time, y=abs(_val), mode='lines', name=_name,
                                            hovertemplate="x: %{x} день<br>y: %{y} м^3/день<br>")]
                    else:
                        trace += [go.Scatter(x=time, y=abs(_val) * day_to_sec, mode='lines', name=_name,
                                             hovertemplate="x: %{x} день<br>y: %{y} м^3/день<br>")]
                wells_plots += 1
        elif name == 'Other params':
            for _name, _val in field.items():
                # temperature = input_data['Temperature'][:,0,0] °C
                trace += [go.Scatter(x=time, y=abs(_val), mode='lines', name=_name,
                                     hovertemplate="x: %{x}<br>y: %{y}<br>")]  # xaxis='x2',
                aver_param_plots += 1
        else:
            z_max = np.max(field)
            z_min = np.min(field)
            if CONTOUR_PLOT:
                trace = [go.Contour(x=x_mesh, y=y_mesh, z=field, colorscale='Jet', name=name, zmin=z_min, zmax=z_max,
                                    hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>",
                                    contours=dict(
                                        coloring='fill',
                                        showlabels=True,
                                        labelfont=dict(size=12, color='black')
                                    ))]
            else:
                trace = [go.Heatmap(x=x_mesh, y=y_mesh, z=field, zmin=z_min, zmax=z_max,
                                    colorscale='Jet', name=name,  # colorscale='bluered'
                                    hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>")]
        data_fields += trace

    # Создаем фигуру
    fig = go.Figure(data=data_fields)

    fig.update_layout(
        # xaxis2=dict(autorange="reversed", overlaying='x'),
        yaxis2=dict(side="right", overlaying="y"),
        legend=dict(x=1.05, y=1.0)
    )

    # Создаем массив отображаемых данных (все False, а на диагонали True)
    visibility = np.eye(len(data_fields), dtype=bool)
    if aver_param_plots == 0:
        visibility[n_fields - 1, n_fields - 1:] = True
    else:
        visibility[n_fields - 2, n_fields - 2:-aver_param_plots] = True
        visibility[n_fields - 1, n_fields - 2 + wells_plots:] = True
        visibility[n_fields - 1, n_fields - 1] = False

    # Добавляем слайдеры для изменения данных
    steps = [{}] * n_times
    for i in range(n_times):
        steps[i] = dict(
            method="update",
            args=[{
                "z": [j.z[i] for j in data_fields if j.plotly_name in ['contour', 'heatmap']]
            }],
            label=f'{round(time[i], 5)} день'
        )

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Время: "},
        steps=steps
    )]

    # Добавляем кнопки для выбора разных наборов данных
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="down",
                buttons=[dict(args=[{"visible": visibility[i]}],
                              label=name,
                              method="update") for i, name in enumerate(input_data.keys())],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.35,  # Положение по горизонтали (справа от графика)
                xanchor="left",  # Привязка по горизонтали
                y=0.85,  # Положение по вертикали (сверху)
                yanchor="middle"  # Привязка по вертикали
            ),
        ],
        sliders=sliders,
        width=1000,  # Устанавливаем ширину фигуры
        height=800  # Устанавливаем высоту фигуры
    )

    # По умолчанию показываем первое поле
    fig.update_traces(visible=False)
    fig.data[0].visible = True

    if __name__ == '__main__':
        fig.write_html(results_path.parent / 'Results.html', include_plotlyjs='plotly_script.js')
    else:
        fig.write_html(results_path.parent / 'Results.html', include_plotlyjs=js_path)

    return fig


def _visualize_plots(plots_data):
    """Создание графиков зависящих от радиусов пор"""
    time = plots_data['Time'] / day_to_sec
    data = plots_data['plots']

    n_times = len(time)
    data_fields = []

    # Создаем базовый график с первой строкой
    for _name, _val in data.items():
        if _name in ['Ur', 'Ub']:
            data_fields += [go.Scatter(x=r, y=_val[0], mode='markers+lines', name=_name, yaxis='y2', hovertemplate="x: %{x}<br>y: %{y}<br>")]
        else:
            data_fields += [go.Scatter(x=r, y=_val[0], mode='markers+lines', name=_name, hovertemplate="x: %{x}<br>y: %{y}<br>")]

    fig = go.Figure(data=data_fields)

    fig.update_layout(
        # xaxis2=dict(autorange="reversed", overlaying='x'),
        yaxis2=dict(side="right", overlaying="y"),
        legend=dict(x=1.05, y=1.0)
    )

    # Настраиваем ползунок
    steps = [{}] * n_times
    for i in range(n_times):
        steps[i] = dict(
            method='update',
            args=[
                {'y': [j[i] for j in data.values()]},
            ],
            label=f'{time[i]} день'
        )

    sliders = [dict(
        active=0,
        currentvalue={'prefix': 'Время: '},
        steps=steps
    )]

    # Настраиваем макет
    fig.update_layout(
        sliders=sliders,
        height=600
    )

    # Дополнительные настройки отображения
    fig.update_traces(
        marker=dict(
            size=8,
            line=dict(width=1)
        )
    )

    if __name__ == '__main__':
        fig.write_html(results_path.parent / f'fi_func.html', include_plotlyjs='plotly_script.js')
    else:
        fig.write_html(results_path.parent / f'fi_func.html', include_plotlyjs=js_path)

    return fig


def show_plot(data, name: str = 'map'):
    """Визуализация поля данных."""
    fig = go.Figure(data=go.Heatmap(
        x=x_mesh,
        y=y_mesh,
        z=data,
        colorscale='Jet'
    ))

    # Настраиваем отображение графика
    fig.update_layout(
        title=f'Поле данных {name}',
        xaxis_title='X',
        yaxis_title='Y',
        width=800,
        height=800
    )

    if __name__ == '__main__':
        fig.write_html(results_path.parent / f'{name}.html', include_plotlyjs='plotly_script.js')
    else:
        fig.write_html(results_path.parent / f'{name}.html', include_plotlyjs=js_path)


if __name__ == '__main__':
    Nx, Ny = 128, 128
    ones = np.ones((Nx, Ny))
    n_times = 50
    time = np.linspace(0, 50, n_times)

    X, Y = np.meshgrid(x_mesh, y_mesh)

    data = {
        'Time': time,
        'Pressure': np.array([np.cos(X ** 2 + Y ** 2) + i * 0.01 * np.random.randn(Nx, Ny) for i in range(n_times)]),
        'Temperature': np.array([np.sin(X ** 2 + Y ** 2) + i * 0.01 * np.random.randn(Nx, Ny) for i in range(n_times)]),
        'Saturation': np.array([ones + np.diag(ones.diagonal()) * i * 10 for i in range(n_times)]),
        'Wells': {
            'inj': np.array([i * i for i in time]),
            'prod': np.array([i * 5 for i in time])
        }
    }

    visualize_solution(data)
