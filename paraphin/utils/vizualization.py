from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from paraphin.constants import Nx, Ny, X_min, X_max, hx, hy, Y_max, Y_min, results_path, js_path, bar_to_pa, day_to_sec
from paraphin.utils import read_pkl_files


def visualize_solution(input_data: dict[str, np.ndarray | dict[str, np.ndarray]] | None = None):
    if input_data is None:
        input_data = read_pkl_files()

    x = np.linspace(X_min+hx/2, X_max-hx/2, Nx)
    y = np.linspace(Y_min+hy/2, Y_max-hy/2, Ny)

    time = input_data['Time'] / day_to_sec
    n_times = len(time)
    del input_data['Time']

    # Создаем графики
    data_fields = []
    for name, field in input_data.items():
        if name == 'Pressure':
            trace = go.Heatmap(x=x, y=y, z=field / bar_to_pa,  colorscale='Jet', name=name,
                               zmin=np.min(field) / bar_to_pa, zmax=np.max(field) / bar_to_pa,
                               hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z} Bar<extra></extra>")
        elif name == 'Wells':
            inj = go.Scatter(x=time, y=field['inj'] * day_to_sec, mode='lines', name='Injector',
                             hovertemplate="x: %{x} день<br>y: %{y} м^3/день<br>")  # , xaxis = "Время, день", yaxis = "Дебит, м^3/день"

            trace = go.Scatter(x=time, y=field['prod'] * day_to_sec, mode='lines', name='Producer',
                               hovertemplate="x: %{x} день<br>y: %{y} м^3/день<br>")
            data_fields.append(inj)
        else:
            trace = go.Heatmap(x=x, y=y, z=field,  zmin=np.min(field), zmax=np.max(field), colorscale='Jet', name=name,  # colorscale='Cividis'
                            hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>")
        data_fields.append(trace)

    # Создаем фигуру
    fig = go.Figure(data=data_fields)

    # Создаем массив отображаемых данных (все False, а на диагонали True)
    visibility = np.eye(len(data_fields), dtype=bool)
    visibility[-2, -1] = True

    # Добавляем слайдеры для изменения данных
    steps = []
    for i in range(n_times):
        step = dict(
            method="update",
            args=[{"z": [j.z[i] for j in data_fields[:-2]] + [[j.y[i] for j in data_fields[-2:]]]}],
            label=f'{round(time[i], 5)} день'
        )
        steps.append(step)

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
                              label = name,
                              method = "update") for i, name in enumerate(input_data.keys())],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.35,  # Положение по горизонтали (справа от графика)
                xanchor="left",  # Привязка по горизонтали
                y=0.85,  # Положение по вертикали (сверху)
                yanchor="middle"  # Привязка по вертикали
            ),
        ],
        sliders=sliders,
        width = 1000,  # Устанавливаем ширину фигуры
        height = 800  # Устанавливаем высоту фигуры
    )

    # По умолчанию показываем первое поле
    fig.update_traces(visible=False)
    fig.data[0].visible = True
    # fig.update_layout(sliders=[dict(active=0)], visible=True)

    # Отображаем график
    if __name__ == '__main__':
        fig.write_html(results_path.parent / 'Results.html', include_plotlyjs='plotly_script.js')
    else:
        fig.write_html(results_path.parent / 'Results.html', include_plotlyjs=js_path)

    fig.show()


def show_plot(x, name: str):
    if x.ndim == 1:
        data = x.reshape((Nx, Ny))
    else:
        data = x

    x = np.linspace(X_min, X_max, Nx)
    y = np.linspace(Y_min, Y_max, Ny)

    # Создаем тепловую карту
    fig = go.Figure(data=go.Heatmap(
        x=x,
        y=y,
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

    # Отображаем график
    if __name__ == '__main__':
        fig.write_html(results_path.parent / f'{name}.html', include_plotlyjs='plotly_script.js')
    else:
        fig.write_html(results_path.parent / f'{name}.html', include_plotlyjs=js_path)


if __name__ == '__main__':
    Nx, Ny = 128, 128
    ones = np.ones((Nx, Ny))
    n_times = 50
    time = np.linspace(0, 50, n_times)

    x = np.linspace(X_min, X_max, Nx)
    y = np.linspace(Y_min, Y_max, Ny)
    X, Y = np.meshgrid(x, y)

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
