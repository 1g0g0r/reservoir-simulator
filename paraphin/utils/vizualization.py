from pickle import load

import numpy as np
import plotly.graph_objects as go

from paraphin.constants import Nx, Ny, X_min, X_max, Y_max, Y_min, results_path, js_path, bar_to_pa, day_to_sec


def visualize_solution(input_data: dict[str, np.ndarray]|None = None):
    if input_data is None:
        with open(results_path, 'rb') as f:
            """Pickle файл имеет следующую структуру:
            input_data = {
                'Time': массив точек времени,
                'Pressure': массив двумерных полей данных давления,
                'Temperature': массив двумерных полей данных температуры,
                'Saturation': массив двумерных полей данных насыщенности
            }
            """
            input_data = load(f)

    x = np.linspace(X_min, X_max, Nx)
    y = np.linspace(Y_min, Y_max, Ny)

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
        else:
            trace = go.Heatmap(x=x, y=y, z=field,  zmin=np.min(field), zmax=np.max(field), colorscale='Jet', name=name,  # colorscale='Cividis'
                            hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>")
        data_fields.append(trace)

    # Создаем фигуру
    fig = go.Figure(data=data_fields)

    # Создаем массив отображаемых данных (все False, а на диагонали True)
    visibility = np.eye(len(data_fields), dtype=bool)

    # Добавляем слайдеры для изменения данных
    steps = []
    for i in range(n_times):
        step = dict(
            method="update",
            args=[{"z": [j.z[i] for j in data_fields]}],
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
                              label = val.name,
                              method = "update") for i, val in enumerate(data_fields)],
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

    # Отображаем график
    if __name__ == '__main__':
        fig.write_html(results_path.parent / 'results.html', include_plotlyjs='plotly_script.js')
    else:
        fig.write_html(results_path.parent / 'results.html', include_plotlyjs=js_path)

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
        height=600
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

    x = np.linspace(X_min, X_max, Nx)
    y = np.linspace(Y_min, Y_max, Ny)
    X, Y = np.meshgrid(x, y)

    data = {
        'Time': np.linspace(0, 50, n_times),
        'Pressure': np.array([np.cos(X ** 2 + Y ** 2) + i * 0.01 * np.random.randn(Nx, Ny) for i in range(n_times)]),
        'Temperature': np.array([np.sin(X ** 2 + Y ** 2) + i * 0.01 * np.random.randn(Nx, Ny) for i in range(n_times)]),
        'Saturation': np.array([ones + np.diag(ones.diagonal()) * i * 10 for i in range(n_times)]),
    }

    visualize_solution(data)
