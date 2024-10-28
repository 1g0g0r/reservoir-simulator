from matplotlib.pyplot import imshow, show, colorbar
from numpy import linspace, ones, meshgrid, array, diag, cos, sin, random
from plotly.graph_objects import Heatmap, Figure

from paraphin.constants import Nx, Ny, X_min, X_max, Y_max, Y_min


def visualize_solution(input_data):
    x = linspace(X_min, X_max, Nx)
    y = linspace(Y_min, Y_max, Ny)

    time = input_data['time']
    n_times = len(time)
    del input_data['time']

    # Создаем графики
    traces = []
    for name, field in input_data.items():
        trace = Heatmap(x=x, y=y, z=field, colorscale='Jet', name=name,
                           hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>")
        traces.append(trace)

    # Создаем фигуру
    fig = Figure(data=traces)

    # Создаем массив отображаемых данных
    visibility = []
    for i, val in enumerate(input_data):
        temp = [False] * len(input_data)
        temp[i] = True
        visibility.append(temp)

    # Добавляем слайдеры для изменения данных
    steps = []
    for i in range(n_times):
        step = dict(
            method="update",
            args=[{"z": [j.z[i] for j in traces]}],
            label=round(time[i], 5)
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
                              method = "update") for i, val in enumerate(traces)],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.15,  # Положение по горизонтали (справа от графика)
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
    fig.write_html('results.html', include_plotlyjs='paraphin\\utils\\plotly_script.js')


def show_plot(x, type):
    if x.ndim == 1:
        data = x.reshape((Nx, Ny))
    else:
        data = x

    if type == 'mpl':
        # Отображаем массив с помощью imshow
        imshow(data, cmap='viridis')

        # Добавляем цветовую шкалу с дополнительными параметрами
        cbar = colorbar(orientation='horizontal', shrink=0.8)
        cbar.set_label('Значения данных')

        # Отображаем график
        show()

    elif type == 'plotly':
        x = linspace(X_min, X_max, Nx)
        y = linspace(Y_min, Y_max, Ny)

        # Создаем тепловую карту
        fig = Figure(data=Heatmap(
            x=x,
            y=y,
            z=data,
            colorscale='Jet'
        ))

        # Настраиваем отображение графика
        fig.update_layout(
            title='Поле данных',
            xaxis_title='X',
            yaxis_title='Y',
            width=800,
            height=600
        )

        # Отображаем график
        fig.write_html('results.html', include_plotlyjs='paraphin\\utils\\plotly_script.js')


if __name__ == '__main__':
    ones = ones((Nx, Ny))
    n_times = 50

    x = linspace(X_min, X_max, Nx)
    y = linspace(Y_min, Y_max, Ny)
    X, Y = meshgrid(x, y)

    data = {
        'time': linspace(0, 50, n_times),
        'pressure': array([cos(X ** 2 + Y ** 2) + i * 0.01 * random.randn(Nx, Ny) for i in range(n_times)]),
        'temperature': array([sin(X ** 2 + Y ** 2) + i * 0.01 * random.randn(Nx, Ny) for i in range(n_times)]),
        'saturation': array([ones + diag(ones.diagonal()) * i * 10 for i in range(n_times)]),
    }

    visualize_solution(data)
