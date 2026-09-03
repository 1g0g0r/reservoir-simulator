"""Создание интерактивной визуализации решения в формате .html."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import plotly.graph_objects as go

from paraphin import r
from paraphin.constants import (Nx, Ny, X_min, X_max, hx, hy, Y_max, Y_min, results_path, layers_file, js_path, bar_to_pa,
                                day_to_sec, CONTOUR_PLOT, S_max, init_Wp)
from .read_data_files import read_solution_data, convert_pkl_files

x_mesh = np.linspace(X_min + hx / 2, X_max - hx / 2, Nx)
y_mesh = np.linspace(Y_min + hy / 2, Y_max - hy / 2, Ny)

names_converter = {
    'fi': '$$\\varphi$$',
    'fi_o': '$$\\varphi_0$$'
}


def visualize_solution():
    """Визуализация данных расчета."""
    try:
        # Файл слоев остается на диске, только если расчет не дошел до склейки: значит данные
        # свежее обработанного файла и их надо собрать заново
        if layers_file.is_file():
            convert_pkl_files()
        _n_times, input_data = read_solution_data(f'Wp={init_Wp}_processed_data.pkl')  #'(wp5 40)processed_data.pkl'

    # Если файл конвертированных данных отсутствует, то сами создаем его
    except ValueError:
        convert_pkl_files()
        _n_times, input_data = read_solution_data(f'Wp={init_Wp}_processed_data.pkl')

    print('Временных слоев:', _n_times)

    fig_fields = _visualize_fields(input_data)

    if 'plots' in input_data.keys():
        fig_plots = _visualize_plots_fi(input_data)
        # fig_plots.show()

    fig_fields.show()


def _visualize_fields(data):
    """Создание анимации полей данных и параметров скважин."""
    input_data = deepcopy(data)
    wells_plots, other_plots, fields_maps = 0, 0, 0
    wells_accumulated_plots = 0

    time = input_data['Time'] / day_to_sec
    input_data['Pressure'] /= bar_to_pa

    n_times = len(time)
    del input_data['Time']
    if len(input_data['Wells']) == 0:
        del input_data['Wells']
        del input_data['Wells_accumulated']

    if 'plots' in input_data.keys():
        input_data['Wps'] *= (S_max - input_data['Saturation'])
        del input_data['plots']

    # 'Pressure', 'Saturation', 'Temperature', 'm mult', 'Wps', 'Wps dep','mu_o', 'mu_w', 'Wells', 'Other params'
    skip_fields = ['Wo', 'Wp', 'Wps dep', 'Wps', 'm', 'mu_w', 'qp']

    _f_names = [name for name in input_data.keys() if name not in skip_fields]
    data_fields = []

    # Создаем графики
    for name, field in input_data.items():
        if name in skip_fields:
            continue

        trace = []
        if name == 'Wells':
            for _name, _val in field.items():
                if np.all(np.isclose(_val, 0.0)) or np.all(np.isclose(_val, 1.0)): continue
                if 'eta' in _name:
                    trace += [go.Scatter(x=time, y=_val, mode='lines', name=_name, yaxis='y2',
                                         hovertemplate="x: %{x} день<br>y: %{y}<br>")]
                else:
                    trace += [go.Scatter(x=time, y=abs(_val) * day_to_sec, mode='lines', name=_name,
                                             hovertemplate="x: %{x} день<br>y: %{y} м^3/день<br>")]
                wells_plots += 1
        elif name == 'Wells_accumulated':
            for _name, _val in field.items():
                if np.all(np.isclose(_val, 0.0)): continue
                trace += [go.Scatter(x=time, y=abs(_val), mode='lines', name=_name,
                                     hovertemplate="x: %{x} день<br>y: %{y} м^3<br>")]
                wells_accumulated_plots += 1
        elif name == 'Other params':
            for _name, _val in field.items():
                trace += [go.Scatter(x=time, y=abs(_val), mode='lines', name=_name,
                                     hovertemplate="x: %{x}<br>y: %{y}<br>")]  # xaxis='x2',
                other_plots += 1
        else:
            z_max = np.max(field)
            z_min = np.min(field)
            if CONTOUR_PLOT:
                trace = [go.Contour(x=x_mesh, y=y_mesh, z=field[0], colorscale='Jet', name=name, zmin=z_min, zmax=z_max,
                                    hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>",
                                    contours=dict(
                                        coloring='fill',  # 'lines',
                                        showlabels=True,
                                        labelfont=dict(size=12, color='black')
                                    ))]
            else:
                trace = [go.Heatmap(x=x_mesh, y=y_mesh, z=field[0], zmin=z_min, zmax=z_max,
                                    colorscale='Jet', name=name,  # colorscale='bluered'
                                    hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>")]
            fields_maps += 1
        data_fields += trace

    # Создание кастомной карты
    _f_names += ['Sat and Temp']
    data_fields += [go.Contour(x=x_mesh, y=y_mesh, z=input_data['Saturation'][0], colorscale='Jet', name='Saturation',
                               contours=dict(coloring='fill', showlabels=True))]

    data_fields += [go.Contour(x=x_mesh, y=y_mesh, z=input_data['Temperature'][0], name='Temperature',
                             contours=dict(coloring='lines', showlabels=True,
                                           start=25 * 1.001, end=70 * 0.99, size=10),
                             line=dict(width=3), colorscale=[[0, 'black'], [1, 'black']], showscale=False, showlegend=True)]

    # Создаем фигуру
    fig = go.Figure(data=data_fields)

    fig.update_layout(
        yaxis2=dict(side="right", overlaying="y"),
        legend=dict(x=1.05, y=1.0)
    )

    # Создаем массив отображаемых данных
    visibility = _get_visibility(data_fields, _f_names, fields_maps, wells_plots, wells_accumulated_plots, other_plots)

    # Добавляем слайдеры для изменения данных
    steps = [{}] * n_times
    for i in range(n_times):
        steps[i] = dict(
            method="update",
            args=[{"z": [input_data[j.name][i] for j in data_fields if j.plotly_name in ['contour', 'heatmap']]}],
            label=f'{round(time[i], 1)} день'
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
                buttons=[dict(args=[{"visible": visibility[i]}], label=name,
                              method="update") for i, name in enumerate(_f_names)],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.35,  # Положение по горизонтали (справа от графика)
                xanchor="left",  # Привязка по горизонтали
                y=0.85,  # Положение по вертикали (сверху)
                yanchor="middle"  # Привязка по вертикали
            ),
        ],
        sliders=sliders, plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1, title_font=dict(size=18)),
        yaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1, title_font=dict(size=18)),
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


def _visualize_plots_fi(plots_data):
    """Создание графиков зависящих от радиусов пор"""
    time = plots_data['Time'] / day_to_sec
    data = plots_data['plots']

    n_times = len(time)
    data_fields = []

    # Создаем базовый график с первой строкой
    for _name, _val in data.items():
        if _name in ['Ur', 'Ub']:  # 'markers+lines'
            data_fields += [go.Scatter(x=r, y=_val[0], mode='lines', name=_name, yaxis='y2',
                                       hovertemplate="x: %{x}<br>y: %{y}<br>", line=dict(width=3))]
        else:
            data_fields += [go.Scatter(x=r, y=_val[0], mode='lines', name=names_converter[_name],
                                       hovertemplate="x: %{x}<br>y: %{y}<br>", line=dict(width=4))]

    fig = go.Figure(data=data_fields)

    fig.update_layout(
        xaxis_title='r, м', yaxis_title=names_converter['fi'],
        plot_bgcolor='white', margin=dict(t=0, b=0),
        xaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1, title_font=dict(size=18)),
        yaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1, title_font=dict(size=18)),
        yaxis2=dict(side="right", overlaying="y"),
        legend=dict(x=1.01, y=0.8, font=dict(size=18))
    )
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="black", width=1)
    )
    fig.add_hline( y=0, line=dict(color='black', width=1))

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


def show_plot(data, name: str = 'map', show: bool = False):
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
    if show:
        fig.show()
    else:
        if __name__ == '__main__':
            fig.write_html(results_path.parent / f'{name}.html', include_plotlyjs='plotly_script.js')
        else:
            fig.write_html(results_path.parent / f'{name}.html', include_plotlyjs=js_path)


def _get_visibility(data_fields, names, fields_maps, wells_plots, wells_accumulated_plots, other_plots):
    """Создание массива отображаемых данных для каждой кнопки."""
    visibility = np.eye(len(data_fields), dtype=bool)
    n_button = 0
    end_idx = fields_maps

    for name, n_graphs in [('Wells', wells_plots), ('Wells_accumulated', wells_accumulated_plots), ('Other params', other_plots)]:
        # Если данные есть, то создаем для них маску для кнопки
        if name in names:
            visibility[fields_maps + n_button, end_idx: end_idx + n_graphs] = True
            if n_button > 0:
                visibility[fields_maps + n_button, fields_maps + n_button] = False
            end_idx += n_graphs
            n_button += 1

    # Все дополнительные графики отображаем одновременно
    visibility[fields_maps + n_button, end_idx: ] = True
    visibility[fields_maps + n_button, fields_maps + n_button] = False

    return visibility
