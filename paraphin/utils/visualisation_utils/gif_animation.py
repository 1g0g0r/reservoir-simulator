import glob
import os
import re
from pickle import load

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from joblib import Parallel, delayed

from paraphin import r
from paraphin.constants import Nx, Ny, X_max, X_min, Y_min, Y_max, outputs_path, bar_to_pa, day_to_sec, S_min, init_T, \
    Twater, init_Wp

x_mesh = np.linspace(X_min, X_max, Nx)
y_mesh = np.linspace(Y_min, Y_max, Ny)

names_converter = {
    'fi': '$$\\varphi$$',
    'fi_o': '$$\\varphi_0$$'
}

isolines_settings = {
    'Saturation': [S_min*1.02, 1, 0.03],
    'Temperature': [Twater*1.001, init_T*0.99, 10],
}


def create_gif():
    with open(outputs_path / 'data' / 'wp0_processed_data.pkl', 'rb') as file:  # 'wp0_processed_data.pkl'
        _, data = load(file)

    with open(outputs_path / 'data' / 'wp5_0607_processed_data.pkl', 'rb') as file:  # 'wp5_processed_data.pkl'
        _, data_wp = load(file)

    data['Pressure'] /= bar_to_pa
    data_wp['Pressure'] /= bar_to_pa
    data['Time'] /= day_to_sec
    data_wp['Time'] /= day_to_sec

    fields_settings = [
        # ['fi', 'r, м', names_converter['fi']],
        # ['Saturation', 'X, м', 'Y, м'],
        # ['Temperature', 'X, м', 'Y, м'],
        # ['m mult', 'X, м', 'Y, м'],
        # ['k mult', 'X, м', 'Y, м'],
        ['Wps dep', 'X, м', 'Y, м']
    ]

    for _settings in fields_settings:
        crating_pictures(10, data, data_wp, *_settings)
        create_gif_from_png(_settings[0], duration=100, loop=0)

    print('Done!')


def crating_pictures(skip_steps, data, data_wp, name, x_axis_name, y_axis_name):
    """Создание картинок нестационарных полей задачи."""
    (outputs_path / name).mkdir(parents=True, exist_ok=True)
    for file_path in (outputs_path / name).glob(f'*.png'):  # Перебор всех файлов .pkl
        file_path.unlink()

    times = data['Time']
    n_times = len(times)

    # Настраиваем ползунок
    steps = [{}] * n_times

    for i in range(n_times):
        steps[i] = dict(
            method='update',
            args=[{'y': []}],
            label=f'{times[i]} день'
        )
    fig = go.Figure()

    fig.update_layout(
        xaxis_title=x_axis_name, yaxis_title=y_axis_name,
        plot_bgcolor='white', margin=dict(t=0, b=0),
        xaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1, title_font=dict(size=18)),
        yaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1, title_font=dict(size=18)),
        yaxis2=dict(side="right", overlaying="y"),
        legend=dict(
            x=1.01, y=0.8,
            font=dict(size=18)
        )
    )
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="black", width=1)
    )
    fig.add_hline(y=0, line=dict(color='black', width=1))

    Parallel(n_jobs=-1, backend='loky')(
        delayed(_process_iter_picture)(idx, fig, name, data, data_wp, steps, times)
        for idx in range(0, len(times), skip_steps)
    )


def create_gif_from_png(name, duration=200, loop=0):
    """Создание анимированный GIF из набора PNG изображений.

    Parameters:
    ---------
    name: str
        Имя параметра визуализации
    duration: int
        Длительность кадра в миллисекундах
    loop: int
        Количество циклов (0 - бесконечно)
    """
    def extract_number(_path):
        """Находим все числа в имени файла"""
        numbers = re.findall(r'\d+', _path)
        return int(numbers[0]) if numbers else 0

    file_pattern = os.path.join(outputs_path / name, "*.png")
    png_files = sorted(glob.glob(file_pattern), key=extract_number)

    if not png_files:
        raise FileNotFoundError(f"PNG файлы не найдены в папке: {name}")

    # Открываем все изображения
    images = [Image.open(f) for f in png_files]  # png_files[:77]+png_files[77::10]

    # Сохраняем как анимированный GIF
    images[0].save(
        outputs_path / f'{name}.gif',
        format="GIF",
        append_images=images[1:],
        save_all=True,
        duration=duration,
        loop=loop,
        optimize=True
    )


def _process_iter_picture(_ii, _fig, _name, _data, _data_wp, _steps, _times):
    _fig.data = []
    if _name == 'fi':
        for _name, _val in _data_wp['plots'].items():
            _fig.add_trace(go.Scatter(x=r, y=_val[_ii], mode='lines', name=names_converter[_name],  # 'markers+lines'
                                     hovertemplate="x: %{x}<br>y: %{y}<br>", line=dict(width=4)))
    elif _name in ['Wps dep', 'k mult', 'm mult']:
        if _name == 'Wps dep':
            z_min = 0.0
            z_max = init_Wp
        else:
            z_max = 1.0
            if _name == 'k mult':
                z_min = 0.65
            else:
                z_min = 0.75
        _fig.add_trace(go.Contour(
            x=x_mesh, y=y_mesh, z=_data_wp[_name][_ii], zmin=z_min, zmax=z_max,
            contours=dict(
                coloring='fill', showlabels=True,
                # start=start, end=end, size=step,
            ),
            colorscale='Jet',
            showscale=True, showlegend=False
        ))
    else:
        isolines = isolines_settings[_name]
        _fig.add_trace(go.Contour(
            x=x_mesh, y=y_mesh, z=_data_wp[_name][_ii],
            contours=dict(
                coloring='lines', showlabels=True,
                start=isolines[0], end=isolines[1], size=isolines[2]
            ),
            line=dict(width=3), colorscale=[[0, 'black'], [1, 'black']],
            showscale=False, showlegend=False
        ))
        if _name in _data.keys():
            _fig.add_trace(go.Contour(
                x=x_mesh, y=y_mesh, z=_data[_name][_ii],
                contours=dict(
                    coloring='lines', showlabels=True,
                    start=isolines[0], end=isolines[1], size=isolines[2]
                ),
                line=dict(width=3, dash='dash'), colorscale=[[0, 'red'], [1, 'red']],
                showscale=False, showlegend=False
            ))

    sliders = [dict(active=_ii, currentvalue={'prefix': 'Время: '}, steps=_steps)]

    # Настраиваем макет
    _fig.update_layout(sliders=sliders, height=600)

    # Дополнительные настройки отображения
    if _name == 'fi':
        _fig.update_traces(marker=dict(size=8, line=dict(width=1)))
        _fig.write_image(outputs_path / _name / f'{_times[_ii]}.png', width=750, height=350)
    else:
        _fig.write_image(outputs_path / _name / f'{_times[_ii]}.png', width=700, height=600)
