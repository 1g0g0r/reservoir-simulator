"""Сохранение графика plotly в векторном формате eps"""
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import to_hex, Colormap, LinearSegmentedColormap

from paraphin.constants import pictures_path


def plotly_to_eps(fig_plotly: go.Figure, filename: str, dpi: int = 1000, **kwargs) -> None:
    """Сохраняет график Plotly в формате EPS с промежуточным преобразованием в Matplotlib.

    Parameters:
    -----------
    fig_plotly: go.Figure
        Plotly Figure объект
    filename: str
        Имя файла для сохранения
    dpi: int
        Разрешение для EPS

    Example:
    -------
    plotly_to_eps(
        fig_plotly=fig,
        filename='plotly_graph',
        dpi=1200
    )
    """
    width, height = fig_plotly.layout.width / 100, fig_plotly.layout.height / 100
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)

    if hasattr(fig_plotly.layout, 'yaxis2'):
        ax2 = ax.twinx()

        # поменяли оси местами
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax2.yaxis.set_label_position("left")
        ax2.yaxis.tick_left()

    for trace in fig_plotly.data:
        trace_type = trace.type.lower()

        if trace_type == 'scatter':
            line_style = '-'

            # Стиль линии и маркеров
            if hasattr(trace, 'mode'):
                if 'lines' not in trace.mode.lower():
                    line_style = ''
                if trace.line.dash:
                    line_style = '--'

            # Цвет
            color = None
            if hasattr(trace, 'line') and hasattr(trace.line, 'color') and trace.line.color:
                color = trace.line.color
            elif hasattr(trace, 'marker') and hasattr(trace.marker, 'color') and trace.marker.color:
                color = trace.marker.color

            # Название для легенды
            name = trace.name if hasattr(trace, 'name') and trace.name else None

            if not trace.yaxis:
                yaxis = fig_plotly.layout.yaxis
                axis = ax
            else:
                yaxis = fig_plotly.layout.yaxis2
                axis = ax2

            yaxis_title = yaxis.title.text if yaxis.title.text else ''
            yaxis_font_size = yaxis.title.font.size if yaxis.title.font.size else 14
            axis.set_ylabel(yaxis_title.replace("$$", "$"), fontsize=yaxis_font_size)
            plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)

            axis.plot(trace.x[:len(trace.y)], trace.y,
                    linestyle=line_style, color=color, label=name, linewidth=trace.line.width)
            # TODO добавить эксп формат и подписи возле кривых

        elif trace_type == 'bar':
            color = None
            if hasattr(trace, 'marker') and hasattr(trace.marker, 'color') and trace.marker.color:
                color = trace.marker.color
            name = trace.name if hasattr(trace, 'name') and trace.name else None
            ax.bar(trace.x, trace.y, color=color, label=name, alpha=0.7)

        elif trace_type == 'histogram':
            color = None
            if hasattr(trace, 'marker') and hasattr(trace.marker, 'color') and trace.marker.color:
                color = trace.marker.color
            name = trace.name if hasattr(trace, 'name') and trace.name else None
            ax.hist(trace.x, bins=30, color=color, alpha=0.7, label=name)

        elif trace_type == 'contour':
            if hasattr(trace, 'z') and trace.z is not None:
                has_heatmap_or_contour = True
                colorbar_trace = trace
                z = np.array(trace.z)
                x = np.array(trace.x) if hasattr(trace, 'x') and trace.x is not None else np.arange(z.shape[1])
                y = np.array(trace.y) if hasattr(trace, 'y') and trace.y is not None else np.arange(z.shape[0])

                # Создаем сетку для контуров
                X, Y = np.meshgrid(x, y)

                # Параметры контуров
                contours = trace.contours if hasattr(trace, 'contours') else None
                if contours is None:
                    continue

                levels = np.linspace(contours.start, contours.end, int((contours.end - contours.start) / contours.size))

                # Цветовая карта
                colorscale = trace.colorscale if hasattr(trace, 'colorscale') else 'Viridis'
                cmap = _convert_plotly_colorscale_to_cmap(colorscale)
                _, color = colorscale[0]

                # Рисуем контуры
                contour_type = trace.contours_type if hasattr(trace, 'contours_type') else 'levels'
                linestyle = 'dashed' if trace.line.dash is not None else 'solid'
                width = trace.line.width

                if contour_type == 'constraint':
                    # Заполненные контуры
                    contourf = ax.contourf(X, Y, z, levels, cmap=cmap, alpha=0.8)
                    # Добавляем контурные линии поверх
                    contour_lines = ax.contour(X, Y, z, levels, colors=color, linestyles=linestyle, linewidths=width)
                else:
                    # Только контурные линии
                    contour_lines = ax.contour(X, Y, z, levels, colors=color,
                                               linestyles=linestyle, linewidths=width)
                    contourf = contour_lines

                # Добавляем подписи к контурам
                if contours.showlabels if hasattr(contours, 'showlabels') else False:
                    ax.clabel(contour_lines, inline=True, fontsize=10)

                # Название для легенды
                if hasattr(trace, 'name') and trace.name:
                    contourf.set_label(trace.name)

        elif trace_type == 'heatmap':
            # Обработка тепловой карты
            if hasattr(trace, 'z') and trace.z is not None:
                has_heatmap_or_contour = True
                colorbar_trace = trace

                z = np.array(trace.z)
                x = np.array(trace.x) if hasattr(trace, 'x') and trace.x is not None else np.arange(z.shape[1])
                y = np.array(trace.y) if hasattr(trace, 'y') and trace.y is not None else np.arange(z.shape[0])

                # Цветовая карта
                colorscale = trace.colorscale if hasattr(trace, 'colorscale') else 'Viridis'
                zmin = trace.zmin if hasattr(trace, 'zmin') else None
                zmax = trace.zmax if hasattr(trace, 'zmax') else None
                zmid = trace.zmid if hasattr(trace, 'zmid') else None

                cmap = _convert_plotly_colorscale_to_cmap(colorscale)

                # Если указан zmid, создаем симметричную цветовую карту
                if zmid is not None and zmin is not None and zmax is not None:
                    norm = plt.Normalize(vmin=zmin, vmax=zmax)
                else:
                    norm = None

                # Рисуем heatmap
                if len(x) == z.shape[1] + 1 and len(y) == z.shape[0] + 1:
                    # Если x и y задают границы ячеек
                    im = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm, shading='flat')
                else:
                    # Если x и y задают центры ячеек
                    im = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm, shading='auto')

                # Название для легенды
                if hasattr(trace, 'name') and trace.name:
                    im.set_label(trace.name)

                # Настройка aspect ratio
                if hasattr(trace, 'ygap') and trace.ygap:
                    ax.set_aspect('equal')

    # Заголовок
    if hasattr(fig_plotly.layout, 'title'):
        if isinstance(fig_plotly.layout.title, dict) and fig_plotly.layout.title.text is not None:
            title_text = fig_plotly.layout.title['text']
            title_text_font_size = fig_plotly.layout.title
            title_fontsize = fig_plotly.layout.title.font.size if fig_plotly.layout.title.font.size else 16
            ax.set_title(title_text, fontsize=title_fontsize, pad=20)

    # Подписи осей
    if hasattr(fig_plotly.layout, 'xaxis') and hasattr(fig_plotly.layout.xaxis, 'title'):
        x_title = fig_plotly.layout.xaxis.title.text if hasattr(fig_plotly.layout.xaxis.title, 'text') else str(
            fig_plotly.layout.xaxis.title)
        if x_title:
            label_fontsize = fig_plotly.layout.xaxis.title.font.size if fig_plotly.layout.xaxis.title.font.size else 14
            ax.set_xlabel(x_title.replace("$$", "$"), fontsize=label_fontsize)
    if hasattr(fig_plotly.layout, 'yaxis') and hasattr(fig_plotly.layout.yaxis, 'title'):
        y_title = fig_plotly.layout.yaxis.title.text if hasattr(fig_plotly.layout.yaxis.title, 'text') else str(
            fig_plotly.layout.yaxis.title)
        if y_title:
            label_fontsize = fig_plotly.layout.yaxis.title.font.size if fig_plotly.layout.yaxis.title.font.size else 14
            ax.set_ylabel(y_title.replace("$$", "$"), fontsize=label_fontsize)

    # Сетка
    if hasattr(fig_plotly.layout, 'xaxis') and hasattr(fig_plotly.layout.xaxis, 'showgrid'):
        ax.grid(fig_plotly.layout.xaxis.showgrid, axis='x')
    if hasattr(fig_plotly.layout, 'yaxis') and hasattr(fig_plotly.layout.yaxis, 'showgrid'):
        ax.grid(fig_plotly.layout.yaxis.showgrid, axis='y')

    # Легенда
    # TODO поправить легенду
    # if any(hasattr(trace, 'name') and trace.name for trace in fig_plotly.data):
    #     ax.legend(fontsize=legend_fontsize, bbox_to_anchor=(1.05, 1))

    # plt.show()

    # Диапазоны осей
    if hasattr(fig_plotly.layout, 'xaxis') and hasattr(fig_plotly.layout.xaxis, 'range'):
        if fig_plotly.layout.xaxis.range:
            ax.set_xlim(fig_plotly.layout.xaxis.range)
    if hasattr(fig_plotly.layout, 'yaxis') and hasattr(fig_plotly.layout.yaxis, 'range'):
        if fig_plotly.layout.yaxis.range:
            ax.set_ylim(fig_plotly.layout.yaxis.range)

    plt.tight_layout()

    save_kwargs = {
        'format': 'eps',
        'dpi': dpi,
        'bbox_inches': 'tight',
        'pad_inches': 0.0,
        'transparent': False,
        'metadata': {
            'Creator': 'Plotly to Matplotlib Converter',
            'Title': title_text if 'title_text' in locals() else 'Plot'
        }
    }

    save_kwargs.update(kwargs)
    plt.savefig(pictures_path / (filename + '.eps'), **save_kwargs)
    plt.savefig(pictures_path / (filename + '.jpg'))
    plt.close(fig)


def _convert_plotly_colorscale_to_cmap(colorscale: str|list) -> Colormap:
    """Конвертирует цветовые шкалы Plotly в Matplotlib Colormap.

    Parameters:
    -----------
    colorscale: str | list
        Название цветовой шкалы Plotly или список цветов

    Returns:
    --------
    Matplotlib Colormap
    """
    # Стандартные цветовые шкалы Plotly и их аналоги в Matplotlib
    plotly_to_mpl = {
        'Viridis': 'viridis',
        'Plasma': 'plasma',
        'Inferno': 'inferno',
        'Magma': 'magma',
        'Cividis': 'cividis',
        'Greys': 'Greys',
        'YlGnBu': 'YlGnBu',
        'YlOrRd': 'YlOrRd',
        'Bluered': 'coolwarm',
        'RdBu': 'RdBu',
        'Picnic': 'Spectral',
        'Portland': 'coolwarm',
        'Jet': 'jet',
        'Hot': 'hot',
        'Blackbody': 'gist_heat',
        'Earth': 'terrain',
        'Electric': 'gist_earth',
        'Rainbow': 'rainbow',
        'Blues': 'Blues',
        'Greens': 'Greens',
        'Reds': 'Reds'
    }

    if isinstance(colorscale, str):
        # Пытаемся найти стандартную цветовую шкалу
        colorscale_name = colorscale.capitalize()
        if colorscale_name in plotly_to_mpl:
            cmap_name = plotly_to_mpl[colorscale_name]
        else:
            cmap_name = colorscale.lower()

        try:
            return cm.get_cmap(cmap_name)
        except:
            return cm.get_cmap('viridis')

    elif isinstance(colorscale, list):
        # Если colorscale - это список цветов
        try:
            colors = [to_hex(c[-1]) if isinstance(c, list) else to_hex(c) for c in colorscale]
            return LinearSegmentedColormap.from_list("custom", colors)
        except:
            return cm.get_cmap('viridis')

    else:
        return cm.get_cmap('viridis')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # Данные для графиков
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)  # Первая линия - синус
    y2 = np.cos(x)  # Вторая линия - косинус

    # Создаем график и рисуем линии
    plt.plot(x, y1, label='sin(x)', color='blue')
    plt.plot(x, y2, label='cos(x)', color='red')

    # Добавляем текст рядом с линиями
    plt.text(5, 0.5, 'Вершина синуса', color='blue',
             horizontalalignment='center', fontsize=10)
    plt.text(3, -0.8, 'Нижняя точка косинуса', color='red',
             horizontalalignment='center', fontsize=10)

    # Добавляем аннотацию со стрелкой
    plt.annotate('Начало косинуса', xy=(0, 1), xytext=(2, 0.7),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 color='red', fontsize=10)

    plt.grid(True)
    plt.legend()
    plt.show()
