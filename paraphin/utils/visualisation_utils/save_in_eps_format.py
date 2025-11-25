"""Сохранение графика plotly в векторном формате eps"""
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plotly_to_eps(fig_plotly: go.Figure, filename: str, figsize: tuple = (10, 6), dpi: int = 1000,
                  title_fontsize: int = 14, label_fontsize: int = 12, legend_fontsize: int = 10, **kwargs) -> None:
    """Сохраняет график Plotly в формате EPS с промежуточным преобразованием в Matplotlib.

    Parameters:
    -----------
    fig_plotly: go.Figure
        Plotly Figure объект
    filename: str
        Имя файла для сохранения (с расширением .eps)
    figsize: tuple
        Размер фигуры в дюймах (ширина, высота)
    dpi: int
        Разрешение для EPS (чем выше, тем лучше качество)
    title_fontsize: int
        Размер шрифта заголовка
    label_fontsize: int
        Размер шрифта меток осей
    legend_fontsize: int
        Размер шрифта легенды

    Example:
    -------
    plotly_to_matplotlib_eps(
        fig_plotly=fig,
        filename='plotly_graph.eps',
        figsize=(10, 6),
        dpi=1200,
        title_fontsize=16,
        label_fontsize=14,
        legend_fontsize=12
    )
    """

    # Создаем Matplotlib фигуру и оси
    fig, ax = plt.subplots(figsize=figsize, dpi=100)  # dpi=100 для отображения, реальное будет в savefig

    # Переносим данные из Plotly в Matplotlib
    for trace in fig_plotly.data:
        trace_type = trace.type

        if trace_type == 'scatter':
            # Обрабатываем scatter plot (линии и/или точки)
            x = trace.x
            y = trace.y

            line_style = '-'
            marker_style = 'o'

            # Определяем стиль линии и маркеров
            if hasattr(trace, 'mode'):
                if 'lines' not in trace.mode:
                    line_style = ''
                if 'markers' not in trace.mode:
                    marker_style = ''
                if 'lines+markers' in trace.mode:
                    line_style = '-'
                    marker_style = 'o'

            # Получаем цвет
            color = trace.line.color if hasattr(trace, 'line') and hasattr(trace.line, 'color') else None
            if color is None and hasattr(trace, 'marker') and hasattr(trace.marker, 'color'):
                color = trace.marker.color

            # Получаем название для легенды
            name = trace.name if hasattr(trace, 'name') and trace.name else None

            # Добавляем на график
            ax.plot(x, y,
                    linestyle=line_style,
                    marker=marker_style,
                    color=color,
                    label=name,
                    linewidth=2,
                    markersize=8)

        elif trace_type == 'bar':
            # Обрабатываем bar plot
            x = trace.x
            y = trace.y
            color = trace.marker.color if hasattr(trace, 'marker') and hasattr(trace.marker, 'color') else None
            name = trace.name if hasattr(trace, 'name') and trace.name else None

            ax.bar(x, y, color=color, label=name, alpha=0.7)

        elif trace_type == 'histogram':
            # Обрабатываем histogram
            x = trace.x
            color = trace.marker.color if hasattr(trace, 'marker') and hasattr(trace.marker, 'color') else None
            name = trace.name if hasattr(trace, 'name') and trace.name else None

            ax.hist(x, bins=30, color=color, alpha=0.7, label=name)


    # Заголовок
    if hasattr(fig_plotly.layout, 'title'):
        if isinstance(fig_plotly.layout.title, dict) and fig_plotly.layout.title.text is not None:
            title_text = fig_plotly.layout.title['text']
            ax.set_title(title_text, fontsize=title_fontsize, pad=20)

    # Подписи осей
    if hasattr(fig_plotly.layout, 'xaxis') and hasattr(fig_plotly.layout.xaxis, 'title'):
        x_title = fig_plotly.layout.xaxis.title.text if hasattr(fig_plotly.layout.xaxis.title, 'text') else str(
            fig_plotly.layout.xaxis.title)
        if x_title:
            ax.set_xlabel(x_title, fontsize=label_fontsize)

    if hasattr(fig_plotly.layout, 'yaxis') and hasattr(fig_plotly.layout.yaxis, 'title'):
        y_title = fig_plotly.layout.yaxis.title.text if hasattr(fig_plotly.layout.yaxis.title, 'text') else str(
            fig_plotly.layout.yaxis.title)
        if y_title:
            ax.set_ylabel(y_title, fontsize=label_fontsize)

    # Сетка
    if hasattr(fig_plotly.layout, 'xaxis') and hasattr(fig_plotly.layout.xaxis, 'showgrid'):
        ax.grid(fig_plotly.layout.xaxis.showgrid, axis='x', alpha=0.3)
    if hasattr(fig_plotly.layout, 'yaxis') and hasattr(fig_plotly.layout.yaxis, 'showgrid'):
        ax.grid(fig_plotly.layout.yaxis.showgrid, axis='y', alpha=0.3)

    # Легенда
    if any(hasattr(trace, 'name') and trace.name for trace in fig_plotly.data):
        ax.legend(fontsize=legend_fontsize, loc='best')

    # Диапазоны осей
    if hasattr(fig_plotly.layout, 'xaxis') and hasattr(fig_plotly.layout.xaxis, 'range'):
        if fig_plotly.layout.xaxis.range:
            ax.set_xlim(fig_plotly.layout.xaxis.range)

    if hasattr(fig_plotly.layout, 'yaxis') and hasattr(fig_plotly.layout.yaxis, 'range'):
        if fig_plotly.layout.yaxis.range:
            ax.set_ylim(fig_plotly.layout.yaxis.range)

    plt.tight_layout()  # Настройки для EPS

    save_kwargs = {
        'format': 'eps',
        'dpi': dpi,
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
        'transparent': False,
        'metadata': {
            'Creator': 'Plotly to Matplotlib Converter',
            'Title': title_text if 'title_text' in locals() else 'Plot'
        }
    }

    save_kwargs.update(kwargs)
    plt.savefig(filename, **save_kwargs)
    plt.close(fig)
