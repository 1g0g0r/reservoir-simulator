import plotly.graph_objects as go
import numpy as np


if __name__ == '__main__':
    # 1. Подготовка данных (создание примерной сетки и значений)
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)

    # Функция для данных Heatmap и Contour (например, два гауссовых "холма")
    Z_heatmap = np.exp(-(X**2 + Y**2) / 8) + 0.8 * np.exp(-((X-2)**2 + (Y+2)**2) / 4)
    Z_contour = Z_heatmap + 10  # Можно использовать другие данные для изолиний

    # 2. Создание фигуры
    fig = go.Figure()

    # 3. Добавление Heatmap (нижний слой)
    fig.add_trace(go.Heatmap(
        x=x, y=y, z=Z_heatmap,
        colorscale='Viridis',  # Цветовая схема для Heatmap
        name='Тепловая карта', # Название для легенды
        colorbar=dict(title="Интенсивность")  # Заголовок для цветовой шкалы
    ))

    # 4. Добавление Contour (верхний слой, только линии)
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z_contour,
        # Настройки контуров: рисуем ТОЛЬКО линии
        contours=dict(
            coloring='lines',      # Только линии, без фона
            showlabels=True,       # Показать значения на линиях
            labelfont=dict(size=10, color='white')  # Белые подписи для контраста
        ),
        line=dict(width=1.5), colorscale=[[0, 'white'], [1, 'white']],  # Белые линии средней толщины
        name='Изолинии',          # Название для легенды
        showscale=False           # Скрываем отдельную шкалу для Contour
    ))

    # 5. Настройка внешнего вида графика
    fig.update_layout(
        title='Наложение изолиний на Heatmap',
        xaxis_title='Ось X',
        yaxis_title='Ось Y',
        # Равные масштабы осей, чтобы не искажать форму
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=True  # Показать легенду
    )

    fig.show()