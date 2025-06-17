import numpy as np
import plotly.graph_objects as go


z_min = min(np.min(Z1), np.min(Z2))
z_max = max(np.max(Z1), np.max(Z2))
levels = np.linspace(z_min, z_max, num=10)  # 10 уровней


fig = go.Figure()

# Первая трасса (например, синие изолинии)
fig.add_trace(go.Contour(
    x=x, y=y, z=Z1,
    colorscale='Blues',
    contours=dict(levels=levels),
    showscale=False  # Отключить цветовую шкалу для первой трассы
))

# Вторая трасса (например, красные изолинии)
fig.add_trace(go.Contour(
    x=x, y=y, z=Z2,
    colorscale='Reds',
    contours=dict(levels=levels),
    showscale=True  # Включить цветовую шкалу для второй трассы
))

fig.update_layout(
    title="Изолинии двух полей с совпадающими уровнями",
    xaxis_title="X", yaxis_title="Y",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.show()
