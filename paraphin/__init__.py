"""Модуль решения задачи двухфазной неизотермической фильтрации с учетом кольматаци пласта парафином."""
import numpy as np

from paraphin.constants import data_type, Nx, Ny, Nr

N = Nx * Ny  # размер матрицы
NN = 5 * Nx * Ny - 2 * (Nx + Ny)  # количество ненулевых элементов в матрице давления

r = np.linspace(0, 40 * 1e-6, Nr, endpoint=True)
# fi_0 = np.array([0.0, 0.013, 0.023, 0.031, 0.035, 0.034, 0.027, 0.021, 0.016, 0.018, 0.025, 0.032, 0.041, 0.052, 0.061, 0.073, 0.082, 0.086, 0.081, 0.07, 0.059, 0.048, 0.035, 0.024, 0.013, 0])
_sigma = 2.0 * np.pi
_m = np.max(r) / 2

# TODO перейти на лог-нормальное распределение
fi_0_np = np.exp(-0.5 * ((r - _m) / 1e-6 / _sigma)**2) / _sigma
fi_0_np[0] = fi_0_np[-1] = 0.0
fi_0 = fi_0_np / np.sum(fi_0_np)

# массивы радиусов пор в необходимых степенях
r1 = r
r2 = r * r
r3 = r2 * r
r4 = r3 * r
r5 = r4 * r
r6 = r5 * r


if __name__ == '__main__':
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=fi_0_np, mode='lines'))
    fig.show()
