import numpy as np
import taichi as ti

from paraphin.constants import data_type, Nx, Ny, Nr

N = Nx * Ny  # размер матрицы
NN = (Nx - 2) * (Ny - 2) * 5 + (Nx - 2) * 8 + (Ny - 2) * 8 + 12  # количество ненулевых элементов в матрице давления

r = np.linspace(0, 40 * 1e-6, Nr, endpoint=True)
# fi_0 = np.array([0.0, 0.013, 0.023, 0.031, 0.035, 0.034, 0.027, 0.021, 0.016, 0.018, 0.025, 0.032, 0.041, 0.052, 0.061, 0.073, 0.082, 0.086, 0.081, 0.07, 0.059, 0.048, 0.035, 0.024, 0.013, 0])
_sigma = 2.0 * np.pi
_m = np.max(r) / 2
fi_0 = np.exp(-0.5 * ((r - _m) / 1e-6 / _sigma)**2) / _sigma
fi_0[0] = fi_0[-1] = 0.0
fi_0 = fi_0 / np.sum(fi_0)

# массивы радиусов пор в необходимых степенях
r1 = ti.field(dtype=data_type, shape=Nr)
r2 = ti.field(dtype=data_type, shape=Nr)
r3 = ti.field(dtype=data_type, shape=Nr)
r4 = ti.field(dtype=data_type, shape=Nr)
r5 = ti.field(dtype=data_type, shape=Nr)
r6 = ti.field(dtype=data_type, shape=Nr)
r2_np = r * r
r3_np = r2_np * r
r4_np = r3_np * r
r5_np = r4_np * r
r6_np = r5_np * r
r1.from_numpy(r)
r2.from_numpy(r2_np)
r3.from_numpy(r3_np)
r4.from_numpy(r4_np)
r5.from_numpy(r5_np)
r6.from_numpy(r6_np)


if __name__ == '__main__':
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=fi_0, mode='lines'))
    fig.show()
