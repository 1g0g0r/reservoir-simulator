import taichi as ti

from paraphin.constants import Nx, Ny, hx, hy, h
from paraphin.utils import up_kw, up_ko, mid


@ti.func
def flows_in_cells(i, j, p, S, T, k, mu_o, mu_w, dt_val, up_kw_val, up_ko_val) -> None:
	"""Вычисление потоков в ячейках.

	Parameters
	----------
	i, j: int
		Индексы текущей ячейки, [-]
	p: taichi.field(Nx, Ny)
		Давление, [Па]
	S: taichi.field(Nx, Ny)
		Водонасыщенность, [-]
	T: taichi.field(Nx, Ny)
		Температура, [C]
	k: taichi.field(Nx, Ny)
		Проницаемость, [м^2]
	mu_o: taichi.field(Nx, Ny)
		Вязкость нефти, [Па*с]
	mu_w: taichi.field(Nx, Ny)
		Вязкость воды, [Па*с]
	dt_val: taichi.field(Nx, Ny)
		Величина (T_i - T_j) * area / h_ij, [C*м]
	up_kw_val: taichi.field(Nx, Ny)
		Перетоки воды в ячейках, [Па*м]
	up_ko_val: taichi.field(Nx, Ny)
		Перетоки нефти в ячейках, [Па*м]
	"""
	arr = [[i + 1, j, hx, hy * h], [i - 1, j, hx, hy * h], [i, j + 1, hy, hx * h], [i, j - 1, hy, hx * h]]
	kw, ko, dtemp = 0.0, 0.0, 0.0

	for idx in ti.static(ti.ndrange(4)):
		i1, j1, hij, areaij = arr[idx]
		if (0 <= i1 < Nx) and (0 <= j1 < Ny):
			value = areaij * (p[i1, j1] - p[i, j]) / hij * mid(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
															   k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1])
			kw += up_kw(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
						k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * value
			ko += up_ko(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
						k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * value
			dtemp += areaij * (T[i1, j1] - T[i, j]) / hij

	up_kw_val[i, j] = kw
	up_ko_val[i, j] = ko
	dt_val[i, j] = dtemp
