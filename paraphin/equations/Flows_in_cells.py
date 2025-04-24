import taichi as ti

from paraphin.constants import Nx, Ny, hx, hy, h, K_o, K_f, K_w, K_p
from paraphin.utils import up_kw, up_ko, mid_Ko_Kw, mid


@ti.func
def flows_in_cells(i, j, p, S, T, k, mu_o, mu_w, m, Wps, dt_val, up_kw_val, up_ko_val) -> None:
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
	m: taichi.field(Nx, Ny)
        Пористость, [-]
    Wps: taichi.field(Nx, Ny)
        Объемная доля масляного компонента в нефти, [-]
	dt_val: taichi.field(Nx, Ny)
		Величина (T_i - T_j) * area / h_ij, [C*м]
	up_kw_val: taichi.field(Nx, Ny)
		Перетоки воды в ячейках, [Па*м]
	up_ko_val: taichi.field(Nx, Ny)
		Перетоки нефти в ячейках, [Па*м]
	"""
	lam = m[i, j] * (S[i, j] * K_w + (1.0 - S[i, j]) * ((1.0 - Wps[i, j]) * K_o + Wps[i, j] * K_p)) + (1.0 - m[i, j]) * K_f

	arr = [[i + 1, j, hx, hy * h], [i - 1, j, hx, hy * h], [i, j + 1, hy, hx * h], [i, j - 1, hy, hx * h]]
	kw, ko, dtemp = 0.0, 0.0, 0.0

	for idx in ti.static(ti.ndrange(4)):
		i1, j1, hij, areaij = arr[idx]
		if (0 <= i1 < Nx) and (0 <= j1 < Ny):
			lam_ij = m[i1, j1] * (S[i1, j1] * K_w + (1.0 - S[i1, j1]) * ((1.0 - Wps[i1, j1]) * K_o +
																		 Wps[i1, j1]* K_p)) + (1.0 - m[i1, j1]) * K_f
			value = areaij * (p[i1, j1] - p[i, j]) / hij * mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
																	 k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1])
			kw += up_kw(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
						k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * value
			ko += up_ko(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
						k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * value
			dtemp += mid(lam, lam_ij) * areaij * (T[i1, j1] - T[i, j]) / hij
			# TODO посмотреть перетоки энергии

	up_kw_val[i, j] = kw
	up_ko_val[i, j] = ko
	dt_val[i, j] = dtemp
