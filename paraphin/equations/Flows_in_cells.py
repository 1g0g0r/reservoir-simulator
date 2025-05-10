import taichi as ti

from paraphin.constants import Nx, Ny, hx, hy, h, K_o, K_f, K_w, K_p, ro_w, ro_o, ro_p
from paraphin.utils import up_kw, up_ko, mid_Ko_Kw, mid, up_T


@ti.func
def flows_in_cells(i, j, p, S, T, k, mu_o, mu_w, m, Wps, C_o, C_w, C_p, temp_eq_val, up_kw_val, up_ko_val) -> None:
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
    C_o: taichi.field(Nx, Ny)
        Теплоемкость нефти, [Дж/(кг*C)]
    C_w: taichi.field(Nx, Ny)
        Теплоемкость воды, [Дж/(кг*C)]
    C_p: taichi.field(Nx, Ny)
        Теплоемкость парафина, [Дж/(кг*C)]
	temp_eq_val: taichi.field(Nx, Ny)
		Сумма величин перетоков тепла в уравнении энергии
	up_kw_val: taichi.field(Nx, Ny)
		Перетоки воды в ячейках, [Па*м]
	up_ko_val: taichi.field(Nx, Ny)
		Перетоки нефти в ячейках, [Па*м]
	"""
	lam = m[i, j] * (S[i, j] * K_w + (1.0 - S[i, j]) * ((1.0 - Wps[i, j]) * K_o + Wps[i, j] * K_p)) + (1.0 - m[i, j]) * K_f
	T_Co = ro_o * C_o[i, j] * (1.0 - Wps[i, j]) + ro_p * Wps[i, j] * C_p[i, j]

	arr = [[i + 1, j, hx, hy * h], [i - 1, j, hx, hy * h], [i, j + 1, hy, hx * h], [i, j - 1, hy, hx * h]]
	kw, ko, t = 0.0, 0.0, 0.0

	for idx in ti.static(ti.ndrange(4)):
		i1, j1, hij, areaij = arr[idx]
		if (0 <= i1 < Nx) and (0 <= j1 < Ny):
			# TODO придумать, как объединить вычисление слагаемых вверх по потоку

			T_Co_ij = ro_o * C_o[i1, j1] * (1.0 - Wps[i1, j1]) + ro_p * Wps[i1, j1] * C_p[i1, j1]
			lam_ij = m[i1, j1] * (S[i1, j1] * K_w + (1.0 - S[i1, j1]) * ((1.0 - Wps[i1, j1]) * K_o +
																		 Wps[i1, j1]* K_p)) + (1.0 - m[i1, j1]) * K_f
			value = (p[i1, j1] - p[i, j]) * areaij / hij * mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
																	 k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1])
			up_k_w = up_kw(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
						   k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * value
			up_k_o = up_ko(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
						   k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * value
			up_t = up_T(p[i, j], T[i, j], p[i1, j1], T[i1, j1])

			ko += up_k_o
			kw += up_k_w
			t += mid(lam, lam_ij) * areaij * (T[i1, j1] - T[i, j]) / hij
			t += C_w[i, j] * ro_w * up_t * up_k_w
			t += mid(T_Co, T_Co_ij) * up_t * up_k_o

	up_kw_val[i, j] = kw
	up_ko_val[i, j] = ko
	temp_eq_val[i, j] = t
