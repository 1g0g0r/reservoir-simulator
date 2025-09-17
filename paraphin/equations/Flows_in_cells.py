"""Вычисление перетоков массы и энергии при решении методом конечных объемов на прямоугольной сетке."""
import taichi as ti

from paraphin.constants import Nx, Ny, hx, hy, h, ro_w, ro_o, ro_p
from paraphin.utils import up_kw, up_ko, mid_Ko_Kw, mid, up_T, up_wp, mid_lam


@ti.func
def flows_in_cells(i, j, p, S, T, k, mu_o, mu_w, m, Wp, Wps, C_o, C_w, C_p, cells_T_eq, cells_Wp_eq, cells_S_eq) -> None:
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
    Wp: taichi.field(Nx, Ny)
        Массовая доля растворенного парафина в нефти, [-]
    Wps: taichi.field(Nx, Ny)
        Массовая доля взвешенного парафина в нефти, [-]
    C_o: taichi.field(Nx, Ny)
        Теплоемкость нефти, [Дж/(кг*C)]
    C_w: taichi.field(Nx, Ny)
        Теплоемкость воды, [Дж/(кг*C)]
    C_p: taichi.field(Nx, Ny)
        Теплоемкость парафина, [Дж/(кг*C)]
	cells_T_eq: taichi.field(Nx, Ny)
		Сумма величин перетоков тепла в уравнении энергии
	cells_S_eq: taichi.field(Nx, Ny)
		Перетоки воды в ячейках, [Па*м]
	cells_Wp_eq: taichi.field(Nx, Ny)
		Перетоки нефти в ячейках, [Па*м]
	"""
	Co = ro_o * C_o[i, j] * (1.0 - Wps[i, j]) + ro_p * Wps[i, j] * C_p[i, j]

	arr = [[i + 1, j, hx, hy * h], [i - 1, j, hx, hy * h], [i, j + 1, hy, hx * h], [i, j - 1, hy, hx * h]]
	_s, _wp, _t = 0.0, 0.0, 0.0
	for idx in ti.static(ti.ndrange(4)):
		i1, j1, hij, areaij = arr[idx]
		if (0 <= i1 < Nx) and (0 <= j1 < Ny):
			# TODO придумать, как объединить вычисление слагаемых вверх по потоку
			Co_ij = ro_o * C_o[i1, j1] * (1.0 - Wps[i1, j1]) + ro_p * Wps[i1, j1] * C_p[i1, j1]
			value = (p[i1, j1] - p[i, j]) * areaij / hij * mid_Ko_Kw(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
																	 k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1])
			up_k_w = up_kw(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
						   k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * value
			up_k_o = up_ko(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
						   k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * value
			up_t = up_T(p[i, j], T[i, j], p[i1, j1], T[i1, j1])

			_wp += up_k_o * up_wp(p[i, j], Wp[i, j], Wps[i, j], p[i1, j1], Wp[i1, j1], Wps[i1, j1])
			_s += up_k_w
			_t += (T[i1, j1] - T[i, j]) * areaij / hij * mid_lam(S[i, j], m[i, j], Wps[i, j],
																 S[i1, j1], m[i1, j1], Wps[i1, j1])
			_t += C_w[i, j] * ro_w * up_t * up_k_w
			_t += mid(Co, Co_ij) * up_t * up_k_o

		else:
			# TODO для каждого случая нужно как-то обработать: температуру, насыщенность, массовую доля взвешенного парафина
			# обрабатывать в цикле по полям данных (а может ли taichi вообще так????)
			if i == 0:
				# гу на левой границе
				pass
			elif i == Nx-1:
				# гу на правой границе
				pass
			if j == 0:
				# гу на нижней границе
				pass
			elif j == Ny-1:
				# гу на верхней границе
				pass


	cells_S_eq[i, j] = _s
	cells_Wp_eq[i, j] = _wp
	cells_T_eq[i, j] = _t
