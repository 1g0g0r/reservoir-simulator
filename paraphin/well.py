import taichi as ti

from paraphin.constants import data_type
from paraphin.utils import pf_w, pf_o, Buckley_Leverett

WellStruct = ti.types.struct(
	i = ti.i32,
	j = ti.i32,
	rw = data_type,
	p = data_type,
	T = data_type,
	is_injector = ti.i32,
	conductivity_mult = data_type,
	q = ti.types.vector(3, data_type),
	idx_rhs = ti.i32,
	idx_mat = ti.i32,
	eta = data_type
)

@ti.func
def upd_q_and_eta(well, p, S, k, mu_o, mu_w) -> WellStruct:
	"""Вычисление дебета скважины."""
	mult = (p[well.i, well.j] - well.p) * k[well.i, well.j] * well.conductivity_mult

	if well.is_injector == 1:
		well.q[0] = 0.0
		well.q[1] = mult / mu_w[well.i, well.j]
	else:
		well.q[0] = mult * pf_o(S[well.i, well.j]) / mu_o[well.i, well.j]
		well.q[1] = mult * pf_w(S[well.i, well.j]) / mu_w[well.i, well.j]

	well.q[2] = well.q[0] + well.q[1]
	well.eta = Buckley_Leverett(S[well.i, well.j], mu_w[well.i, well.j], mu_o[well.i, well.j])

	return well


@ti.func
def calc_q_mult(well, S, k, mu_o, mu_w) -> data_type:
	"""Вычисление дебета скважины."""
	q_o, q_w = 0.0, 0.0
	mult = k[well.i, well.j] * well.conductivity_mult

	if well.is_injector == 1:
		q_o = 0.0
		q_w = mult / mu_w[well.i, well.j]
	else:
		q_o = mult * pf_o(S[well.i, well.j]) / mu_o[well.i, well.j]
		q_w = mult * pf_w(S[well.i, well.j]) / mu_w[well.i, well.j]

	return q_o + q_w
