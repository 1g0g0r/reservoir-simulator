import taichi as ti

from .constants import data_type, dt
from .utils.math_utils import pf_w, pf_o, Buckley_Leverett

WellStruct = ti.types.struct(
	i = ti.i32,
	j = ti.i32,
	idx_rhs = ti.i32,
	idx_mat = ti.i32,
	rw = data_type,
	p = data_type,
	T = data_type,
	is_injector = ti.i32,
	productivity_mult = data_type,
	q = ti.types.vector(3, data_type),
	Q = ti.types.vector(3, data_type),
	eta = data_type,
	dp = data_type
)


@ti.func
def upd_q_and_eta(well, p, S, k, mu_o, mu_w) -> WellStruct:
	"""Вычисление дебета скважины."""
	well.dp = p[well.i, well.j] - well.p
	mult = well.dp * well.productivity_mult * k[well.i, well.j]

	if well.is_injector == 1:
		well.q[0] = 0.0
		well.q[1] = mult / mu_w[well.i, well.j]
		well.eta = 1.0

	else:
		well.q[0] = mult * pf_o(S[well.i, well.j]) / mu_o[well.i, well.j]
		well.q[1] = mult * pf_w(S[well.i, well.j]) / mu_w[well.i, well.j]
		well.eta = Buckley_Leverett(S[well.i, well.j], mu_w[well.i, well.j], mu_o[well.i, well.j])

	well.q[2] = well.q[0] + well.q[1]

	well.Q[0] += well.q[0] * dt
	well.Q[1] += well.q[1] * dt
	well.Q[2] += well.q[2] * dt

	return well


@ti.func
def _deprecated_calc_well_mult(well, S, k, mu_o, mu_w) -> float:
	"""Вычисление множителя дебета скважины."""
	ret = 0.0
	mult = well.productivity_mult * k[well.i, well.j]

	if well.is_injector == 1:
		ret = mult / mu_w[well.i, well.j]
	else:
		ret = mult * (pf_o(S[well.i, well.j]) / mu_o[well.i, well.j] +
					  pf_w(S[well.i, well.j]) / mu_w[well.i, well.j])

	return ret
