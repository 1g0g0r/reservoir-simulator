"""Пакет содержит вспомогательные математические процедуры."""
import taichi as ti
from paraphin.constants import data_type
from .FVM_utils import mid_Ko_Kw, mid, up_ko, up_kw, up_T, up_wp, mid_lam
from .fluids_correlations import calc_mu_o, calc_mu_w, calc_c_f, calc_c_o, calc_c_w, calc_c_p
from .phase_f import pf_o, pf_w, Buckley_Leverett


@ti.func
def ti_erfc(x: data_type) -> data_type:
	"""Аппроксимация Abramowitz and Stegun функции erfc(x) (точность ~1e-7)."""
	result = 0.0
	a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
	a4, a5, p = -1.453152027, 1.061405429, 0.3275911

	x_abs = ti.abs(x)
	t = 1.0 / (1.0 + p * x_abs)

	if x_abs > 6.0:
		if x < 0.0:
			result = 2.0
	else:
		# Вычисление полинома: a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
		y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
		# Умножение на exp(-x^2)
		y *= ti.exp(-x_abs * x_abs)

		# Коррекция для отрицательных значений: erfc(-x) = 2 - erfc(x)
		if x >= 0:
			result = y
		else:
			result = 2.0 - y

	return result
