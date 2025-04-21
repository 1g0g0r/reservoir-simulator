import numpy as np
import taichi as ti

from paraphin.constants import h, data_type, Nx, hx, hy
from paraphin.utils import _pf_o, _pf_w


@ti.data_oriented
class Well:
	def __init__(self, name: str, i: int, j: int, p: float, is_injector: bool, T: float | None, rw: float):
		self.name = name
		self.i = i
		self.j = j
		self.rw = rw
		self.p = p
		self.T = T
		self.is_injector = is_injector
		self.q = ti.field(dtype=data_type, shape=3)
		self.idx = i + j * Nx

	def conductivity(self) -> float:
		"""Вычисление множителя продуктивности скважины."""
		_re = 0.14 * np.sqrt(hx * hx + hy * hy)  # Радиус контура питания скважины, [м]
		return 2.0 * np.pi * h / np.log(_re / self.rw) * 0.25  # тк участвует только 0.25 дебита

	def calc_q(self, p, S, k, mu_o, mu_w) -> ti.field(dtype=data_type, shape=3):
		"""Вычисление дебета скважины."""
		conductivity_well = self.conductivity()
		if isinstance(p, float):
			mult = (p - self.p) * k[self.i, self.j] * conductivity_well
		else:
			mult = (p[self.i, self.j] - self.p) * k[self.i, self.j] * conductivity_well

		if self.is_injector:
			self.q[0] = 0.0
			self.q[1] = mult / mu_w[self.i, self.j]
		else:
			self.q[0] = mult * _pf_o(S[self.i, self.j]) / mu_o[self.i, self.j]
			self.q[1] = mult * _pf_w(S[self.i, self.j]) / mu_w[self.i, self.j]

		self.q[2] = self.q[0] + self.q[1]
