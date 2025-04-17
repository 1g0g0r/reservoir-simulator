import numpy as np
import taichi as ti

from paraphin.constants import h, data_type, _re
from paraphin.utils import _pf_o, _pf_w


@ti.data_oriented
class Well:
	def __init__(self, name: str, i: int, j: int, p: float, type_well: str, T: float | None, rw: float):
		self.name = name
		self.i = i
		self.j = j
		self.rw = rw
		self.p = p
		self.T = T
		self.q = ti.field(dtype=data_type, shape=3)
		self.type_well = type_well

	def calc_q(self, p, S, k, mu_o, mu_w) -> ti.field(dtype=data_type, shape=3):

		conductivity_well = 2.0 * np.pi * k[self.i, self.j] * h / np.log(_re / self.rw) * 0.25  # тк участвует только 0.25 дебита
		mult = (p[self.i, self.j] - self.p) * conductivity_well

		if self.type_well == 'inj':
			self.q[0] = 0.0
			self.q[1] = mult / mu_w[self.i, self.j]
		else:
			self.q[0] = mult * _pf_o(S[self.i, self.j]) / mu_o[self.i, self.j]
			self.q[1] = mult * _pf_w(S[self.i, self.j]) / mu_w[self.i, self.j]

		self.q[2] = self.q[0] + self.q[1]
