"""Enum классы и методы описания граничных условий."""
from enum import Enum
import taichi as ti


class TypeBC(Enum):
	Dirichlet: str
	Neyman: str


class Bound(Enum):
	Left: str
	Right: str
	Top: str
	Bottom: str


def apply_bc(type: TypeBC, bound: Bound, data_field: ti.template(), value):
	# если гу стандартное (непротекания), то сразу же возвращает ноооль
	pass


MyStruct = ti.types.struct(
    oil=ti.f32,
    water=ti.f32,
    total=ti.f32,
    eta=ti.f32
)

# Создание поля структур
n_fields = 3  # давление/насыщенность/температура
wells = MyStruct.field(shape=(n_fields,))
