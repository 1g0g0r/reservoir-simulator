"""Enum классы и методы описания граничных условий."""
from enum import Enum
import taichi as ti


class TypeBC(Enum):
	Dirichlet: str  # 1
	Neyman: str     # 0


class Bound(Enum):
	Left: str
	Right: str
	Top: str
	Bottom: str


def apply_bc(type: TypeBC, bound: Bound, data_field: ti.template(), value):
	# если гу стандартное (непротекания), то сразу же возвращает ноооль
	pass


DataFields = ti.types.struct(
    Pressure    = ti.types.struct(type=ti.i32, value=ti.f32),
    Saturation  = ti.types.struct(type=ti.i32, value=ti.f32),
    Temperature = ti.types.struct(type=ti.i32, value=ti.f32)
)

Boundary = ti.types.struct(
    Left   = DataFields,
    Right  = DataFields,
    Bottom = DataFields,
    Top	   = DataFields
)

# Создание поля структур
ffields = Boundary()
print()
