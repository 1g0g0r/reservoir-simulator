"""Enum классы и методы описания граничных условий."""
from enum import Enum
import taichi as ti

from paraphin.constants import Nx, Ny, data_type


class TypeBC(Enum):
    Neyman    = 0
    Dirichlet = 1


class Bound(Enum):
    Left   = 0
    Right  = 1
    Top    = 2
    Bottom = 3


class DataField(Enum):
    Pressure    = 0
    Saturation  = 1
    Temperature = 2


def add_bc(boundary_conditions, boundary: int, field: int, type_bc: int, value: float):
    boundary_conditions[boundary, field, 0] = type_bc
    boundary_conditions[boundary, field, 1] = value


@ti.func
def apply_bc(boundary_conditions: ti.template(), bound: ti.int32, data_field_idx: ti.int32,
             data_field: ti.template(), i: ti.int32, j: ti.int32, h: data_type) -> data_type:
    """Учет граничных условий на границе области."""
    ret = 0.0
    bc_type  = boundary_conditions[bound, data_field_idx, 0]
    bc_value = boundary_conditions[bound, data_field_idx, 1]

    if bc_type == 1:  # Дирихле (1 рода)
        ret = bc_value
    else:             # Неймана (2 рода)
        ret = data_field[i, j] + bc_value * h

    return ret


@ti.func
def get_bound(i: ti.int32, j: ti.int32) -> ti.int32:
    """Определение границы области по индексу ячейки."""
    bound = 0
    if j < 0:         # левая граница
        bound = 0
    elif j > Ny - 1:  # правая граница
        bound = 1
    elif i < 0:       # нижняя граница
        bound = 2
    elif i > Nx - 1:  # верхняя граница
        bound = 3

    return bound
