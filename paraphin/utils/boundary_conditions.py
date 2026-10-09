"""Enum классы и методы описания граничных условий."""
import numpy as np
from enum import Enum
from numba import njit

from paraphin.constants import Nx, Ny, data_type


class TypeBC(Enum):
    Neumann    = 0
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
    Paraffin    = 3  # Суммарная доля парафина во втекающей нефти; учитывается только Дирихле, см. `flows_in_cells`


# Граничные условия - массив [граница, поле] (индексы - `Bound`, `DataField`) с dtype BC: тип (`TypeBC`) и значение.
# В ядрах - `bc[bound, PARAFFIN].type == DIRICHLET`, в Python - `bc['value'][bound, field]`
BC = np.dtype([('type', np.int64), ('value', np.float64)])
PRESSURE, SATURATION, TEMPERATURE, PARAFFIN = (f.value for f in DataField)
DIRICHLET = TypeBC.Dirichlet.value


def add_bc(boundary_conditions, boundary: int, field: int, type_bc: int, value: float):
    boundary_conditions['type'][boundary, field] = type_bc
    boundary_conditions['value'][boundary, field] = value


@njit(cache=True)
def apply_bc(boundary_conditions: np.ndarray, bound: int, data_field_idx: int, data_field: np.ndarray,
             i: int, j: int, h: data_type) -> data_type:
    """Учет граничных условий на границе области."""
    ret = 0.0
    bc = boundary_conditions[bound, data_field_idx]
    if bc.type == DIRICHLET:  # Дирихле (1 рода)
        ret = bc.value
    else:                     # Неймана (2 рода)
        ret = data_field[i, j] + bc.value * h

    return ret


@njit(cache=True)
def get_bound(i: int, j: int) -> int:
    """Определение границы области по вылету индекса за сетку.

    Соответствие индексов и enum `Bound`: j за сеткой - это Left/Right, i за сеткой - Top/Bottom.
    Enum разворачивается в константы на этапе компиляции, поэтому njit это не мешает.
    """
    bound = Bound.Left.value
    if j < 0:
        bound = Bound.Left.value
    elif j > Ny - 1:
        bound = Bound.Right.value
    elif i < 0:
        bound = Bound.Top.value
    elif i > Nx - 1:
        bound = Bound.Bottom.value

    return bound
