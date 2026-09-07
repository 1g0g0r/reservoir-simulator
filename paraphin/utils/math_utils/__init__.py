"""Пакет содержит вспомогательные математические процедуры."""
import math

from numba import njit

from .FVM_utils import (mid, lam_heat, calc_mobility, up_fraction, mobility_o,
                        mobility_w, DI, DJ, HIJ, AREA)
from .band_solver import solve_band_system
from .fluids_correlations import calc_mu_o, calc_mu_w, calc_c_f, calc_c_o, calc_c_w, calc_c_p
from .phase_f import pf_o, pf_w, Buckley_Leverett


@njit(cache=True)
def erfc(x: float) -> float:
    """Реализация функции дополнительной ошибки, совместимая с numba.njit."""
    return 1.0 - math.erf(x)
