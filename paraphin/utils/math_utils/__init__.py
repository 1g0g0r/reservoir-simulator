"""Пакет содержит вспомогательные математические процедуры."""
import math

from numba import njit

from .FVM_utils import (mid, lam_heat, calc_mobility, calc_mobility_w, up_fraction, mobility_o,
                        mobility_w, DI, DJ, HIJ, AREA)
from .mg_solver import solve_mg_system, MG_ROWS, MG_TOTAL, MG_COARSE, MG_COARSE_KD, P_STATE
from .guess import project_guess
from .fluids_correlations import calc_mu_o, calc_mu_p, calc_mu_w, crystal_volume_fraction
from .phase_f import pf_o, pf_w, pf_o_mix, pf_w_mix, Buckley_Leverett


@njit(cache=True)
def erfc(x: float) -> float:
    """Реализация функции дополнительной ошибки, совместимая с numba.njit."""
    return 1.0 - math.erf(x)
