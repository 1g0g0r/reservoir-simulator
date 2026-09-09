"""Модуль содержит вспомогательные методы математической части кода.

Визуализация сюда намеренно не ре-экспортируется: `visualisation_utils` тянет matplotlib, plotly,
PIL и joblib (полсекунды импорта), а расчету они не нужны. Точки входа берут ее напрямую из
`paraphin.utils.visualisation_utils`.
"""
from .boundary_conditions import Bound, TypeBC, DataField, add_bc, apply_bc, get_bound
from .read_data_files import convert_pkl_files, read_solution_data
from .save_data_fields import save_fields
from .well import WellStruct, upd_q_and_eta, calc_well_pi, preprocess_wells

from .math_utils import (mid, lam_heat, calc_mu_o, calc_mu_w, pf_o, pf_w,
                         Buckley_Leverett, calc_mobility, up_fraction, solve_band_system,
                         mobility_o, mobility_w, DI, DJ, HIJ, AREA)
