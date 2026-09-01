"""Модуль содержит вспомогательные методы математической части кода и методы визуализации."""
from .boundary_conditions import Bound, TypeBC, DataField, add_bc, apply_bc, get_bound
from .save_data_fields import save_fields
from .preprocess_matrix_and_wells import preprocess_matrix_and_wells
from .well import WellStruct, upd_q_and_eta, calc_well_mult

from .math_utils import (mid_Ko_Kw, mid, up_ko, up_kw, up_T, up_wp, mid_lam, calc_mu_o, calc_mu_w, calc_c_f,
                         calc_c_o, calc_c_w, calc_c_p, pf_o, pf_w, Buckley_Leverett, calc_mobility,
                         up_fraction, solve_band_system)

from .visualisation_utils import (read_solution_data, convert_pkl_files, show_plot, plotly_to_eps,
								  visualize_solution, create_graphs_and_maps, create_gif)
