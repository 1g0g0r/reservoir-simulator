"""Модуль содержит вспомогательные методы математической части кода и методы визуализации."""
from .boundary_conditions import Bound, TypeBC
from .save_data_fields import save_fields

from .math_utils import (mid_Ko_Kw, mid, up_ko, up_kw, up_T, up_wp, mid_lam, preprocess_matrix_and_wells,
						 calc_mu_o, calc_mu_w, calc_c_f, calc_c_o, calc_c_w, calc_c_p, pf_o, pf_w, Buckley_Leverett)

from .visualisation_utils import (read_solution_data, convert_pkl_files, show_plot,
								  visualize_solution, create_graphs_and_maps, create_gif)
