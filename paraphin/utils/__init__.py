"""Модуль содержит вспомогательные методы математической части кода и методы визуализации."""
from .bc_enums import Bound, TypeBC

from .math_utils import (mid_Ko_Kw, mid, up_ko, up_kw, K_w, K_o, up_T, preprocess_matrix_and_wells,
						 calc_mu_o, calc_mu_w, calc_c_f, calc_c_o, calc_c_w, calc_c_p, calc_Um_r2,
						 pf_o, pf_w, Buckley_Leverett)

from .visualisation_utils import read_pkl_files, show_plot, visualize_solution, create_diplom_graphs
