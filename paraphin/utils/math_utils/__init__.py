"""Пакет содержит вспомогательные математические процедуры."""
from .FVM_utils import mid_Ko_Kw, mid, up_ko, up_kw, up_T, up_wp, up_lam
from .fluids_correlations import calc_mu_o, calc_mu_w, calc_c_f, calc_c_o, calc_c_w, calc_c_p
from .phase_f import pf_o, pf_w, Buckley_Leverett
from .preprocess_matrix_and_wells import preprocess_matrix_and_wells
