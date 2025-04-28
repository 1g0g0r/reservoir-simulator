import taichi as ti

from paraphin.constants import data_type, ro_w, ro_p, ro_o
from paraphin.utils.phase_f import pf_o, pf_w


@ti.func
def mid(x: data_type, y: data_type) -> data_type:
    """Среднее значение"""
    return 2.0 * x * y / (x + y)


@ti.func
def mid_Ko_Kw(k_i: data_type, s_i: data_type, mu_o_i: data_type, mu_w_i: data_type,
              k_j: data_type, s_j: data_type, mu_o_j: data_type, mu_w_j: data_type) -> data_type:
    """mid(Ko + Kw)_ij"""
    x = K_o(k_i, s_i, mu_o_i) + K_w(k_i, s_i, mu_w_i)
    y = K_o(k_j, s_j, mu_o_j) + K_w(k_j, s_j, mu_w_j)

    return 2.0 * x * y / (x + y)

# TODO придумать, как объединить все слагаемые вверх по потоку

@ti.func
def up_kw(k_i: data_type, s_i: data_type, p_i: data_type, mu_o_i: data_type, mu_w_i: data_type,
		  k_j: data_type, s_j: data_type, p_j: data_type, mu_o_j: data_type, mu_w_j: data_type) -> data_type:
    """Значение берется вверх по потоку: up(kw / (ko + kw)"""
    ret = 0.0

    if p_i >= p_j:
        ret = K_w(k_i, s_i, mu_w_i) / (K_w(k_i, s_i, mu_w_i) + K_o(k_i, s_i, mu_o_i))
    else:
        ret = K_w(k_j, s_j, mu_w_j) / (K_w(k_j, s_j, mu_w_j) + K_o(k_j, s_j, mu_o_j))

    return ret


@ti.func
def up_ko(k_i: data_type, s_i: data_type, p_i: data_type, mu_o_i: data_type, mu_w_i: data_type,
		  k_j: data_type, s_j: data_type, p_j: data_type, mu_o_j: data_type, mu_w_j: data_type) -> data_type:
    """Значение берется вверх по потоку: up(ko / (ko + kw)"""
    ret = 0.0

    if p_i >= p_j:
        ret = K_o(k_i, s_i, mu_o_i) / (K_w(k_i, s_i, mu_w_i) + K_o(k_i, s_i, mu_o_i))
    else:
        ret = K_o(k_j, s_j, mu_o_j) / (K_w(k_j, s_j, mu_w_j) + K_o(k_j, s_j, mu_o_j))

    return ret


@ti.func
def up_t_ko(k_i: data_type, s_i: data_type, p_i: data_type, mu_o_i: data_type, mu_w_i: data_type, T_i: data_type,
            C_o_i: data_type, C_p_i: data_type, Wps_i: data_type,
		    k_j: data_type, s_j: data_type, p_j: data_type, mu_o_j: data_type, mu_w_j: data_type, T_j: data_type,
            C_o_j: data_type, C_p_j: data_type, Wps_j: data_type,) -> data_type:
    """Значение берется вверх по потоку: up(kw / (ko + kw)"""
    ret = 0.0

    if p_i >= p_j:
        ret = T_i * (ro_o * C_o_i * (1.0 - Wps_i) + ro_p * Wps_i * C_p_i) * K_o(k_i, s_i, mu_w_i) / (K_w(k_i, s_i, mu_w_i) + K_o(k_i, s_i, mu_o_i))
    else:
        ret = T_j * (ro_o * C_o_j * (1.0 - Wps_j) + ro_p * Wps_j * C_p_j) * K_o(k_i, s_i, mu_w_i) / (K_w(k_j, s_j, mu_w_j) + K_o(k_j, s_j, mu_o_j))

    return ret


@ti.func
def up_t_kw(k_i: data_type, s_i: data_type, p_i: data_type, mu_o_i: data_type, mu_w_i: data_type, T_i: data_type, C_w_i: data_type,
		  k_j: data_type, s_j: data_type, p_j: data_type, mu_o_j: data_type, mu_w_j: data_type, T_j: data_type, C_w_j: data_type) -> data_type:
    """Значение берется вверх по потоку: up(kw / (ko + kw)"""
    ret = 0.0

    if p_i >= p_j:
        ret = T_i * C_w_i * K_w(k_i, s_i, mu_w_i) / (K_w(k_i, s_i, mu_w_i) + K_o(k_i, s_i, mu_o_i))
    else:
        ret = T_j * C_w_j * K_w(k_j, s_j, mu_w_j) / (K_w(k_j, s_j, mu_w_j) + K_o(k_j, s_j, mu_o_j))

    return ret


@ti.func
def K_o(k: data_type, s: data_type, mu_o: data_type) -> data_type:
    return k * pf_o(s) / mu_o


@ti.func
def K_w(k: data_type, s: data_type, mu_w: data_type) -> data_type:
    return k * pf_w(s) / mu_w
