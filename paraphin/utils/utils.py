import taichi as ti

from paraphin.utils.phase_f import pf_o, pf_w
from paraphin.constants import default_type


@ti.func
def mid(k1: default_type, s1: default_type, mu_o1: default_type, mu_w1: default_type,
        k2: default_type, s2: default_type, mu_o2: default_type, mu_w2: default_type) -> default_type:
    """mid(Ko + Kw)_ij"""
    x = K_o(k1, s1, mu_o1) + K_w(k1, s1, mu_w1)
    y = K_o(k2, s2, mu_o2) + K_w(k2, s2, mu_w2)
    return 2.0 * x * y / (x + y)


@ti.func
def up_kw(k1: default_type, s1: default_type, p1: default_type, mu_o1: default_type, mu_w1: default_type,
          k2: default_type, s2: default_type, p2: default_type, mu_o2: default_type, mu_w2: default_type) -> default_type:
    """up(kw / (ko + kw)"""
    ret = 0.0

    if p1 >= p2:
        ret = K_w(k1, s1, mu_w1) / (K_w(k1, s1, mu_w1) + K_o(k1, s1, mu_o1))
    else:
        ret = K_w(k2, s2, mu_w2) / (K_w(k2, s2, mu_w2) + K_o(k2, s2, mu_o2))

    return ret


@ti.func
def up_ko(k1: default_type, s1: default_type, p1: default_type, mu_o1: default_type, mu_w1: default_type,
          k2: default_type, s2: default_type, p2: default_type, mu_o2: default_type, mu_w2: default_type) -> default_type:
    """up(ko / (ko + kw)"""
    ret = 0.0

    if p1 >= p2:
        ret = K_o(k1, s1, mu_o1) / (K_w(k1, s1, mu_w1) + K_o(k1, s1, mu_o1))
    else:
        ret = K_o(k2, s2, mu_o2) / (K_w(k2, s2, mu_w2) + K_o(k2, s2, mu_o2))

    return ret


@ti.func
def K_o(k: default_type, s: default_type, mu_o: default_type) -> default_type:
    return k * pf_o(s) / mu_o


@ti.func
def K_w(k: default_type, s: default_type, mu_w: default_type) -> default_type:
    return k * pf_w(s) / mu_w
