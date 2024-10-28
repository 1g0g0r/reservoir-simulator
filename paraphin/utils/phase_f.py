from taichi import func

from paraphin.constants import S_min, S_max


@func
def pf_o(s):
    """
    Функция отностельной фазовой проницаемости нефти
        [(Smax-S)/(Smax-Smin)]^2
    """
    ret = 0.0
    if s < S_min:
        ret = 1.0  # однофазная фильтрация нефти
    elif s > S_max:
        ret = 0.0  # однофазная фильтрация воды
    else:
        ret = ((S_max - s) / (S_max-S_min)) ** 2

    return ret


@func
def pf_w(s):
    """
    Функция отностельной фазовой проницаемости воды
        [(S-Smin)/(Smax-Smin)]^2
    """
    ret = 0.0
    if s < S_min:
        ret = 0.0   # однофазная фильтрация нефти
    elif s > S_max:
        ret = 1.0    # однофазная фильтрация воды
    else:
        ret = ((s - S_min) / (S_max-S_min)) ** 2

    return ret
