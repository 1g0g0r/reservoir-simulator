from taichi import func, exp, f32


@func
def calc_mu_o(t: f32) -> f32:  # types.f32
    """Вязкость нефти, [Pa*c] Уравнение Аррениуса"""
    return 0.001 * exp(5000 / 8.314 / (t + 273.15))


@func
def calc_mu_w(t: f32) -> f32:
    """Вязкость воды, [Pa*c]  уравнение Андраде"""
    return 2.414 * 10 ** -5 * 10 ** (247.8 / (t + 133.15))


@func
def calc_c_w(t: f32) -> f32:
    """"Теплоемкость воды, [Дж/C]"""
    return 4217 - 2.15 * t + 0.002 * t ** 2


@func
def calc_c_o(t: f32) -> f32:
    """"Теплоемкость нефти, [Дж/C]"""
    return 1800 + 4 * t + 0.01 * t ** 2


@func
def calc_c_f(t: f32) -> f32:
    """"Теплоемкость пласта, [Дж/C]"""
    return 800 + 0.75 * t


@func
def calc_c_p(t: f32) -> f32:
    """"Теплоемкость парафина, [Дж/C]"""
    return 1840 + 3.56 * (t + 273.15)


# теплопроводности
# oil 0.13 + 0.0005(t-20)
# water 0.58 + 0.001(t-20)
# paraphin 0.25 + 0.0007(t-20)
# formation 2.5 + 0.0008(t-20)
