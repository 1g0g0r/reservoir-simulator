"""Проверки потерь тепла через кровлю и подошву пласта.

Член `_heat_losses_lauwerier` - это обмен пласта с окружающими породами из схемы Ловерье,
Theta = -2*K_ff*dT/dz на границах z = +-h/2, посчитанный в приближении ступеньки: температура
ячейки считается постоянной с начала расчета. Проверок две.

1. `test_heat_loss_energy_balance` - сколько тепла ушло за время t. Интеграл потока по времени
   обязан совпасть с теплом, которое поглотили два полубесконечных массива пород с заданным на
   границе перегревом. Ловит потерянную двойку (кровля без подошвы), h не в том месте и чужую
   температуропроводность.

2. `test_lauwerier_solution_residual` - тот же коэффициент, но со стороны самой модели Ловерье.
   Ее аналитическое решение подставляется в уравнение энергии пласта, а поток в породы считается
   точной сверткой по истории температуры. Невязка обязана быть нулевой - это и есть доказательство,
   что множитель 2*K_ff/(h*sqrt(pi*alpha)) в коде тот самый, которому удовлетворяет решение Ловерье,
   а приближение ступеньки отличается от него только заменой свертки на отклик на ступеньку.

3. `test_vinsome_matches_step_response` - рекурсия Винсома-Вестервельда, прогнанная на постоянной
   температуре границы, обязана лечь на тот же отклик. Проверяет метод целиком: формулу для p,
   обновление состояния E_ff (деление на объемную теплоемкость пород) и множитель 2/h.
"""
import numpy as np

from paraphin.constants import K_ff, c_ff, ro_ff, c_f, ro_f, h, init_T
from paraphin.equations.Temperature import (_heat_losses_vw, _heat_losses_lauwerier,
                                            alpha_ff)
from paraphin.utils.math_utils import erfc

M_ff = c_ff * ro_ff  # объемная теплоемкость окружающих пород, [Дж/(м^3*C)]
M_f = c_f * ro_f     # объемная теплоемкость пласта, [Дж/(м^3*C)]


def test_heat_loss_energy_balance():
    """Интеграл потерь по времени = тепло, поглощенное кровлей и подошвой."""
    dT = -50.0  # перегрев ячейки относительно невозмущенных пород (закачка холодной воды)
    t_end = 365.0 * 86400.0

    T = np.full((1, 1), init_T + dT)
    # Замена t = s^2 снимает особенность 1/sqrt(t) в нуле, иначе трапеции мажут на начале
    s_grid = np.linspace(0.0, np.sqrt(t_end), 20_001)[1:]
    q = np.array([_heat_losses_lauwerier(0, 0, s * s, T) for s in s_grid])
    accumulated = np.trapezoid(q * 2.0 * s_grid, s_grid)  # [Дж/м^3 пласта]

    # Полубесконечный массив с перегревом dT на границе поглощает 2*K*dT*sqrt(t/(pi*alpha))
    # с единицы площади; двойка снаружи - кровля и подошва, деление на h - на объем пласта.
    exact = 2.0 * (2.0 * K_ff * dT * np.sqrt(t_end / (np.pi * alpha_ff))) / h

    err = abs(accumulated - exact) / abs(exact)
    assert err < 1e-3, f'накопленные потери разошлись с точным решением: {err:.2e}'


def _lauwerier(x, t, U, dT):
    """Решение Ловерье: температура пласта при закачке с перегревом dT в точке x = 0.

    T = T0 + dT*erfc[ (x/(h*U))*sqrt(K_ff*M_ff) / sqrt(t - x*M_f/U) ], до прихода фронта T = T0.
    Тепловой фронт приходит в точку x в момент x*M_f/U - конвективный перенос запаздывает на
    теплоемкость скелета. U = ro*c*V - конвективный поток теплоемкости, [Вт/(м^2*C)].
    """
    lag = x * M_f / U
    if t <= lag:
        return init_T

    return init_T + dT * erfc(x / (h * U) * np.sqrt(K_ff * M_ff) / np.sqrt(t - lag))


def test_lauwerier_solution_residual():
    """Решение Ловерье обязано обращать уравнение энергии пласта в ноль с этим потоком в породы."""
    U, dT, x = 5.0, -50.0, 20.0
    t = 4.0 * x * M_f / U  # заметно позже прихода фронта, чтобы производные были гладкими

    def T_of(xx, tt):
        return _lauwerier(xx, tt, U, dT)

    # Производные - центральными разностями по аналитическому решению, без сеточной диффузии
    dx, dt = 1e-4 * x, 1e-4 * t
    dT_dt = (T_of(x, t + dt) - T_of(x, t - dt)) / (2.0 * dt)
    dT_dx = (T_of(x + dx, t) - T_of(x - dx, t)) / (2.0 * dx)

    # Точный поток в породы: свертка истории температуры с ядром 1/sqrt(t - tau).
    # Замена tau = t - s^2 убирает особенность подынтегральной функции в tau = t.
    s = np.linspace(0.0, np.sqrt(t), 200_001)
    dT_dtau = np.array([(T_of(x, tau + dt) - T_of(x, tau - dt)) / (2.0 * dt)
                        for tau in t - s * s])
    q_loss = 2.0 * K_ff / (h * np.sqrt(np.pi * alpha_ff)) * np.trapezoid(2.0 * dT_dtau, s)

    # M_f*dT/dt + U*dT/dx + q_loss = 0
    residual = M_f * dT_dt + U * dT_dx + q_loss
    scale = abs(U * dT_dx)
    assert abs(residual) / scale < 2e-2, f'решение Ловерье не удовлетворяет балансу: {residual:.4g}'


def test_vinsome_matches_step_response():
    """Винсом-Вестервельд на ступеньке = отклик полубесконечного массива.

    Автомодельный режим рекурсии дает E_ff = (24/11)*d*dT и поток на 3.3% ниже точного - это
    известная точность пробной функции, а не ошибка. Допуск взят с запасом до 6%.
    """
    dT, t_end, n_steps = -50.0, 365.0 * 86400.0, 20_000
    dt = t_end / n_steps

    T = np.full((1, 1), init_T + dT)
    T_0 = T.copy()          # ступенька: граница держит постоянную температуру с t = 0
    E_ff = np.zeros((1, 1))

    t, q = 0.0, 0.0
    for _ in range(n_steps):
        t += dt
        q = _heat_losses_vw(0, 0, t, T, T_0, E_ff, dt)

    q_exact = 2.0 * K_ff * dT / (h * np.sqrt(np.pi * alpha_ff * t_end))
    err = abs(q - q_exact) / abs(q_exact)
    assert err < 0.06, f'Винсом-Вестервельд разошелся с откликом на ступеньку: {err:.1%}'

    # Состояние обязано выйти на автомодельное E = (24/11)*d*dT
    d = 0.5 * np.sqrt(alpha_ff * t_end)
    assert abs(E_ff[0, 0] - 24.0 / 11.0 * d * dT) / abs(24.0 / 11.0 * d * dT) < 0.02


if __name__ == '__main__':
    test_heat_loss_energy_balance()
    test_lauwerier_solution_residual()
    test_vinsome_matches_step_response()
    print('OK')
