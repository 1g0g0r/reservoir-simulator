"""Решение уравнения концентрации взвешенных частиц парафина по явной схеме."""
from numba import njit

from paraphin import r1, r2, r3, r4, r5, r6, n_pass
from paraphin.constants import Nr, init_m, init_k, min_Wps_bound


@njit(cache=True)
def calc_qp_m_k_fi(i, j, Wps, m, k, fi, Ur, Ub, integr_r2_fi0, integr_r4_fi0, a_tdma, b_tdma,
                   new_qp1, new_qp2, new_fi, new_k, new_m, dt) -> None:
    """Скорости потери порового объема, пористость, проницаемость и функция пор по размерам.

    Пористость и проницаемость восстанавливаются по интегралам r^2*fi и r^4*fi, отнесенным к тем же
    интегралам от начального распределения (`integr_r2_fi0`, `integr_r4_fi0`). Сама `fi`
    обновляется прогонкой по сетке радиусов.

    Скорости потери порового объема разделены, потому что в балансы они входят по-разному:
    осаждение изымает чистый парафин, блокирование - смесь состава нефтяной фазы.

        q_p1 = -2*m * int r*ur*fi dr / int r^2*fi dr  >= 0   (осаждение на стенках, ur < 0)
        q_p2 =    m * int_0^r_pass r^2*ub dr / int r^2*fi dr >= 0   (блокирование каналов)

    Прежняя запись `m*(2*qp1 + Wps*qp2)/r2fi` складывала величины разных знаков (при ur < 0 первое
    слагаемое отрицательно) и несла лишний множитель Wps во втором: он уже входит в `ub`, а
    блокируется весь объем канала, а не только взвешенные в нем частицы.

    Сумма нормируется на фактическое изменение пористости: q_p1 + q_p2 = (m - m^new)/dt. Этим
    тождеством связаны (9), (11) и (12), и на него опираются уравнения насыщенности и парафина.

    a_tdma, b_tdma: numpy.ndarray(Nr)
        Прогоночные коэффициенты. Своя строка на каждый i: один общий буфер на все ячейки давал
        гонку в prange - потоки затирали друг другу коэффициенты, и fi считалась по мусору.

    Описание остальных аргументов - в докстринге пакета `paraphin.equations`.
    """
    if Wps[i, j] > min_Wps_bound:
        # Вычисление изменения пористости и проницаемости пласта
        int_r_ur_fi, int_r2_ub, r2fi, r4fi = _calculate_integrals(fi, Ur, Ub, i, j)
        new_m[i, j] = init_m * r2fi / integr_r2_fi0
        new_k[i, j] = init_k * r4fi / integr_r4_fi0

        qp1 = max(-2.0 * m[i, j] * int_r_ur_fi / r2fi, 0.0)
        qp2 = max(m[i, j] * int_r2_ub / r2fi, 0.0)

        # Нормировка на фактическую убыль пористости: dm/dt = -(q_p1 + q_p2) по построению (9)
        dm_dt = (m[i, j] - new_m[i, j]) / dt
        qp_sum = qp1 + qp2
        if qp_sum > 0.0 and dm_dt > 0.0:
            scale = dm_dt / qp_sum
            qp1 *= scale
            qp2 *= scale
        else:
            qp1, qp2 = 0.0, 0.0

        new_qp1[i, j] = qp1
        new_qp2[i, j] = qp2

        # Обновление функции пор по размерам
        _update_fi(new_fi, fi, Ur, Ub, i, j, a_tdma, b_tdma, dt)
    else:
        # Ниже порога кольматации поля не меняются: переносим текущие значения, чтобы new_*
        # оставались согласованы с текущим слоем (new_m читает `saturation_equation`).
        new_qp1[i, j] = 0.0
        new_qp2[i, j] = 0.0
        new_m[i, j] = m[i, j]
        new_k[i, j] = k[i, j]


@njit(cache=True)
def _calculate_integrals(fi, Ur, Ub, i: int, j: int):
    """Интегралы функции пор по размерам для ячейки (i, j).

    Подынтегральные функции восстанавливаются кусочно-линейно по узлам сетки радиусов, поэтому
    интегралы берутся точно по каждому отрезку.

    Returns
    -------
    int_r_ur_fi, int_r2_ub, r2fi, r4fi: float
        Интегралы r*ur*fi, r^2*ub, r^2*fi и r^4*fi
    """
    int_r_ur_fi, int_r2_ub, r2fi, r4fi = 0.0, 0.0, 0.0, 0.0

    for ij in range(1, Nr):
        dr = r1[ij] - r1[ij - 1]
        A_fi = (fi[i, j, ij - 1] * r1[ij] - fi[i, j, ij] * r1[ij - 1]) / dr
        B_fi = (fi[i, j, ij] - fi[i, j, ij - 1]) / dr
        A_ur = (Ur[i, j, ij - 1] * r1[ij] - Ur[i, j, ij] * r1[ij - 1]) / dr
        B_ur = (Ur[i, j, ij] - Ur[i, j, ij - 1]) / dr

        int_r_ur_fi += ((r2[ij] - r2[ij - 1]) * A_fi * A_ur / 2 +
                        (r3[ij] - r3[ij - 1]) * (A_fi * B_ur + B_fi * A_ur) / 3 +
                        (r4[ij] - r4[ij - 1]) * B_fi * B_ur / 4)  # r * ur * fi
        r2fi += (r3[ij] - r3[ij - 1]) * A_fi / 3 + (r4[ij] - r4[ij - 1]) * B_fi / 4  # r^2 * fi
        r4fi += (r5[ij] - r5[ij - 1]) * A_fi / 5 + (r6[ij] - r6[ij - 1]) * B_fi / 6  # r^4 * fi

    # Ub отлична от нуля только при r < r_pass (частица не проходит горло), поэтому интеграл
    # r^2*ub идет отдельным коротким циклом, а не проверкой радиуса на каждом узле общего.
    # `Ub` хранит коэффициент b(r), сама скорость блокирования ub = b(r)*fi.
    for ij in range(1, n_pass):
        dr = r1[ij] - r1[ij - 1]
        ub_l = Ub[i, j, ij - 1] * fi[i, j, ij - 1]
        ub_r = Ub[i, j, ij] * fi[i, j, ij]
        A_ub = (ub_l * r1[ij] - ub_r * r1[ij - 1]) / dr
        B_ub = (ub_r - ub_l) / dr
        int_r2_ub += (r3[ij] - r3[ij - 1]) * A_ub / 3 + (r4[ij] - r4[ij - 1]) * B_ub / 4  # r^2 * ub

    return int_r_ur_fi, int_r2_ub, r2fi, r4fi


@njit(cache=True)
def _update_fi(new_fi, fi, Ur, Ub, i: int, j: int, a_tdma, b_tdma, dt):
    """Обновление функции пор по размерам по неявной схеме методом прогонки.

    Поток через грань ij+1/2 расщеплен по знаку скорости:
    F = max(u_ij, 0)*fi_ij + min(u_ij+1, 0)*fi_ij+1. В области одного знака u это обычная
    противопоточная схема; на стыке r_pass (слева u = 0, справа u < 0) она, в отличие от
    ветвления по знаку u_ij, не теряет поток из узла n_pass в узел n_pass-1: сузившиеся до r_pass
    капилляры попадают под блокирование, а не исчезают из баланса.

    Слагаемое блокирования ub = b(r)*fi берется неявно: `Ub` хранит коэффициент b(r), и он уходит
    на диагональ `d`, а не в правую часть. При явной записи большое b*dt уводит `fi` в
    отрицательные значения - безусловная устойчивость прогонки положительности не гарантирует.
    Матрица - M-матрица (диагональ 1/dt + |u|/dr + b, внедиагональные <= 0), поэтому прогонка
    устойчива и положительность fi сохраняется без зажима.

    На правой границе (r = r_max) соседа нет, что равносильно условию fi = 0: капилляров шире
    r_max нет, а те, что на r_max, сужаются внутрь и ничем не замещаются - поэтому правый узел
    проседает первым. Это свойство модели, а не схемы: на сетке в 100 раз мельче падение то же.

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    dr = r1[1] - r1[0]

    # Вычисление прогоночных коэффициентов
    d = 1.0 / dt + abs(Ur[i, j, 0]) / dr + Ub[i, j, 0]
    e = min(Ur[i, j, 1], 0.0) / dr
    a_tdma[0] = -e / d
    b_tdma[0] = fi[i, j, 0] / dt / d

    for ij in range(1, Nr):
        c = -max(Ur[i, j, ij - 1], 0.0) / dr
        d = 1.0 / dt + abs(Ur[i, j, ij]) / dr + Ub[i, j, ij]
        e = min(Ur[i, j, ij + 1], 0.0) / dr if ij + 1 < Nr else 0.0  # за Nr-1 соседа нет
        denominator = c * a_tdma[ij - 1] + d
        a_tdma[ij] = -e / denominator
        b_tdma[ij] = (fi[i, j, ij] / dt - c * b_tdma[ij - 1]) / denominator

    # Вычисление функции пор размерам
    new_fi[i, j, Nr - 1] = b_tdma[Nr - 1]
    for ij in range(Nr - 2, -1, -1):  # обратный ход
        new_fi[i, j, ij] = new_fi[i, j, ij + 1] * a_tdma[ij] + b_tdma[ij]
