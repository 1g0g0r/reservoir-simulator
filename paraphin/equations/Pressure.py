"""Решение уравнения давления: сборка матрицы (МКО) и решение ленточной СЛАУ."""
from numba import njit, prange

from paraphin.constants import Nx, Ny
from paraphin.utils import (apply_bc, get_bound, calc_well_prod, mid, solve_band_system,
                            mobility_o, mobility_w, DI, DJ, HIJ, AREA)


@njit(cache=True)
def calc_pressure(k, S, mu_o, mu_w, lam_o, lam_w, wells,
                  diag, ex, ey, rhs, band_w, p_vec, pcg_r, pcg_z, pcg_p, pcg_q,
                  boundary_condition, band_age, p):
    """Сборка матрицы и решение СЛАУ уравнения давления (МКО).

    Матрица собирается не в CSC, а сразу в три диагонали положительно определенной формы
    `M = -A` (см. `utils/math_utils/band_solver.py`): пятиточечный шаблон при нумерации
    `idx = i + j*Nx` дает ленту с полушириной Nx, для которой разложение Холецкого на порядок
    дешевле SuperLU. Инвариант «порядок записи в препроцессинге и в сборке должен совпадать»
    при этом исчезает: каждая ячейка пишет в свои `diag[idx]`, `ex[idx]`, `ey[idx]`.

    Parameters
    ----------
    k: numpy.ndarray(Nx, Ny)
        Проницаемость, [м^2]
    S: numpy.ndarray(Nx, Ny)
        Водонасыщенность, [-]
    mu_o, mu_w: numpy.ndarray(Nx, Ny)
        Вязкости нефти и воды, [Па*с]
    lam_o, lam_w: numpy.ndarray(Nx, Ny)
        Подвижности фаз k*pf/mu, посчитанные `calc_mobility` до вызова, [м^2/(Па*с)]
    wells: numpy.ndarray(n_wells)
        Массив скважин
    diag, ex, ey: numpy.ndarray(Nx*Ny)
        Диагонали матрицы: центр, связь с idx+1 (сосед по x), связь с idx+Nx (сосед по y)
    rhs: numpy.ndarray(Nx*Ny)
        Правая часть в форме M x = rhs, то есть с обратным знаком к исходной
    band_w: numpy.ndarray(Nx*Ny, Nx+1)
        Буфер фактора Холецкого, живет между шагами
    p_vec: numpy.ndarray(Nx*Ny)
        Решение; на входе - давление с прошлого шага, оно же начальное приближение для PCG
    pcg_r, pcg_z, pcg_p, pcg_q: numpy.ndarray(Nx*Ny)
        Рабочие векторы PCG
    boundary_condition: numpy.ndarray(4, 3, 2)
        Граничные условия: Граница -> Поле -> Тип, Значение
    band_age: int
        Возраст фактора Холецкого в шагах
    p: numpy.ndarray(Nx, Ny)
        Поле давления - результат; заполняется на месте, [Па]
    """
    _fill_matrix_and_rhs(k, S, mu_o, mu_w, lam_o, lam_w,
                         diag, ex, ey, rhs, boundary_condition)
    _adding_wells(wells, S, k, mu_o, mu_w, diag, rhs)

    band_age = solve_band_system(diag, ex, ey, rhs, band_w, p_vec, pcg_r, pcg_z, pcg_p, pcg_q, band_age)

    # Неизвестная нумеруется как idx = i + j*Nx (быстрый индекс - i), поэтому раскладка идет по этой же формуле.
    # Без нее поле давления оказывается зеркальным относительно главной диагонали (транспонировалось).
    for i in range(Nx):
        for j in range(Ny):
            p[i, j] = p_vec[i + j * Nx]

    return band_age


@njit(parallel=True, cache=True)
def _fill_matrix_and_rhs(k, S, mu_o, mu_w, lam_o, lam_w,
                         diag, ex, ey, rhs, boundary_conditions):
    """Заполнение диагоналей матрицы и правой части уравнения давления.

    Собирается форма `M = -A`: диагональ положительна, внедиагональные элементы отрицательны,
    правая часть - с обратным знаком. Ячейка владеет гранями «вправо» и «вверх», поэтому записи
    независимы и цикл распараллеливается.

    В правой части стоят только дебиты (их добавляет `_adding_wells`) и вклад условий Дирихле.
    Уравнение давления получается сложением балансов *фаз* (воды и нефтяной фазы целиком), при
    котором производные по времени сокращаются: sum_j T_ij*(P_i - P_j) = q_w + q_o. Прежний
    источник -((m - m_0) + m_0*S_0*(Wo - Wo_0)/Wo)/dt*volume возникал из-за сложения баланса воды
    с балансом отдельного компонента (2) вместо баланса фазы и давал невязку суммарных дебитов
    нагнетательной и добывающей скважин.
    """
    for i in prange(Nx):
        for j in range(Ny):
            idx = i + j * Nx
            lam_ij = lam_o[i, j] + lam_w[i, j]

            acc = 0.0  # только вклад Дирихле: производные по времени сократились при сложении фаз
            dg = 0.0

            for qq in range(4):
                i1 = i + DI[qq]
                j1 = j + DJ[qq]
                hij = HIJ[qq]
                areaij = AREA[qq]

                if (0 <= i1 < Nx) and (0 <= j1 < Ny):
                    val = mid(lam_ij, lam_o[i1, j1] + lam_w[i1, j1]) * areaij / hij
                    dg += val
                    if qq == 0:
                        ex[idx] = -val
                    elif qq == 2:
                        ey[idx] = -val
                else:
                    bound = get_bound(i1, j1)
                    hij *= 0.5
                    S_ij = apply_bc(boundary_conditions, bound, 1, S, i, j, hij)
                    lam_gh = mobility_o(k[i, j], S_ij, mu_o[i, j]) + mobility_w(k[i, j], S_ij, mu_w[i, j])
                    val = mid(lam_ij, lam_gh) * areaij / hij
                    if boundary_conditions[bound, 0, 0] == 1:  # Дирихле: поток val*(P_гр - P_ячейки)
                        acc += boundary_conditions[bound, 0, 1] * val
                        dg += val
                    else:  # Нейман: фиктивная ячейка P + g*h (`apply_bc`), поток g*h*val от P не зависит
                        acc += boundary_conditions[bound, 0, 1] * hij * val

            # Правая грань последнего столбца и верхняя грань последней строки связей не дают
            if i == Nx - 1:
                ex[idx] = 0.0
            if j == Ny - 1:
                ey[idx] = 0.0

            diag[idx] = dg
            rhs[idx] = acc


@njit(cache=True)
def _adding_wells(wells, S, k, mu_o, mu_w, diag, rhs):
    """Учет скважин в уравнении давления, неявный по давлению.

        q = prod*(P_забой - P_ячейки)

    Слагаемое с давлением ячейки уходит на диагональ, с забойным - в правую часть. Явные `q^t`
    здесь стоять не могут: задача несжимаемая с непроницаемыми границами, то есть чисто нейманнова,
    матрица вырождена, и решение существует лишь при нулевой сумме дебитов - для `q(P^t)` это не выполняется.
    При неявной записи равенство суммарных дебитов выполняется тождественно.

    Коэффициенты продуктивности при этом не пересчитываются здесь, а считаются `calc_well_pi` и остаются на скважине:
    после решения СЛАУ `upd_q_and_eta` умножает те же самые `prod` на перепад.
    """
    for i in range(len(wells)):
        wells[i] = calc_well_prod(wells[i], S, k, mu_o, mu_w)
        well = wells[i]

        if well.rate_control == 1:
            rhs[well.idx_rhs] += well.q[2]
        else:
            diag[well.idx_rhs] += well.prod[2]
            rhs[well.idx_rhs] += well.prod[2] * well.p
