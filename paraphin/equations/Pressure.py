"""Решение уравнения давления: сборка матрицы (МКО) и решение СЛАУ (PCG с многосеточным предобуславливателем)."""
from numba import njit, prange

from paraphin.constants import Nx, Ny
from paraphin.layout import P0, PX
from paraphin.utils import apply_bc, get_bound, calc_well_prod, mid, mobility_o, mobility_w, DI, DJ, HIJ, AREA
from paraphin.utils.math_utils import solve_mg_system, project_guess


@njit(cache=True)
def calc_pressure(k, S, mu_o, mu_w, lam_o, lam_w, wells, diag, ex, ey, rhs, mg_buf, mg_wc, p_state, p_vec, p_hist,
                  pcg_p, pcg_q, boundary_condition, p):
    """Сборка матрицы и решение СЛАУ уравнения давления (МКО). Возвращает число итераций PCG (0 - начальное
    приближение уже в пределах `p_pcg_rtol`).

    Начальное приближение - проекция на разности прошлых решений (`utils/math_utils/guess.py`): итераций 0.5-0.8 вместо
    1.7 у квадратичной экстраполяции. Точность от этого не зависит - ее держит `p_pcg_rtol`.

    Матрица собирается не в CSC, а сразу в три диагонали положительно определенной формы `M = -A`
    (`utils/math_utils/mg_solver.py`). Каждая ячейка пишет в свои `diag[idx]`, `ex[idx]`, `ey[idx]`, поэтому сборка
    параллельна и согласовывать порядок записи не нужно. Векторы - в раскладке `layout` с рамкой фиктивных ячеек:
    ячейка (i, j) - индекс P0 + i + j*PX, а `diag`, `ex`, `ey` - строки мелкого уровня `mg_buf` (`Solver.__init__`),
    так что матрица собирается прямо в уровень многосеточного решателя.

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
    wells: numpy.ndarray(n_wells), dtype `WELL`
        Скважины - структурный массив (`utils/well.py`)
    diag, ex, ey: numpy.ndarray(NP)
        Диагонали матрицы: центр, связь с idx+1 (сосед по x), связь с idx+PX (сосед по y)
    rhs: numpy.ndarray(NP)
        Правая часть в форме M x = rhs, то есть с обратным знаком к исходной
    mg_buf, mg_wc: numpy.ndarray(MG_ROWS, MG_TOTAL), numpy.ndarray(MG_COARSE, MG_COARSE_KD+1)
        Уровни многосеточного решателя (мелкий - сама матрица, невязки PCG - его строки) и фактор самого грубого
    p_state: numpy.ndarray(P_STATE)
        Состояние решателя: возраст грубых уровней, сумма итераций с пересборки, флаг пересборки, голова кольца `p_hist`
    p_vec: numpy.ndarray(NP)
        Решение; на входе - давление с прошлого шага
    p_hist: numpy.ndarray(p_guess_m, NP)
        Кольцо прошлых решений до прошлого шага (`guess.py`); после вызова сдвинуто на шаг
    pcg_p, pcg_q: numpy.ndarray(NP)
        Рабочие векторы PCG: направление поиска и M*p
    boundary_condition: numpy.ndarray(4, 3, 2)
        Граничные условия: Граница -> Поле -> Тип, Значение
    p: numpy.ndarray(Nx, Ny)
        Поле давления - результат; заполняется на месте, [Па]
    """
    _fill_matrix_and_rhs(k, S, mu_o, mu_w, lam_o, lam_w,
                         diag, ex, ey, rhs, boundary_condition)
    _adding_wells(wells, S, k, mu_o, mu_w, lam_o, lam_w, diag, rhs)

    project_guess(diag, ex, ey, rhs, p_vec, p_hist, p_state)
    it = solve_mg_system(rhs, p_vec, pcg_p, pcg_q, mg_buf, mg_wc, p_state)

    # Неизвестная нумеруется как idx = P0 + i + j*PX (быстрый индекс - i), поэтому раскладка идет по этой же формуле.
    # Без нее поле давления оказывается зеркальным относительно главной диагонали (транспонировалось).
    for i in range(Nx):
        for j in range(Ny):
            p[i, j] = p_vec[P0 + i + j * PX]

    return it


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
            idx = P0 + i + j * PX
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
def _adding_wells(wells, S, k, mu_o, mu_w, lam_o, lam_w, diag, rhs):
    """Учет скважин в уравнении давления, неявный по давлению.

        q = J*(P_забой - P_ячейки)

    Слагаемое с давлением ячейки уходит на диагональ, с забойным - в правую часть. Явные `q^t`
    здесь стоять не могут: задача несжимаемая с непроницаемыми границами, то есть чисто нейманнова,
    матрица вырождена, и решение существует лишь при нулевой сумме дебитов - для `q(P^t)` это не выполняется.
    При неявной записи равенство суммарных дебитов выполняется тождественно.

    Коэффициенты продуктивности при этом не пересчитываются здесь, а считаются `calc_well_pi` и остаются на скважине:
    после решения СЛАУ `upd_q_and_eta` умножает те же самые `J` на перепад.
    """
    for w in range(wells.shape[0]):
        calc_well_prod(wells, w, S, k, mu_o, mu_w, lam_o, lam_w)
        wl = wells[w]
        idx = wl.idx_rhs

        if wl.rate_control:
            rhs[idx] += wl.q[2]
        else:
            diag[idx] += wl.J[2]
            rhs[idx] += wl.J[2] * wl.p
