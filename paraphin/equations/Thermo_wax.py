"""Равновесие твердое-жидкость групп парафина: multi-solid, давление (Пойнтинг) и растворенный газ.

Каждая группа н-алканов, выпадая, образует свою чистую твердую фазу (multi-solid: Lira-Galeana,
Firoozabadi & Prausnitz, AIChE J 1996, 42:239; Pan, Firoozabadi & Fotland, SPE PF 1997, 12:250).
Для чистой твердой фазы активность в твердом равна единице, и равновесие задает предел растворимости -
мольную долю группы в насыщенном растворе:

    ln x_k^sat = -(dH_k/R)*(1/T - 1/Tm_k) - dv_k*(P - P_ref)/(R*T),             (1)

где второе слагаемое - поправка Пойнтинга: твердая фаза плотнее жидкой (dv = v_L - v_S > 0), поэтому
давление понижает растворимость и повышает WAT. Продифференцировав (1) при постоянном составе,
получаем уравнение Клапейрона-Клаузиуса для точки начала кристаллизации: dWAT/dP = WAT*dv/dH.

Раствор - все, что не выпало: растворитель (молярная масса M_o), растворенный газ n_g(P) и растворенные
группы. Мольная доля группы x_k = n_k/n_L, где n_L - число молей раствора на грамм нефти:

    n_L = n_0 + sum_k w_k^dis/M_k,   n_0 = (1 - sum_k w_k)/M_o + n_g(P).

Группа насыщена, если w_k/M_k > x_k^sat*n_L, и тогда w_k^dis = x_k^sat*n_L*M_k; иначе растворена целиком.
Для множества насыщенных групп S решение точное: n_L = (n_0 + sum_{k не в S} w_k/M_k)/(1 - sum_{k в S} x_k^sat).
Насыщение группы только уменьшает n_L, поэтому S набирается жадно по убыванию порога
t_k = w_k/(M_k*x_k^sat): не более N_w шагов, без итераций и без выделения памяти. При одной группе это в
точности прежняя формула (6.1)-(6.2) `Wp_balance._wp_saturated`.

Растворенный газ добавляет моли в раствор и тем понижает WAT (эффект состава у Pan et al., 1997);
газосодержание линейно по давлению до давления насыщения P_b. Свободный газ в течении не отслеживается.

С флагом `wax_eos` x_k^sat и газосодержание берутся из таблиц уравнения состояния (`thermo/tables.py`): та же
форма раствора, но неидеальность - из фугитивностей Пенга-Робинсона (multi-solid Lira-Galeana), а газ - из flash.
Замкнутой формулы WAT тогда нет, и `wat_cell` ищет ее бисекцией.
"""
import math

from numba import njit, prange

from paraphin.constants import M_o, Nx, Ny, P_ref_wax, P_bubble, wax_pressure, wax_pore_shift, wax_eos
from paraphin.layout import N_W
from paraphin.oil_composition import WAX_M, WAX_TM_K, WAX_DH_R, WAX_DV_R, WAX_L_REL, N_GAS_B
from paraphin.thermo.tables import LNXSAT, NG, T_LO, T_HI, eos_interp


@njit(cache=True)
def n_gas(T, p):
    """Растворенный газ на грамм дегазированной нефти, [моль/г]: линейно по давлению до P_b или, с `wax_eos`,
    из flash уравнения состояния. T в C, p в Па."""
    if wax_pressure:
        if wax_eos:
            return eos_interp(NG, T, p)
        return N_GAS_B * min(max(p, 0.0) / P_bubble, 1.0)
    return 0.0


@njit(cache=True)
def x_saturation(k, T, p):
    """Мольная доля насыщения группы k по (1), [-]. T в C, p в Па. Выше температуры плавления - 1."""
    if wax_eos:
        return min(1.0, math.exp(eos_interp(LNXSAT[k], T - wax_pore_shift, p)))
    t_abs = T + 273.15 - wax_pore_shift
    arg = -WAX_DH_R[k] * (1.0 / t_abs - 1.0 / WAX_TM_K[k])
    if wax_pressure:
        arg -= WAX_DV_R[k] * (p - P_ref_wax) / t_abs
    if arg >= 0.0:
        return 1.0
    return math.exp(arg)


@njit(cache=True)
def sle_split(wax, T, p, sus):
    """Разделение групп парафина на растворенные и взвешенные при температуре T и давлении p.

    Parameters
    ----------
    wax: numpy.ndarray(N_w)
        Суммарные (растворенные + взвешенные) массовые доли групп в нефтяной фазе, [-]
    T, p: float
        Температура [C] и давление [Па]
    sus: numpy.ndarray(N_w)
        Результат: взвешенные доли групп. На время расчета в нем хранятся x_k^sat - отдельного буфера
        не нужно, и в parfor ничего не выделяется

    Returns
    -------
    w_dis, w_sus, hl: float
        Суммарные растворенная и взвешенная доли и носитель скрытой теплоты
        sum_k (L_k/latent_heat)*w_k^dis - растворенный парафин в единицах прежней удельной теплоты
    """
    w_sum = 0.0
    a = n_gas(T, p)
    for k in range(N_W):
        w_sum += wax[k]
        a += wax[k] / WAX_M[k]
        sus[k] = x_saturation(k, T, p)
    a += (1.0 - w_sum) / M_o
    b = 1.0

    # Жадный набор насыщенных групп по убыванию порога t_k: насыщение группы уменьшает n_L = a/b,
    # поэтому раз отвергнутый порог отвергается и дальше
    mask = 0
    for _ in range(N_W):
        best = -1
        t_best = -1.0
        for k in range(N_W):
            if (mask >> k) & 1 or wax[k] <= 0.0 or sus[k] >= 1.0:
                continue
            t_k = wax[k] / (WAX_M[k] * sus[k]) if sus[k] > 0.0 else math.inf
            if t_k > t_best:
                t_best = t_k
                best = k
        if best < 0 or t_best * b <= a:
            break
        mask |= 1 << best
        a -= wax[best] / WAX_M[best]
        b -= sus[best]

    n_l = a / b
    w_dis, w_sus, hl = 0.0, 0.0, 0.0
    for k in range(N_W):
        if (mask >> k) & 1:
            dis = sus[k] * n_l * WAX_M[k]
        else:
            dis = wax[k]
        sus[k] = wax[k] - dis
        w_dis += dis
        w_sus += sus[k]
        hl += WAX_L_REL[k] * dis

    return w_dis, w_sus, hl


@njit(cache=True)
def sle_hl_boundary(wax, T, p):
    """Носитель скрытой теплоты втекающей через границу нефти заданного состава `wax` (ГУ Дирихле).

    Нужен только на гранях с ГУ `DataField.Paraffin`, поэтому буфер выделяется здесь, а не передается."""
    sus = wax.copy()
    return sle_split(wax, T, p, sus)[2]


@njit(cache=True)
def wat_cell(wax, p):
    """Температура начала кристаллизации нефти с составом `wax` при давлении p, [C].

    Пока все растворено, x_k = (w_k/M_k)/n_L. Первой насыщается группа с наибольшей температурой из (1):
        T_k = (dH_k/R + dv_k/R*(P - P_ref)) / (dH_k/(R*Tm_k) - ln x_k).
    С `wax_eos` - бисекция по T в пределах сетки таблиц.
    """
    if wax_eos:
        return _wat_eos(wax, p)
    n_l = n_gas(0.0, p)
    w_sum = 0.0
    for k in range(N_W):
        n_l += wax[k] / WAX_M[k]
        w_sum += wax[k]
    n_l += (1.0 - w_sum) / M_o

    t_max = -273.15
    for k in range(N_W):
        if wax[k] <= 0.0:
            continue
        num = WAX_DH_R[k]
        if wax_pressure:
            num += WAX_DV_R[k] * (p - P_ref_wax)
        t_k = num / (WAX_DH_R[k] / WAX_TM_K[k] - math.log(wax[k] / WAX_M[k] / n_l))
        t_max = max(t_max, t_k)

    return t_max - 273.15 + wax_pore_shift


@njit(cache=True)
def _saturated(wax, T, p):
    """Насыщена ли хоть одна группа при температуре T [C]: w_k/M_k > x_k^sat*n_L при всем растворенном."""
    n_l = n_gas(T, p)
    w_sum = 0.0
    for k in range(N_W):
        n_l += wax[k] / WAX_M[k]
        w_sum += wax[k]
    n_l += (1.0 - w_sum) / M_o
    for k in range(N_W):
        if wax[k] > WAX_M[k] * x_saturation(k, T, p) * n_l:
            return True
    return False


@njit(cache=True)
def _wat_eos(wax, p):
    """WAT по таблицам уравнения состояния - бисекцией на [T_LO, T_HI], за сеткой - ее край, [C]."""
    lo, hi = T_LO, T_HI
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _saturated(wax, mid, p):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@njit(parallel=True, cache=True)
def calc_wat_field(Wc, p, WAT):
    """Поле WAT по текущему составу и давлению - только для выгрузки, в расчет не входит."""
    for i in prange(Nx):
        for j in range(Ny):
            WAT[i, j] = wat_cell(Wc[i, j, :N_W], p[i, j])
