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
точности однокомпонентная формула (6.1)-(6.2) `single_wp_saturated`.

Растворенный газ добавляет моли в раствор и тем понижает WAT (эффект состава у Pan et al., 1997);
газосодержание линейно по давлению до давления насыщения P_b. Свободный газ в течении не отслеживается.

С флагом `wax_eos` x_k^sat и газосодержание берутся из таблиц уравнения состояния (`thermo/tables.py`): та же
форма раствора, но неидеальность - из фугитивностей Пенга-Робинсона (multi-solid Lira-Galeana), а газ - из flash.
Замкнутой формулы WAT тогда нет, и `wat_cell` ищет ее бисекцией.

С флагом `wax_solid_solution` группы выпадают в один твердый раствор (предсказательный UNIQUAC Coutinho, как
`thermo.solids.UniquacSolidWax` с идеальной жидкостью): x_k^L = x_k^S gamma_k^S(x^S, T) x_k^0(T, P), где x_k^0 - (1)
со свойствами плавления группы по Coutinho (с твердо-твердым переходом) и физическим скачком объема. Насыщение
группы зависит от состава твердого, поэтому таблиц по (T, P) нет: flash (Рачфорд-Райс по доле твердого и простые
итерации по его составу) решается в ячейке, `ss_split`. Старт состава твердого - всегда идеальный твердый раствор,
как у эталона: решений flash с одной твердой фазой бывает несколько (UNIQUAC предсказывает и расслоение твердого), и
старт со взвеси прошлого слоя давал бы ответ, зависящий от истории ячейки.
"""
import math

import numpy as np
from numba import njit, prange

from paraphin.constants import (M_o, Nx, Ny, P_ref_wax, P_bubble, wax_pressure, wax_pore_shift, wax_eos, MW, Tm, Tm_K,
                                alpha, R, wax_solid_solution, ro_wax_liq, ro_p)
from paraphin.layout import N_W
from paraphin.oil_composition import WAX_M, WAX_TM_K, WAX_DH_R, WAX_DV_R, WAX_L_REL, N_GAS_B
from paraphin.thermo.characterization import coutinho_nalkane, nalkane_dh_vap
from paraphin.thermo.tables import LNXSAT, NG, T_LO, T_HI, eos_interp

# Твердый раствор (`wax_solid_solution`): группа - н-алкан средней длины цепи; UNIQUAC Coutinho: r, q по n, энергия
# пары - как у более короткой молекулы, lambda_i = -(2/Z)(dH_sub,i - RT) (RT в разностях сокращается)
_NC = (WAX_M - 2.016) / 14.027
SS_TM, SS_DHM, SS_TTR, SS_DHTR = coutinho_nalkane(_NC)
SS_R, SS_Q = 0.1 * _NC + 0.0672, 0.1 * _NC + 0.1141
SS_LAM = -(2.0 / 6.0) * (nalkane_dh_vap(_NC) + SS_DHM + SS_DHTR)
SS_DV_R = WAX_M * 1e-3 * (1.0 / ro_wax_liq - 1.0 / ro_p) / R  # физический скачок объема при плавлении / R


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
    if wax_solid_solution:
        return ss_split(wax, T, p, sus)
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
def _ss_ln_k0(k, t_abs, p):
    """ln(1/x_k^0): идеальная растворимость чистой группы k по Coutinho (плавление и переход) с Пойнтингом."""
    r = SS_DHM[k] / (R * t_abs) * (1.0 - t_abs / SS_TM[k])
    if t_abs < SS_TTR[k]:
        r += SS_DHTR[k] / (R * t_abs) * (1.0 - t_abs / SS_TTR[k])
    if wax_pressure:
        r += SS_DV_R[k] * (p - P_ref_wax) / t_abs
    return r


@njit(cache=True)
def _ss_ln_gamma(xs, t_abs, s, out):
    """ln gamma_k твердого раствора состава xs (UNIQUAC Coutinho, `thermo.solids.UniquacSolidWax.ln_gamma`, Z = 6);
    s - скретч длины N_W."""
    sr, sq = 0.0, 0.0
    for i in range(N_W):
        sr += xs[i] * SS_R[i]
        sq += xs[i] * SS_Q[i]
    for i in range(N_W):  # s_i = sum_j theta_j tau_ji
        acc = 0.0
        for j in range(N_W):
            acc += xs[j] * SS_Q[j] / sq * math.exp(-(max(SS_LAM[j], SS_LAM[i]) - SS_LAM[i]) / (SS_Q[i] * R * t_abs))
        s[i] = acc
    for i in range(N_W):
        acc = 0.0
        for j in range(N_W):
            acc += math.exp(-(max(SS_LAM[i], SS_LAM[j]) - SS_LAM[j]) / (SS_Q[j] * R * t_abs)) * xs[j] * SS_Q[j] / sq / s[j]
        f = SS_R[i] / sr                 # phi_i/x_i
        g = f / (SS_Q[i] / sq)           # phi_i/theta_i
        out[i] = (math.log(f) + 1.0 - f - 3.0 * SS_Q[i] * (math.log(g) + 1.0 - g)
                  + SS_Q[i] * (1.0 - math.log(s[i]) - acc))


@njit(cache=True)
def _ss_incipient(z, lk0, t_abs, xs, w, s):
    """S = sum z_k K_k пробной твердой фазы: S > 1 - твердый раствор выпадает. xs на входе - старт, на выходе -
    пробный состав; w, s - скретчи длины N_W."""
    big = 0.0
    for _ in range(300):
        _ss_ln_gamma(xs, t_abs, s, w)
        big = 0.0
        for k in range(N_W):
            w[k] = z[k] * math.exp(lk0[k] - w[k])  # W_k
            big += w[k]
        diff = 0.0
        for k in range(N_W):
            v = w[k] / big
            diff = max(diff, abs(v - xs[k]))
            xs[k] = v
        if diff < 1e-11:
            break
    return big


@njit(cache=True)
def ss_split(wax, T, p, sus):
    """`sle_split` для твердого раствора: flash жидкость - твердый раствор в ячейке.

    Моли на грамм нефти: группы n_k = w_k/M_k и не выпадающая часть (растворитель с асфальтенами и смолами, газ).
    Доля твердого psi - корень уравнения Рачфорда-Райса, K_k = x_k^S/x_k^L = exp(ln(1/x_k^0) - ln gamma_k), состав
    твердого - простыми итерациями (как `UniquacSolidWax.precipitate`). sus - результат: взвешенные доли групп.

    ponytail: одна твердая фаза; расслоение на несколько твердых растворов - когда понадобится."""
    t_abs = T + 273.15 - wax_pore_shift
    ws = np.empty((5, N_W))
    xs, lk0, z, kk, s = ws[0], ws[1], ws[2], ws[3], ws[4]
    n0 = n_gas(T, p)
    w_sum, nt = 0.0, 0.0
    for k in range(N_W):
        w_sum += wax[k]
        nt += wax[k] / WAX_M[k]
    n0 += (1.0 - w_sum) / M_o
    nt += n0
    z0 = n0 / nt
    big = 0.0
    for k in range(N_W):
        z[k] = wax[k] / WAX_M[k] / nt
        lk0[k] = _ss_ln_k0(k, t_abs, p)
        xs[k] = z[k] * math.exp(lk0[k])
        big += xs[k]
    for k in range(N_W):
        xs[k] = xs[k] / big if big > 0.0 else 1.0 / N_W
    if _ss_incipient(z, lk0, t_abs, xs, kk, s) <= 1.0:
        hl = 0.0
        for k in range(N_W):
            sus[k] = 0.0
            hl += WAX_L_REL[k] * wax[k]
        return w_sum, 0.0, hl
    psi = 0.0
    for _ in range(500):
        _ss_ln_gamma(xs, t_abs, s, kk)
        for k in range(N_W):
            kk[k] = math.exp(lk0[k] - kk[k])
        # Рачфорд-Райс: f(psi) = sum_k z_k (K_k - 1)/(1 + psi (K_k - 1)) - z0/(1 - psi), убывает на (0, 1)
        lo, hi = 0.0, 1.0
        for _ in range(100):
            f, df = -z0 / (1.0 - psi), -z0 / (1.0 - psi) ** 2
            for k in range(N_W):
                d = 1.0 + psi * (kk[k] - 1.0)
                f += z[k] * (kk[k] - 1.0) / d
                df -= z[k] * (kk[k] - 1.0) ** 2 / (d * d)
            if f > 0.0:
                lo = psi
            else:
                hi = psi
            step = psi - f / df
            psi_new = step if lo < step < hi else 0.5 * (lo + hi)
            done = abs(psi_new - psi) < 1e-15
            psi = psi_new
            if done:
                break
        big = 0.0
        for k in range(N_W):
            s[k] = kk[k] * z[k] / (1.0 + psi * (kk[k] - 1.0))  # x_k^S до нормировки
            big += s[k]
        diff = 0.0
        for k in range(N_W):
            v = s[k] / big
            diff = max(diff, abs(v - xs[k]))
            xs[k] = v
        if diff < 1e-11:
            break
    w_dis, w_sus, hl = 0.0, 0.0, 0.0
    for k in range(N_W):
        sk = min(wax[k], max(0.0, psi * nt * xs[k] * WAX_M[k]))
        sus[k] = sk
        w_dis += wax[k] - sk
        w_sus += sk
        hl += WAX_L_REL[k] * (wax[k] - sk)
    return w_dis, w_sus, hl


@njit(cache=True)
def single_split(w, T, p):
    """`sle_split` при одной группе (N_W = 1) в замкнутой форме: растворенная и взвешенная доли.

    Насыщение - если w > x*n0/(1 - x)*M, n0 = (1 - w)/M_o + n_g: тогда растворено x*n0/(1 - x)*M. Без давления и газа
    это формула (6.2) однокомпонентной модели (`single_wp_saturated`). Ни буфера, ни цикла по группам - путь шага
    однокомпонентного расчета (`components_equation`, `flows_in_cells`)."""
    x = x_saturation(0, T, p)
    if x >= 1.0:
        return w, 0.0
    dis = min(w, x * ((1.0 - w) / M_o + n_gas(T, p)) / (1.0 - x) * WAX_M[0])
    return dis, w - dis


@njit(cache=True)
def sle_hl_boundary(bc_wc, bound, T, p):
    """Носитель скрытой теплоты втекающей через границу `bound` нефти состава `bc_wc[bound]` (ГУ Дирихле) - тот же жадный
    набор насыщенных групп, что `sle_split`, но без буфера: x_k^sat пересчитывается на месте.

    Зовется в цикле по граням `flows_in_cells`, поэтому без буфера и без среза `bc_wc[bound, :]`: срез (вид на массив)
    в этой ветке, даже невыполняемой, замедлял весь цикл по ячейкам (75x75: 630 -> 870 мкс). Путь редкий (только грани
    с ГУ `DataField.Paraffin`), поэтому лишние экспоненты не стоят ничего."""
    if N_W == 1:
        return WAX_L_REL[0] * single_split(bc_wc[bound, 0], T, p)[0]
    if wax_solid_solution:
        return ss_split(bc_wc[bound, :N_W].copy(), T, p, np.empty(N_W))[2]
    w_sum = 0.0
    a = n_gas(T, p)
    for k in range(N_W):
        w_sum += bc_wc[bound, k]
        a += bc_wc[bound, k] / WAX_M[k]
    a += (1.0 - w_sum) / M_o
    b = 1.0
    mask = 0
    for _ in range(N_W):
        best = -1
        t_best = -1.0
        x_best = 0.0
        for k in range(N_W):
            x_k = x_saturation(k, T, p)
            if (mask >> k) & 1 or bc_wc[bound, k] <= 0.0 or x_k >= 1.0:
                continue
            t_k = bc_wc[bound, k] / (WAX_M[k] * x_k) if x_k > 0.0 else math.inf
            if t_k > t_best:
                t_best = t_k
                best = k
                x_best = x_k
        if best < 0 or t_best * b <= a:
            break
        mask |= 1 << best
        a -= bc_wc[bound, best] / WAX_M[best]
        b -= x_best
    n_l = a / b
    hl = 0.0
    for k in range(N_W):
        dis = x_saturation(k, T, p) * n_l * WAX_M[k] if (mask >> k) & 1 else bc_wc[bound, k]
        hl += WAX_L_REL[k] * dis
    return hl


@njit(cache=True)
def wat_cell(wax, p):
    """Температура начала кристаллизации нефти с составом `wax` при давлении p, [C].

    Пока все растворено, x_k = (w_k/M_k)/n_L. Первой насыщается группа с наибольшей температурой из (1):
        T_k = (dH_k/R + dv_k/R*(P - P_ref)) / (dH_k/(R*Tm_k) - ln x_k).
    С `wax_eos` - бисекция по T в пределах сетки таблиц.
    """
    if wax_eos or wax_solid_solution:
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
    """Насыщена ли хоть одна группа при температуре T [C]: w_k/M_k > x_k^sat*n_L при всем растворенном
    (твердый раствор - выпадает ли он)."""
    if wax_solid_solution:
        return ss_split(wax, T, p, np.empty(N_W))[1] > 0.0
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
    """WAT бисекцией на [T_LO, T_HI] (таблицы уравнения состояния или твердый раствор), за сеткой - ее край, [C]."""
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


@njit(cache=True)
def single_wp_saturated(w_sum, T):
    """Растворенный парафин одного псевдокомпонента (MW, Tm, alpha), формула (6.1)-(6.2) - эталон редукции `sle_split`
    к одной группе (`wax_characterization = 'single'`, `tests/test_thermo_wax.py`) и рисунков описания модели.

    Предел растворимости идеального раствора (Шредер - ван Лаар) с эффективными alpha, Tm:
        x_sat(T) = min(1, exp[-(alpha/R)*(1/T - 1/Tm)]),   T, Tm в К,
    переводится в массовую долю через молярные массы; взвесь в раствор не входит, поэтому предел относится к
    w_o + w_p = 1 - w_ps:
        w_hat = x_sat*MW/(x_sat*MW + (1 - x_sat)*M_o),   w_p_sat = w_hat/(1 - w_hat)*(1 - w_sum).
    """
    if T >= Tm:
        return w_sum  # выше температуры плавления парафин растворим полностью
    x_sat = math.exp(-alpha / R * (1.0 / (T + 273.15) - 1.0 / Tm_K))
    if x_sat >= 1.0:
        return w_sum
    w_hat = x_sat * MW / (x_sat * MW + (1.0 - x_sat) * M_o)
    return min(w_sum, w_hat / (1.0 - w_hat) * (1.0 - w_sum))
