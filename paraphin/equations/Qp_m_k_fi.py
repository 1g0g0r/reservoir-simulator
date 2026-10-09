"""Кольматация пучка капилляров: функция пор fi, пористость, проницаемость и скорости потери порового объема."""
from numba import njit

from paraphin.geometry import n_pass, w2_cv, w4_cv, plug_cv, w43_cv, n_pass_a, cbrt_r1
from paraphin.constants import (Nr, init_m, init_k, min_Wps_bound, ro_o, ro_p, ro_asph_dep, resin_in_deposit, volume,
                                asphaltenes)
from paraphin.layout import IA_F, I_R
from paraphin.equations.Pore_bundle import update_fi_rows, Q00 as _Q00, Q01 as _Q01, Q11 as _Q11

# Интегралы по кусочно-линейным fi и ur (`_calculate_integrals`) - скалярные произведения с узловыми весами
# (`geometry.w2_cv`, `w4_cv`) и билинейная форма по соседним узлам (`Pore_bundle.Q00` ...). Раньше на каждом отрезке
# делилось на dr и вычитались степени r: активная ячейка стоила ~3 мкс, почти все - в интегралах
# (docs/PERFORMANCE_FINDINGS.md, «Раунд 6»).


@njit(cache=True)
def _blocking(Ub, fi, i: int, j: int, weights) -> float:
    """sum(weights*Ub*fi) по блокируемым узлам (r < r_pass): с весами `w2_cv` - объем каналов, уходящих из
    проводящих в тупиковые за единицу времени, с весами `plug_cv` - объем пробок, оба в единицах
    интеграла r^2*fi. `Ub` хранит коэффициент b(r), сама скорость блокирования ub = b(r)*fi."""
    s = 0.0
    for ij in range(n_pass):
        s += weights[ij] * Ub[i, j, ij] * fi[i, j, ij]
    return s


@njit(cache=True)
def _calculate_integrals(fi, Ur, i: int, j: int):
    """Интегралы функции пор по размерам для ячейки (i, j).

    Подынтегральные функции восстанавливаются кусочно-линейно по узлам сетки радиусов, поэтому
    интегралы берутся точно по каждому отрезку (для r^2*fi это то же, что sum(w2_cv*fi)).

    Returns
    -------
    int_r_ur_fi, r2fi, r4fi: float
        Интегралы r*ur*fi, r^2*fi и r^4*fi
    """
    int_r_ur_fi, r2fi, r4fi = 0.0, 0.0, 0.0
    for ij in range(Nr):
        f = fi[i, j, ij]
        r2fi += w2_cv[ij] * f  # r^2 * fi
        r4fi += w4_cv[ij] * f  # r^4 * fi
    for ij in range(1, Nr):    # r * ur * fi
        f0, f1, u0, u1 = fi[i, j, ij - 1], fi[i, j, ij], Ur[i, j, ij - 1], Ur[i, j, ij]
        int_r_ur_fi += f0 * (_Q00[ij - 1] * u0 + _Q01[ij - 1] * u1) + f1 * (_Q01[ij - 1] * u0 + _Q11[ij - 1] * u1)

    return int_r_ur_fi, r2fi, r4fi


_RO_P_RO_O = ro_p / ro_o       # множитель стока кристаллов q_p1 + q_p2 в балансах, поделенных на ro_o, а не на ro_p
_RO_AD_O = ro_asph_dep / ro_o  # множитель стока осадка асфальтены + смолы в балансах, поделенных на ro_o


@njit(cache=True)
def calc_qp_m_k_fi(i, j, S, Wps, Wc, m, k, fi, Ur, Ub, kx, integr_r2_fi0, integr_r4_fi0, rows,
                     new_qp1, new_qp2, new_fi, new_k, new_m, new_kx, out_o, dt) -> None:
    """Скорости потери порового объема, пористость, проницаемость и функция пор по размерам.

    `fi` описывает только проводящие каналы: проницаемость восстанавливается по интегралу r^4*fi, отнесенному к тому
    же интегралу от начального распределения (`integr_r4_fi0`). Сама `fi` обновляется неявной прогонкой по сетке
    радиусов (`Pore_bundle.update_fi_rows`).

    Блокированный канал из `fi` уходит (проводимость теряется), но из порового пространства - нет: горло затыкает
    один кристалл, а объем канала с нефтью остается тупиковой пористостью ячейки (поровые тела связаны и другими
    горлами - гелиевая пористость кернов после отложения парафина падает много меньше проницаемости, He et al., 2020).
    Поэтому пористость - это проводящие каналы плюс тупиковый объем: m = m0*int r^2*fi/int r^2*fi0 + m_тупик;
    отдельного поля под тупиковый объем нет. Скорости потери порового объема - чистые материалы:
        q_p1 - осадок кристаллов на стенках, q_p2 - пробки (кристалл D^3/(6*Lk) в единицах r^2 на каждый
        блокированный канал, `plug_cv`), q_pa - осадок асфальтены + смолы.
    После прогонки они берутся по фактическому изменению `fi`: блокирование неявное, за шаг из узла уходит
    dt*lim*Ub*new_fi, и с весами `w2_cv` (объем каналов) и `plug_cv` (пробки) это те же единицы, что интеграл r^2*fi.
    Тогда (q_p1 + q_p2 + q_pa)*dt = m - m^new по построению: на это тождество опираются уравнения насыщенности,
    переноса компонентов и энергии.

    Скорость изменения радиуса - сумма вкладов кристаллов парафина и флокул:
        u(r) = lim_w*Ur(r) + lim_a*Ua*r^(1/3),   r >= r_pass_a,
    блокирование - только кристаллами (флокула мельче горла, см. `n_pass_a` в `paraphin/geometry.py`).
    Ограничители подводом свои у каждого вида частиц: за шаг осаждается не больше парафина, чем его
    взвешено, и не больше осадка асфальтены + смолы, чем есть флокул и смол. Запас - то, что останется в
    ячейке после явного оттока: (m*S_o - dt*out_o/V)*w, out_o - отток нефти через грани и добывающую
    скважину, [м^3/с]. Без вычета оттока (прежний запас m*S_o*w_ps однокомпонентной модели) у флокул и почти целиком
    выпавших тяжелых групп, где растворенной части нет, доля уходила в минус, и зажим в `components_equation`
    превращал ее в прибавку массы. Скорости `Ur`, `Ub` со стоксовской диффузией на порядки быстрее переноса, поэтому
    осаждение обычно лимитируется подводом, а кинетика задает лишь протяженность зоны осаждения.

    Убыль проводящих каналов от сужения (как и q_p1 в прежней функции - по фактическому изменению fi)
    делится между парафином и асфальтенами пропорционально их оценкам до прогонки:
        q_p1 + q_pa = q_сужение,   q_pa/q_сужение = I_a/(I_w + I_a),
        I_w = -2*m0*lim_w*int r*Ur*fi dr/int r^2*fi0,   I_a = -2*m0*lim_a*Ua*int r^(4/3)*fi dr/int r^2*fi0.
    Тождество пористости сохраняется точным: (q_p1 + q_p2 + q_pa)*dt = m - m^new. Осадок асфальтенов
    содержит долю смол `resin_in_deposit`, плотность осадка `ro_asph_dep` (Wang & Civan, 2005: общий
    поровый объем делят парафин и асфальтены).

    Если парафин ниже порога кольматации, его скорости Ur, Ub в этой ячейке не обновлялись и могут быть
    устаревшими - они берутся с lim_w = 0.

    Коэффициент сужения флокулами Ua - `kx['ua']` (с прошлого шага), скорость q_pa пишется в
    `new_kx['qpa']`. Прогонка по радиусам - общая `Pore_bundle.update_fi_rows` по профилям, собранным в строки
    скретча `rows` (`Solver.rows[i]`): u_ij = lim_w*Ur_ij + lim_a*ua*r_ij^(1/3) (r_ij >= r_pass_a), b_ij = lim_w*Ub_ij.
    """
    wax_on = Wps[i, j] > min_Wps_bound
    ua = kx[i, j].ua if asphaltenes else 0.0  # без асфальтенов ветка флокул выбрасывается на компиляции
    if wax_on or ua < 0.0:
        to_m = init_m / integr_r2_fi0
        mso_dt = max(m[i, j] * (1.0 - S[i, j]) / dt - out_o / volume, 0.0)
        int_r_ur_fi, r2fi, _ = _calculate_integrals(fi, Ur, i, j)

        lim_w, i_w = 0.0, 0.0
        if wax_on:
            i_w = max(-2.0 * to_m * int_r_ur_fi, 0.0)
            qp2 = to_m * _blocking(Ub, fi, i, j, plug_cv)
            sink = _RO_P_RO_O * (i_w + qp2)
            avail = mso_dt * Wps[i, j]
            lim_w = avail / sink if sink > avail else 1.0

        lim_a, i_a = 0.0, 0.0
        if ua < 0.0:
            s43 = 0.0
            for ij in range(n_pass_a, Nr):
                s43 += w43_cv[ij] * fi[i, j, ij]
            i_a = -2.0 * to_m * ua * s43
            sink_a = _RO_AD_O * i_a
            # Запас на осадок: флокулы дают (1 - f_r) его массы, смолы - f_r
            avail_a = mso_dt * min(Wc[i, j, IA_F] / (1.0 - resin_in_deposit),
                                   Wc[i, j, I_R] / resin_in_deposit if resin_in_deposit > 0.0 else 1e300)
            lim_a = avail_a / sink_a if sink_a > avail_a else 1.0

        u, b = rows.u, rows.tmp
        for ij in range(Nr):
            u[ij] = lim_w * Ur[i, j, ij]
            if ij >= n_pass_a:
                u[ij] += lim_a * ua * cbrt_r1[ij]
            b[ij] = lim_w * Ub[i, j, ij]
        update_fi_rows(new_fi, fi, i, j, u, b, rows.a, rows.b, dt)
        _, r2fi_new, r4fi_new = _calculate_integrals(new_fi, Ur, i, j)

        blocked = dt * lim_w * to_m * _blocking(Ub, new_fi, i, j, w2_cv)
        qp2 = lim_w * to_m * _blocking(Ub, new_fi, i, j, plug_cv)
        q_narrow = max((to_m * (r2fi - r2fi_new) - blocked) / dt, 0.0)
        i_wa = lim_w * i_w + lim_a * i_a
        qpa = q_narrow * (lim_a * i_a / i_wa) if i_wa > 0.0 else 0.0
        qp1 = q_narrow - qpa

        if qp1 + qp2 + qpa > 0.0:
            new_m[i, j] = m[i, j] - (qp1 + qp2 + qpa) * dt
            new_k[i, j] = init_k * r4fi_new / integr_r4_fi0
            scale = (m[i, j] - new_m[i, j]) / ((qp1 + qp2 + qpa) * dt)
            qp1 *= scale
            qp2 *= scale
            qpa *= scale
        else:
            qp1, qp2, qpa = 0.0, 0.0, 0.0
            new_m[i, j] = m[i, j]
            new_k[i, j] = k[i, j]

        new_qp1[i, j] = qp1
        new_qp2[i, j] = qp2
        new_kx[i, j].qpa = qpa
    else:
        new_qp1[i, j] = 0.0
        new_qp2[i, j] = 0.0
        new_kx[i, j].qpa = 0.0
        new_m[i, j] = m[i, j]
        new_k[i, j] = k[i, j]
