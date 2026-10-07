"""Решение уравнения концентрации взвешенных частиц парафина по явной схеме."""
import numpy as np
from numba import njit

from paraphin.geometry import r1, r5, r6, n_pass, dr_cv, w2_cv, plug_cv, w43_cv, n_pass_a, cbrt_r1
from paraphin.constants import Nr, init_m, init_k, min_Wps_bound, ro_o, ro_asph_dep, resin_in_deposit, volume
from paraphin.layout import IA_F, I_R, KX_UA, KX_QPA, ROW_U, ROW_TMP, ROW_A, ROW_B
from paraphin.equations.Pore_bundle import update_fi_rows
from paraphin.equations.Wp_balance import _RO_P_RO_O  # тот же множитель стока кристаллов, что в `wp_equation`

# Веса интегралов по кусочно-линейным fi и ur (`_calculate_integrals`) - константы сетки радиусов: интегралы
# r^2*fi и r^4*fi - скалярные произведения с узловыми весами, r*ur*fi - билинейная форма по соседним узлам. Раньше
# на каждом отрезке делилось на dr и вычитались степени r: активная ячейка стоила ~3 мкс, почти все - в интегралах
# (docs/PERFORMANCE_FINDINGS.md, «Раунд 6»). На отрезке [a, b], r = a + s*dr: fi = fi0*(1 - s) + fi1*s.
_A, _DR = r1[:-1], r1[1:] - r1[:-1]
_W4 = np.zeros(Nr)  # int r^4*fi dr = sum(_W4*fi); для r^2 - `w2_cv` (geometry), те же формулы
_W4[1:] += ((r6[1:] - r6[:-1]) / 6 - _A * (r5[1:] - r5[:-1]) / 5) / _DR
_W4[:-1] += (r1[1:] * (r5[1:] - r5[:-1]) / 5 - (r6[1:] - r6[:-1]) / 6) / _DR
_Q00 = _DR * (_A / 3 + _DR / 12)  # int r*phi0*phi0 dr на отрезке, phi0 = 1 - s
_Q01 = _DR * (_A / 6 + _DR / 12)  # int r*phi0*phi1 dr, phi1 = s
_Q11 = _DR * (_A / 3 + _DR / 4)   # int r*phi1*phi1 dr
_INV_DR_CV = 1.0 / dr_cv          # прогонка `_update_fi`: умножение вместо деления


@njit(cache=True)
def calc_qp_m_k_fi(i, j, S, Wp, Wps, m, k, fi, Ur, Ub, integr_r2_fi0, integr_r4_fi0, a_tdma, b_tdma,
                   new_qp1, new_qp2, new_fi, new_k, new_m, dt) -> None:
    """Скорости потери порового объема, пористость, проницаемость и функция пор по размерам.

    `fi` описывает только проводящие каналы: проницаемость восстанавливается по интегралу r^4*fi,
    отнесенному к тому же интегралу от начального распределения (`integr_r4_fi0`). Сама `fi`
    обновляется прогонкой по сетке радиусов.

    Блокированный канал из `fi` уходит (проводимость теряется), но из порового пространства - нет:
    горло затыкает один кристалл, а объем канала с нефтью остается тупиковой пористостью ячейки
    (поровые тела связаны и другими горлами - гелиевая пористость кернов после отложения парафина
    падает много меньше проницаемости, He et al., 2020). Поэтому пористость - это проводящие каналы
    плюс тупиковый объем: m = m0*int r^2*fi/int r^2*fi0 + m_тупик; отдельного поля под тупиковый
    объем нет, он равен m - m0*int r^2*fi/int r^2*fi0. Раньше блокированный канал выбывал целиком
    вместе с нефтью, и потеря пор в керновых опытах выходила в 4-5 раз больше объема кристаллов.

    Обе скорости потери порового объема - кристаллы (чистый парафин с плотностью ro_p):

        q_p1 - осадок на стенках: убыль проводящих каналов за вычетом ушедших в тупиковые,
               оценка до прогонки -2*m0*int r*ur*fi dr / int r^2*fi0 >= 0 (ur < 0);
        q_p2 - пробки: m0*sum(plug_cv*ub)/int r^2*fi0, объем кристалла D^3/(6*Lk) в единицах r^2
               на каждый блокированный канал (`plug_cv` в `paraphin/geometry.py`).

    После прогонки обе берутся по фактическому изменению `fi`: блокирование в `_update_fi` неявное,
    за шаг из узла уходит dt*limiter*Ub*new_fi, и с весами `w2_cv` (объем каналов) и `plug_cv`
    (пробки) это дает точно те же единицы, что интеграл r^2*fi. Тогда q_p1 + q_p2 = (m - m^new)/dt
    по построению - этим тождеством связаны (9), (11) и (12), и на него опираются уравнения
    насыщенности и парафина.

    Осаждение ограничено подводом: за шаг из нефтяной фазы не может уйти больше кристаллов, чем в ней
    есть, m*S_o*w_ps. Скорости `Ur`, `Ub` считаются по взвеси на начало шага и с физическим
    (стоксовским) коэффициентом диффузии на порядки быстрее переноса, так что без ограничителя
    `wp_equation` зажимала бы долю в нуле и создавала массу. Множитель `limiter` уменьшает обе
    скорости в прогонке, поэтому цепочка «скорости -> fi -> m -> q_p -> сток» остается замкнутой:
    осаждение в этом режиме лимитируется переносом, а не кинетикой. Ради той же замкнутости `m`
    и `k` берутся от `new_fi` этого же шага, а не от `fi` предыдущего.

    a_tdma, b_tdma: numpy.ndarray(Nr)
        Прогоночные коэффициенты. Своя строка на каждый i: один общий буфер на все ячейки давал
        гонку в prange - потоки затирали друг другу коэффициенты, и fi считалась по мусору.

    Описание остальных аргументов - в докстринге пакета `paraphin.equations`.
    """
    if Wps[i, j] > min_Wps_bound:
        to_m = init_m / integr_r2_fi0  # интеграл r^2*fi -> пористость
        int_r_ur_fi, r2fi, _ = _calculate_integrals(fi, Ur, i, j)
        qp1 = max(-2.0 * to_m * int_r_ur_fi, 0.0)
        qp2 = to_m * _blocking(Ub, fi, i, j, plug_cv)

        # Ограничение подводом: осадок и пробки - кристаллы из взвеси, за шаг не больше ее запаса
        sink = _RO_P_RO_O * (qp1 + qp2)
        avail = m[i, j] * (1.0 - S[i, j]) * Wps[i, j] / dt
        limiter = avail / sink if sink > avail else 1.0

        # Обновление функции пор по размерам, по ней - проницаемость и убыль проводящих каналов
        _update_fi(new_fi, fi, Ur, Ub, i, j, a_tdma, b_tdma, dt, limiter)
        _, r2fi_new, r4fi_new = _calculate_integrals(new_fi, Ur, i, j)

        # Точный учет по схеме: сколько каналов ушло в тупиковые (w2_cv) и сколько на пробки ушло кристаллов (plug_cv)
        blocked = dt * limiter * to_m * _blocking(Ub, new_fi, i, j, w2_cv)
        qp2 = limiter * to_m * _blocking(Ub, new_fi, i, j, plug_cv)
        qp1 = max((to_m * (r2fi - r2fi_new) - blocked) / dt, 0.0)

        if qp1 + qp2 > 0.0:
            new_m[i, j] = m[i, j] - (qp1 + qp2) * dt  # тупиковый объем в пористости остается
            new_k[i, j] = init_k * r4fi_new / integr_r4_fi0
            # Разность m - new_m округлена до ulp(m) ~ 5e-17, а убыль бывает того же порядка: скорости -
            # по фактической разности, иначе тождество выполнялось бы лишь с точностью до ulp(m)
            scale = (m[i, j] - new_m[i, j]) / ((qp1 + qp2) * dt)
            qp1 *= scale
            qp2 *= scale
        else:
            # Нечему оседать (Ur = Ub = 0, например при S_o = S_o*): fi не меняется, а пересчет k из нее
            # дает дрейф ~1e-12 от округления. Тождество q_p1 + q_p2 = (m - m^new)/dt должно быть точным.
            qp1, qp2 = 0.0, 0.0
            new_m[i, j] = m[i, j]
            new_k[i, j] = k[i, j]

        new_qp1[i, j] = qp1
        new_qp2[i, j] = qp2
    else:
        # Ниже порога кольматации поля не меняются: переносим текущие значения, чтобы new_*
        # оставались согласованы с текущим слоем (new_m читает `saturation_equation`).
        new_qp1[i, j] = 0.0
        new_qp2[i, j] = 0.0
        new_m[i, j] = m[i, j]
        new_k[i, j] = k[i, j]


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
        r4fi += _W4[ij] * f    # r^4 * fi
    for ij in range(1, Nr):    # r * ur * fi
        f0, f1, u0, u1 = fi[i, j, ij - 1], fi[i, j, ij], Ur[i, j, ij - 1], Ur[i, j, ij]
        int_r_ur_fi += f0 * (_Q00[ij - 1] * u0 + _Q01[ij - 1] * u1) + f1 * (_Q01[ij - 1] * u0 + _Q11[ij - 1] * u1)

    return int_r_ur_fi, r2fi, r4fi


@njit(cache=True)
def _update_fi(new_fi, fi, Ur, Ub, i: int, j: int, a_tdma, b_tdma, dt, limiter=1.0):
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
    Поток через грань делится на ширину контрольного объема узла `dr_cv[ij]`, а не на общий шаг:
    схема не привязана к равномерной сетке, сохраняется взвешенная сумма fi*dr_cv.

    `limiter` - общий множитель к обеим скоростям (ограничение подводом взвеси, см. `calc_qp_m_k_fi`).

    На правой границе (r = r_max) соседа нет, что равносильно условию fi = 0: капилляров шире
    r_max нет, а те, что на r_max, сужаются внутрь и ничем не замещаются - поэтому правый узел
    проседает первым. Это свойство модели, а не схемы: на сетке в 100 раз мельче падение то же.

    Описание аргументов - в докстринге пакета `paraphin.equations`.
    """
    # Вычисление прогоночных коэффициентов
    inv_dt = 1.0 / dt
    d = inv_dt + (abs(Ur[i, j, 0]) * _INV_DR_CV[0] + Ub[i, j, 0]) * limiter
    e = min(Ur[i, j, 1], 0.0) * _INV_DR_CV[0] * limiter
    a_tdma[0] = -e / d
    b_tdma[0] = fi[i, j, 0] * inv_dt / d

    for ij in range(1, Nr):
        c = -max(Ur[i, j, ij - 1], 0.0) * _INV_DR_CV[ij] * limiter
        d = inv_dt + (abs(Ur[i, j, ij]) * _INV_DR_CV[ij] + Ub[i, j, ij]) * limiter
        e = min(Ur[i, j, ij + 1], 0.0) * _INV_DR_CV[ij] * limiter if ij + 1 < Nr else 0.0  # за Nr-1 соседа нет
        inv = 1.0 / (c * a_tdma[ij - 1] + d)
        a_tdma[ij] = -e * inv
        b_tdma[ij] = (fi[i, j, ij] * inv_dt - c * b_tdma[ij - 1]) * inv

    # Вычисление функции пор размерам
    new_fi[i, j, Nr - 1] = b_tdma[Nr - 1]
    for ij in range(Nr - 2, -1, -1):  # обратный ход
        new_fi[i, j, ij] = new_fi[i, j, ij + 1] * a_tdma[ij] + b_tdma[ij]


# ------------------------------------------------------------------------------------------------------
# Совместное осаждение парафина и асфальтенов (флаг `asphaltenes`). Прежние функции выше не трогаются:
# с выключенным флагом `_equations_loop` вызывает их, и расчет побитово совпадает с прежним.
# ------------------------------------------------------------------------------------------------------
_RO_AD_O = ro_asph_dep / ro_o  # множитель стока осадка асфальтены + смолы в балансах, поделенных на ro_o


@njit(cache=True)
def calc_qp_m_k_fi_2(i, j, S, Wps, Wc, m, k, fi, Ur, Ub, kx, integr_r2_fi0, integr_r4_fi0, rows,
                     new_qp1, new_qp2, new_fi, new_k, new_m, new_kx, out_o, dt) -> None:
    """То же, что `calc_qp_m_k_fi`, плюс сужение капилляров флокулами асфальтенов.

    Скорость изменения радиуса - сумма вкладов кристаллов парафина и флокул:
        u(r) = lim_w*Ur(r) + lim_a*Ua*r^(1/3),   r >= r_pass_a,
    блокирование - только кристаллами (флокула мельче горла, см. `n_pass_a` в `paraphin/geometry.py`).
    Ограничители подводом свои у каждого вида частиц: за шаг осаждается не больше парафина, чем его
    взвешено, и не больше осадка асфальтены + смолы, чем есть флокул и смол. Запас - то, что останется в
    ячейке после явного оттока: (m*S_o - dt*out_o/V)*w, out_o - отток нефти через грани и добывающую
    скважину, [м^3/с]. У прежней функции запас m*S_o*w_ps: там сток берется из суммы растворенного и
    взвешенного парафина, и растворенная часть перекрывает отток. У флокул и почти целиком выпавших тяжелых
    групп растворенной части нет, и такой запас давал отрицательную долю, которую зажим в
    `components_equation` превращал в прибавку массы.

    Убыль проводящих каналов от сужения (как и q_p1 в прежней функции - по фактическому изменению fi)
    делится между парафином и асфальтенами пропорционально их оценкам до прогонки:
        q_p1 + q_pa = q_сужение,   q_pa/q_сужение = I_a/(I_w + I_a),
        I_w = -2*m0*lim_w*int r*Ur*fi dr/int r^2*fi0,   I_a = -2*m0*lim_a*Ua*int r^(4/3)*fi dr/int r^2*fi0.
    Тождество пористости сохраняется точным: (q_p1 + q_p2 + q_pa)*dt = m - m^new. Осадок асфальтенов
    содержит долю смол `resin_in_deposit`, плотность осадка `ro_asph_dep` (Wang & Civan, 2005: общий
    поровый объем делят парафин и асфальтены).

    Если парафин ниже порога кольматации, его скорости Ur, Ub в этой ячейке не обновлялись и могут быть
    устаревшими - они берутся с lim_w = 0.

    Коэффициент сужения флокулами Ua - `kx[..., KX_UA]` (с прошлого шага), скорость q_pa пишется в
    `new_kx[..., KX_QPA]`. Прогонка по радиусам - общая `Pore_bundle.update_fi_rows` по профилям, собранным в строки
    скретча `rows` (`Solver.rows[i]`): u_ij = lim_w*Ur_ij + lim_a*ua*r_ij^(1/3) (r_ij >= r_pass_a), b_ij = lim_w*Ub_ij.
    """
    wax_on = Wps[i, j] > min_Wps_bound
    ua = kx[i, j, KX_UA]
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

        u, b = rows[ROW_U], rows[ROW_TMP]
        for ij in range(Nr):
            u[ij] = lim_w * Ur[i, j, ij]
            if ij >= n_pass_a:
                u[ij] += lim_a * ua * cbrt_r1[ij]
            b[ij] = lim_w * Ub[i, j, ij]
        update_fi_rows(new_fi, fi, i, j, u, b, rows[ROW_A], rows[ROW_B], dt)
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
        new_kx[i, j, KX_QPA] = qpa
    else:
        new_qp1[i, j] = 0.0
        new_qp2[i, j] = 0.0
        new_kx[i, j, KX_QPA] = 0.0
        new_m[i, j] = m[i, j]
        new_k[i, j] = k[i, j]
