"""Сравнение с опытом Sandyga et al. (2020): охлаждение керна при прокачке раствора парафина в керосине.

Sandyga M.S., Struchkov I.A., Rogachev M.K. Formation damage induced by wax deposition: laboratory
investigations and modeling // J. Petrol. Explor. Prod. Technol. 2020. V. 10. P. 2541-2558.
doi:10.1007/s13202-020-00924-2 (resources/литература/парафиновое/sandyga2020_formation_damage_wax_core.pdf).

Раствор 20% масс. парафина C20-C40 в керосине прокачивается через песчаник (3 x 5 см, пористость 9%) с
0.5 см^3/мин, пока весь стенд в термошкафу охлаждается от 40 C со скоростью 1 C/ч. Сравнивается:
  A. распределение пор по размерам (томография, рис. 6) с fi_0 модели;
  B. температура начала кристаллизации по (6.1) с параметрами авторов (дH = 81.7 кДж/моль, M 506 и 163 г/моль)
     с измеренными 30 C (реометр) и 33.8 C (по перепаду в керне);
  C. рост градиента давления при охлаждении (рис. 4), пористость после опыта 9.0 -> 2.1% (табл. 1) и
     какие поры закупорены (рис. 6).
Комментарии и выводы - VALIDATION.md.

    python experiments/исходная_модель/validation_sandyga2020.py          # A, B и прогоны C (правят constants.py)
    python experiments/исходная_модель/validation_sandyga2020.py --plot   # только рисунок по сохраненному json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core_flood as cf  # noqa: E402

OUT = cf.ROOT / 'outputs' / 'data' / 'validation_sandyga2020.json'
FIG = cf.ROOT / 'outputs' / 'figures' / 'validation' / 'sandyga2020.png'
DARCY = 9.869233e-13  # [м^2]

# --- Данные Sandyga et al. (2020) ----------------------------------------------------------------------
W_WAX = 0.20                       # массовая доля парафина в растворе
MW_WAX, M_KEROSENE = 506.0, 163.0  # молярные массы псевдокомпонентов авторов (C36 и керосин), [г/моль]
DH = 81.7e3                        # энтальпия кристаллизации в керосине (по Вант-Гоффу, их ур. 9), [Дж/моль]
WAT_BULK, WAT_CORE = 30.0, 33.8    # WAT: реометр (рис. 3) и перелом градиента давления в керне (рис. 2, 4), [C]
RO_SOLUTION, RO_WAX = 800.0, 900.0  # керосин 780-810 кг/м^3, парафин ~900 кг/м^3 (разд. «Materials»)
# Вязкость раствора при 40 C не приведена (на рис. 3 выше 30 C она неотличима от нуля в масштабе 14 Па*с);
# принята как у керосина с растворенным парафином. Входит только в стоксовскую диффузию частиц.
MU_40 = 2.0e-3

# Керн и режим (разд. «The experimental procedure», рис. 2)
CORE_D, CORE_L, CORE_M = 0.03, 0.05, 0.09
Q = 0.5e-6 / 60.0                # 0.5 см^3/мин
T_START, T_END = 40.0, 32.8      # по тексту охлаждали до 33 C, кривая рис. 4 доходит до 32.8 C
COOLING = 1.0 / 3600.0           # 1 C/ч
POROSITY_AFTER = 0.021           # томография после опыта (табл. 1)

# Рис. 4: градиент давления от температуры, (T, C; МПа/м). Оцифровка синей кривой с растра 300 dpi по
# сетке графика (~0.1 C, ~0.3 МПа/м). Плавный рост 40 -> 35 C авторы объясняют ростом вязкости.
GRADIENT = [(40.0, 0.82), (38.0, 0.77), (36.0, 0.92), (35.0, 0.95), (34.5, 1.13), (34.2, 1.38), (34.0, 1.64),
            (33.9, 1.74), (33.6, 6.07), (33.5, 9.38), (33.4, 11.68), (33.3, 14.96), (33.2, 18.7), (33.1, 25.36),
            (33.0, 28.48), (32.9, 31.81), (32.8, 36.14)]

# Рис. 6: доли порового объема по диаметру горл (томография, разрешение 7-8 мкм), % от исходного
# порового объема: до и после опыта. Оцифровка столбцов с растра 300 dpi (~0.3%); последний столбец
# «после» на пределе различимости.
PORE_D_UM = [20.0, 30.0, 40.0, 50.0, 60.0]
PORES_BEFORE = [36.3, 30.2, 22.3, 5.6, 3.2]
PORES_AFTER = [8.7, 6.7, 3.3, 0.7, 0.5]

BEREA_ETA = 5.24  # извилистость керна опыта 1 Sutton & Roberts (калибровка d_p, L_k), см. `core_k0`

# Рис. 10 (данные Struchkov & Rogachev, 2017, визуальный метод): WAT раствора парафина в керосине при
# атмосферном давлении, (массовая доля, C) - маркеры у P = 0, оцифровка по сетке с растра 300 dpi (~0.2 C).
WAT_CONC = [(0.10, 25.4), (0.20, 30.0), (0.30, 33.5), (0.40, 40.2), (0.50, 43.0), (0.60, 45.8)]
# Там же регрессии авторов P = a*ln(WAT) - b (P в МПа, WAT в C) для 10, 30 и 60%: сдвиг WAT с давлением
WAT_P_FIT = {0.10: (175.75, 569.44), 0.30: (197.45, 693.84), 0.60: (207.79, 793.29)}


# --- A. Поры ---------------------------------------------------------------------------------------------

def kosugi(r, r_m, sigma):
    return np.exp(-0.5 * (np.log(r / r_m) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma * r)


def bins_volume(r, fi):
    """Доли порового объема (r^2 * fi) в классах рис. 6: диаметр d +- 5 мкм -> радиус d/2 +- 2.5 мкм."""
    dr = r[1] - r[0]
    return np.array([np.sum((r ** 2 * fi)[(r >= d / 2 - 2.5e-6 + 1e-12) & (r < d / 2 + 2.5e-6)]) * dr
                     for d in np.array(PORE_D_UM) * 1e-6])


def pores():
    """Kosugi по столбцам «до» (нормировка на сумму классов - мельче 20 мкм томограф не видит) и
    сравнение с fi_0 модели (r_m = 12 мкм, sigma_r = 0.4)."""
    r = np.linspace(0.05e-6, 60e-6, 4000)
    before = np.array(PORES_BEFORE) / np.sum(PORES_BEFORE)

    def shares(r_m, sigma):
        b = bins_volume(r, kosugi(r, r_m, sigma))
        return b / b.sum()

    res = least_squares(lambda x: shares(x[0] * 1e-6, x[1]) - before, [12.0, 0.4], bounds=([3, 0.1], [40, 1.5]))
    r_m, sigma = res.x[0] * 1e-6, res.x[1]
    field = shares(12e-6, 0.4)
    print(f'A. Kosugi по томографии: r_m = {r_m * 1e6:.1f} мкм, sigma_r = {sigma:.2f} '
          f'(в модели 12 мкм, 0.40); СКО долей: подбор {np.sqrt(np.mean(res.fun ** 2)):.3f}, '
          f'fi_0 модели {np.sqrt(np.mean((field - before) ** 2)):.3f}')
    print('| d, мкм | томография, % | подбор, % | fi_0 модели, % |')
    for d, b, f, m in zip(PORE_D_UM, before, shares(r_m, sigma), field):
        print(f'| {d:.0f} | {100 * b:.1f} | {100 * f:.1f} | {100 * m:.1f} |')
    return dict(r_m=r_m, sigma=sigma, before=before.tolist(), fit=shares(r_m, sigma).tolist(), field=field.tolist())


# --- B. WAT ------------------------------------------------------------------------------------------------

def tm_for_wat(wat):
    """Tm в (6.1), при котором раствор W_WAX начинает кристаллизоваться при `wat`, с дH авторов."""
    return brentq(lambda tm: float(cf.w_saturated(W_WAX, wat, MW_WAX, M_KEROSENE, tm, DH)) - W_WAX + 1e-9,
                  wat + 0.1, 200.0)


def wat():
    tm_bulk, tm_core = tm_for_wat(WAT_BULK), tm_for_wat(WAT_CORE)
    tm_melt = 52.0  # температура плавления парафина по авторам, если взять ее за Tm
    wat_melt = cf.cloud_point(W_WAX, MW_WAX, M_KEROSENE, tm_melt, DH)
    solids = {t: W_WAX - float(cf.w_saturated(W_WAX, t, MW_WAX, M_KEROSENE, tm_core, DH)) for t in (33.0, 32.8)}
    print(f'B. Tm = 52 C (плавление) -> WAT модели {wat_melt:.1f} C (опыт 30 и 33.8 C); '
          f'WAT = 30 C требует Tm = {tm_bulk:.1f} C, WAT = 33.8 C - Tm = {tm_core:.1f} C. '
          f'Взвеси при 33.0 / 32.8 C (WAT 33.8): {100 * solids[33.0]:.2f} / {100 * solids[32.8]:.2f}%')

    # Кривая растворимости по шести концентрациям: (6.1) с дH авторов и Tm по 20%, и подбор обоих
    w, t = np.array(WAT_CONC).T

    def model_wat(tm, dh):
        return np.array([cf.cloud_point(wi, MW_WAX, M_KEROSENE, tm, dh) for wi in w])

    fit = least_squares(lambda x: model_wat(x[0], x[1] * 1e3) - t, [tm_bulk, DH / 1e3])
    tm_fit, dh_fit = fit.x[0], fit.x[1] * 1e3
    with_dh = model_wat(tm_bulk, DH)
    print(f'B. WAT от концентрации: (6.1) с дH = 81.7 кДж/моль и Tm по 20% - СКО {np.sqrt(np.mean((with_dh - t) ** 2)):.1f} C; '
          f'подбор Tm = {tm_fit:.1f} C, дH = {dh_fit / 1e3:.1f} кДж/моль - СКО {np.sqrt(np.mean(fit.fun ** 2)):.1f} C')
    print('| w, % | WAT опыт, C | (6.1), дH авторов | (6.1), подбор |')
    for row in zip(w, t, with_dh, t + fit.fun):
        print(f'| {100 * row[0]:.0f} | {row[1]:.1f} | {row[2]:.1f} | {row[3]:.1f} |')

    # Давление: в (6.1) его нет, а WAT растет на 0.15-0.25 C/МПа
    shift = {c: [float(np.exp((p + b) / a) - np.exp((0.1 + b) / a)) for p in (10.0, 20.0, 35.0)]
             for c, (a, b) in WAT_P_FIT.items()}
    print('B. Рост WAT с давлением (регрессии авторов), C при 10 / 20 / 35 МПа: ' +
          '; '.join(f'{100 * c:.0f}%: ' + ' / '.join(f'{s:.1f}' for s in v) for c, v in shift.items()))
    return dict(tm_bulk=tm_bulk, tm_core=tm_core, wat_melt=wat_melt, solids_33=solids[33.0], solids_328=solids[32.8],
                conc=w.tolist(), wat_exp=t.tolist(), wat_dh=with_dh.tolist(), wat_fit=(t + fit.fun).tolist(),
                tm_fit=tm_fit, dh_fit=dh_fit, pressure_shift={str(c): v for c, v in shift.items()})


# --- C. Охлаждение керна ------------------------------------------------------------------------------------

def core_k0(r_m, sigma):
    """k0 пучка капилляров с извилистостью керна Berea (5.24), [м^2]. Измеренный перепад (0.82 МПа/м при
    0.5 см^3/мин) дает k0 ~ 0.03 мД - на три порядка меньше, чем у пучка с порами 10-30 мкм, и скорость в
    капилляре по нему была бы нефизична (извилистость ~400). k/k0 нормирован и от этого выбора зависит слабо."""
    r = np.linspace(0.0, 40e-6 * r_m / 12e-6, 31)[1:]
    fi = kosugi(r, r_m, sigma)
    return CORE_M * np.sum(r ** 4 * fi) / (8.0 * BEREA_ETA ** 2 * np.sum(r ** 2 * fi))


def core_runs(pore_fit, wat_res):
    r_m, sigma = pore_fit['r_m'], pore_fit['sigma']
    pores_extra = {'r_m': repr(r_m), 'sigma_r': repr(sigma), 'r_max': repr(40e-6 * r_m / 12e-6)}
    area = np.pi * CORE_D ** 2 / 4.0
    pv0 = CORE_L * area * CORE_M
    base = dict(
        title='wax in kerosene, sandstone (Sandyga et al., 2020)',
        k0=core_k0(r_m, sigma) / DARCY, porosity=CORE_M, length=CORE_L, side=float(np.sqrt(area)), q=Q,
        T=T_START, T_end=T_END, cooling=COOLING, pv_end=Q * (T_START - T_END) / COOLING / pv0, plugged=1e-3,
        w=W_WAX, MW=MW_WAX, M_o=M_KEROSENE, dH=DH, ro_o=RO_SOLUTION, ro_p=RO_WAX, mu=MU_40,
        exp=[],  # кривой k/k0(PV) в опыте нет - сравнение по градиенту от температуры
    )
    print(f'C. k0 пучка = {base["k0"] * 1e3:.0f} мД, {base["pv_end"]:.0f} PV за охлаждение {T_START} -> {T_END} C')
    # d_p, L_k - калибровка по опыту 1 Sutton & Roberts (`core_flood.py`, outputs/data/core_flood.json)
    calibration = json.loads(cf.RESULT.read_text(encoding='utf-8'))['best']['1']
    d_cal, lk = calibration['d_p'], calibration['lk']
    print(f'C. d_p = {d_cal * 1e6:.1f} мкм, L_k = {lk * 1e3:.2f} мм - калибровка по Sutton & Roberts')
    cases = {
        'WAT 30 C (реометр)': (wat_res['tm_bulk'], d_cal),
        'WAT 33.8 C (керн)': (wat_res['tm_core'], d_cal),
        'WAT 33.8 C (керн), d_p 5 мкм': (wat_res['tm_core'], 5e-6),
    }
    t_exp, g_exp = np.array(GRADIENT).T
    ratio_exp = g_exp / g_exp[0]
    runs = {}
    print('| Вариант | grad/grad0 при 35 / 34 / 33.5 / 33 / 32.8 C | T начала роста (x2), C | m/m0 | '
          'СКО lg(grad/grad0) |')
    for name, (tm, d_p) in cases.items():
        res = cf.run_case(4, d_p, lk, dt=2.0, exp=dict(base, Tm=tm), extra=pores_extra, mode='ramp')
        t_hist, k_app = np.array(res['T_hist']), np.array(res['k'])
        if res['plugged'] is not None:  # драйвер дописывает точку k = 0 после закупорки - в сравнении ее нет
            t_hist, k_app = t_hist[:-1], k_app[:-1]
        ratio = 1.0 / k_app  # при постоянном расходе grad/grad0 = 1/k_app
        order = np.argsort(t_hist, kind='stable')           # T убывает по шагам, интерполяция - по возрастанию
        model_at = np.interp(t_exp, t_hist[order], ratio[order])
        onset = float(t_hist[np.argmax(ratio > 2.0)]) if np.any(ratio > 2.0) else None
        err = float(np.sqrt(np.mean((np.log10(model_at) - np.log10(ratio_exp)) ** 2)))
        m_mean = float(np.mean(res['m_profile']))
        r = np.array(res['r'])
        after = bins_volume(r, np.mean(np.array(res['fi_end']), axis=0)) / bins_volume(r, np.array(res['fi0'])).sum()
        at = ' / '.join(f'{np.interp(x, t_hist[order], ratio[order]):.1f}' for x in (35.0, 34.0, 33.5, 33.0, 32.8))
        print(f'| {name} | {at} | {"—" if onset is None else f"{onset:.2f}"} | {m_mean:.3f} | {err:.2f} |', flush=True)
        idx = np.unique(np.linspace(0, len(t_hist) - 1, 400).astype(int))
        runs[name] = dict(T=t_hist[idx].tolist(), grad_ratio=ratio[idx].tolist(), onset=onset, m_mean=m_mean,
                          err_log=err, bins_after=after.tolist(), plugged=res['plugged'],
                          m_profile=res['m_profile'], k_profile=res['k_profile'], eta=res['eta'])
    print('Опыт: grad/grad0 при 35 / 34 / 33.5 / 33 / 32.8 C = '
          + ' / '.join(f'{np.interp(-x, -t_exp, ratio_exp):.1f}' for x in (35.0, 34.0, 33.5, 33.0, 32.8))
          + f'; m/m0 = {POROSITY_AFTER / CORE_M:.3f}')
    return runs


def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    t_exp, g_exp = np.array(GRADIENT).T
    ax1.semilogy(t_exp, g_exp / g_exp[0], 'ko', mfc='none', label='Sandyga et al., рис. 4')
    for (name, run), style in zip(data['runs'].items(), ('k:', 'k-', 'k--')):
        ax1.semilogy(run['T'], run['grad_ratio'], style, label=name)
    ax1.invert_xaxis()
    ax1.set_xlabel('T, °C')
    ax1.set_ylabel('∇p / ∇p(40 °C)')
    ax1.legend(fontsize=7)
    ax1.set_title('а) охлаждение 1 °C/ч, 0.5 см³/мин')

    x = np.arange(len(PORE_D_UM))
    ax2.bar(x - 0.3, PORES_BEFORE, 0.2, color='0.8', edgecolor='k', label='до (томография)')
    ax2.bar(x - 0.1, PORES_AFTER, 0.2, color='0.4', edgecolor='k', label='после (томография)')
    for (name, run), shift, hatch in zip(list(data['runs'].items())[1:], (0.1, 0.3), ('//', '..')):
        ax2.bar(x + shift, 100 * np.array(run['bins_after']), 0.2, color='w', edgecolor='k', hatch=hatch,
                label='модель: ' + name)
    ax2.set_xticks(x, [f'{d:.0f}' for d in PORE_D_UM])
    ax2.set_xlabel('диаметр горла, мкм')
    ax2.set_ylabel('% исходного порового объема')
    ax2.legend(fontsize=7)
    ax2.set_title('б) какие поры закупорены')
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f'Рисунок: {FIG}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if sys.argv[1:2] == ['--plot']:
        plot(json.loads(OUT.read_text(encoding='utf-8')))
        return
    pore_fit = pores()
    wat_res = wat()
    runs = core_runs(pore_fit, wat_res)
    data = dict(pores=pore_fit, wat=wat_res, runs=runs, GRADIENT=GRADIENT)
    OUT.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    print(f'Записано: {OUT}')
    plot(data)


if __name__ == '__main__':
    main()
