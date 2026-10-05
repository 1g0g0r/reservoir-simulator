"""Построение рисунков статьи и сбор всех чисел для текста и таблиц.

Единственная точка входа: перестроить рисунки после нового расчета - запустить этот файл.
Ничего править не нужно; состав вариантов задан в `graphs.ARTICLE_CASES`.

    python твт_статья/make_figures.py

Результат: `твт_статья/figures/*.tif` (600 dpi, для журнала) и `*.png` (для просмотра),
`твт_статья/metrics.md` - все числа, на которые ссылается текст статьи.
"""
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from paraphin import r, fi_0
from paraphin.constants import day_to_sec, geological_reserves, init_Wp
from paraphin.utils.visualisation_utils import read_solution_data, x_mesh, y_mesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'исходная_модель'))  # core_flood.py - проверка по керну

# (ключ, файл данных, номер в подписи, стиль линии, подпись варианта)
# Имена файлов - явные литералы, а не f'Wp={constants.init_Wp}...': та константа - лишь
# отправная точка для `run_cases.py`, после прогона её значение восстанавливается на исходное
# и не совпадает с тем, чем на самом деле считалась статья. Через неё 'base'/'heat_nowax' и
# 'noheat'/'nowax' легко схлопнутся в один и тот же файл (Wp=0.0), если в constants.py на
# момент запуска этого скрипта не то значение - см. ARTICLE_WP ниже, тот же принцип.
ARTICLE_CASES = (
    ('base',      f'Wp={init_Wp}_processed_data.pkl',           1, '-',  f'с теплопотерями (Винсом-Вестервельд), Wp = {init_Wp}'),
    ('noheat',    f'Wp={init_Wp}_noheat_processed_data.pkl',    2, '--', f'без теплопотерь, Wp = {init_Wp}'),
    ('nowax',     'Wp=0.0_noheat_processed_data.pkl',    3, ':',  'без теплопотерь, Wp = 0'),
    ('lauwerier', f'Wp={init_Wp}_lauwerier_processed_data.pkl', 4, '-.', f'с теплопотерями (Ловерье), Wp = {init_Wp}'),
    ('heat_nowax', 'Wp=0.0_processed_data.pkl',           5, (0, (6, 1, 1, 1, 1, 1)), 'с теплопотерями (Винсом-Вестервельд), Wp = 0'),
)

# Маркеры вариантов на графиках: линии одного цвета различаются типом штриха, но на
# печати тонкий штрих и точки сливаются, поэтому каждая кривая несет еще и свой маркер.
MARKERS = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}
LINE_WIDTH = 2.0

TITLES = {key: title for key, _f, _n, _s, title in ARTICLE_CASES}
NUMBERS = {key: number for key, _f, number, _s, _t in ARTICLE_CASES}
FILES = {key: file_name for key, file_name, _n, _s, _t in ARTICLE_CASES}


def _delta(value: float, base: float) -> str:
    """Отклонение от базового варианта: абсолютное и относительное."""
    if base == 0.0:
        return '—'
    return f'{value - base:+.4g} ({100.0 * (value - base) / base:+.1f}%)'


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    cases = create_article_figures()
    if not cases:
        print('Расчетов нет — сначала python твт_статья/run_cases.py')
        return

    base = cases.get('base')
    lines = ['# Числа для текста статьи', '',
             'Файл создается автоматически: `python твт_статья/make_figures.py`.',
             'Вручную не править — правки затрутся при следующей перегенерации.', '']

    lines += ['## Показатели вариантов', '',
              '| № | Вариант | t₁ (прорыв), сут | t₂, сут | t₃ (η = 98%), сут | КИН | Накопл. нефть, м³ | '
              'Средняя T на t₃, °C | η на конце |',
              '|---|---|---|---|---|---|---|---|---|']
    for key in cases:
        m = cases[key]
        flag = '' if m['reached_limit'] else ' ⚠ не достиг 98%'
        lines.append(f'| {NUMBERS[key]} | {TITLES[key]} | {m["t"][0]:.0f} | {m["t"][1]:.0f} | '
                     f'{m["t"][2]:.0f} | {m["KIN"]:.4f} | {m["Q_oil"]:.0f} | {m["mean_T_end"]:.1f} | '
                     f'{m["eta_end"]:.4f}{flag} |')

    if base is not None:
        lines += ['', '## Отклонения от базового варианта (вариант 1)', '',
                  '| Вариант | Δt₁ | Δt₃ | ΔКИН | ΔQ_нефти |', '|---|---|---|---|---|']
        for key, m in cases.items():
            if key == 'base':
                continue
            lines.append(f'| {NUMBERS[key]}. {TITLES[key]} | {_delta(m["t"][0], base["t"][0])} | '
                         f'{_delta(m["t"][2], base["t"][2])} | {_delta(m["KIN"], base["KIN"])} | '
                         f'{_delta(m["Q_oil"], base["Q_oil"])} |')

        # Полный план 2x2 по двум факторам: теплообмен (есть/нет) и парафин (есть/нет).
        # Только он позволяет отделить главные эффекты от взаимодействия: разложение через три
        # варианта - тождество, а не результат.
        if all(k in cases for k in ('base', 'noheat', 'heat_nowax', 'nowax')):
            hw = cases['base']        # теплообмен есть, парафин есть
            w = cases['noheat']       # теплообмена нет, парафин есть
            h = cases['heat_nowax']   # теплообмен есть, парафина нет
            n = cases['nowax']        # нет ни того, ни другого

            lines += ['', '## Разложение эффектов по плану 2x2', '',
                      '| Показатель | Теплообмен при парафине (1−2) | Теплообмен без парафина (5−3) | '
                      'Парафин при теплообмене (1−5) | Парафин без теплообмена (2−3) | Взаимодействие |',
                      '|---|---|---|---|---|---|']
            for key, name, fmt in (('KIN', 'КИН', '{:+.4f}'), ('t3', 't₃, сут', '{:+.0f}')):
                def val(case, _key=key):
                    return case['KIN'] if _key == 'KIN' else case['t'][2]

                e_heat_w, e_heat_n = val(hw) - val(w), val(h) - val(n)
                e_wax_h, e_wax_n = val(hw) - val(h), val(w) - val(n)
                lines.append(f'| {name} | {fmt.format(e_heat_w)} | {fmt.format(e_heat_n)} | '
                             f'{fmt.format(e_wax_h)} | {fmt.format(e_wax_n)} | '
                             f'{fmt.format(e_heat_w - e_heat_n)} |')

            total = hw['KIN'] - n['KIN']
            main_sum = (h['KIN'] - n['KIN']) + (w['KIN'] - n['KIN'])
            lines += ['', f'- суммарное изменение КИН (вар. 1 − вар. 3): {total:+.4f}',
                      f'- сумма главных эффектов: {main_sum:+.4f}',
                      f'- остаток на взаимодействие: {total - main_sum:+.4f} '
                      f'({100.0 * abs(total - main_sum) / abs(total):.1f}% суммарного изменения)']

    lines += text_numbers(cases)
    lines += core_flood_numbers()

    (Path(__file__).parent / 'metrics.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Числа записаны в твт_статья/metrics.md')


def _diag_positions() -> np.ndarray:
    """Расстояния вдоль главной диагонали (0,0)-(Lx,Ly) до каждого узла сетки, м."""
    return x_mesh * np.sqrt(2.0)


def _front_position(field_2d: np.ndarray, level: float):
    """Первое пересечение `level` вдоль главной диагонали, считая от узла (0,0), м.

    Линейная интерполяция между соседними узлами диагонали `field_2d[k, k]`.
    """
    diag = np.diagonal(field_2d)
    pos = _diag_positions()
    for k in range(len(diag) - 1):
        a, b = diag[k] - level, diag[k + 1] - level
        if a == 0.0:
            return float(pos[k])
        if a * b < 0.0:
            return float(pos[k] + a / (a - b) * (pos[k + 1] - pos[k]))
    return float('nan')


def text_numbers(cases: dict) -> list:
    """Числа описательного текста результатов, которых нет в агрегированных таблицах выше.

    В отличие от `case_metrics`, требует самих полей `(Nx, Ny)`, а не только скалярных
    показателей, поэтому читает файлы расчета заново.
    """
    from scipy.optimize import brentq
    from paraphin.constants import Tm, Tm_K, alpha, R, MW, M_o, Twater, phi_max as PHI_MAX

    lines = ['', '## Дополнительные числа из описательного текста', '']
    if 'base' not in cases:
        return lines

    alpha_r = alpha / R

    def _w_hat(t_celsius: float) -> float:
        """Равновесная мольная доля (6.1), переведенная в массовую (6.2)."""
        x_sat = min(1.0, np.exp(-alpha_r * (1.0 / (t_celsius + 273.15) - 1.0 / Tm_K)))
        return x_sat * MW / (x_sat * MW + (1.0 - x_sat) * M_o)

    # Порог начала кристаллизации T* при заданном wp: температура, при которой предел
    # растворимости (6.2) сравнивается с wp - см. текст перед (17).
    t_onset = brentq(lambda t: _w_hat(t) - init_Wp, -50.0, Tm)

    m1 = cases['base']
    idx1, idx2, idx3 = m1['idx']
    _, d1 = read_solution_data(FILES['base'])

    def kolm_min(idx):
        k_field = d1['k'][idx]
        i, j = np.unravel_index(int(np.argmin(k_field)), k_field.shape)
        return float(x_mesh[i]), float(y_mesh[j]), float(k_field[i, j]), float(d1['m'][idx][i, j])

    mean_t = [float(d1['Temperature'][idx].mean()) for idx in (idx1, idx2, idx3)]
    s_front = [_front_position(d1['Saturation'][idx], 0.5) for idx in (idx1, idx2, idx3)]
    t_front = [_front_position(d1['Temperature'][idx], t_onset) for idx in (idx1, idx2, idx3)]
    # Внешняя граница зоны кольматации: где множитель проницаемости еще отличим от единицы
    k_front = [_front_position(d1['k'][idx], 0.99) for idx in (idx1, idx2, idx3)]
    p_range = [(float(d1['Pressure'][idx].min()) * 1e-6, float(d1['Pressure'][idx].max()) * 1e-6)
               for idx in (idx1, idx3)]
    kolm = [kolm_min(idx) for idx in (idx1, idx2, idx3)]
    t3_range1 = (float(d1['Temperature'][idx3].min()), float(d1['Temperature'][idx3].max()))
    wps_max = float(d1['Wps'].max())
    from paraphin.utils.math_utils.fluids_correlations import crystal_volume_fraction
    phi = min(crystal_volume_fraction(wps_max), 0.99 * PHI_MAX)
    kd_mult = (1.0 - phi / PHI_MAX) ** (-2.5 * PHI_MAX)

    # Доля площади элемента, охлажденной ниже порога кристаллизации T* - охват зоны, где вообще
    # возможно выпадение парафина (см. текст перед и после (17)); площадной аналог диагонального
    # фронта t_front, нужен там, где текст говорит о доле площади, а не о положении изотермы.
    cold_area = [float((d1['Temperature'][idx] < t_onset).mean()) * 100.0 for idx in (idx1, idx2, idx3)]
    # Площадной аналог k_front: доля площади со сниженной проницаемостью (k/k0<0.99) на t3.
    kolm_area_t3 = float((d1['k'][idx3] < 0.99).mean()) * 100.0
    # Водонасыщенность в охлажденной зоне на t3 - показывает, что промытая зона не является
    # строго остаточной (нефть еще подвижна, см. текст после (13)).
    s_cooled_t3 = d1['Saturation'][idx3][d1['Temperature'][idx3] < t_onset]
    # Момент и положение глобального максимума взвешенного парафина по всему расчету - "первые
    # сутки закачки у забоя нагнетательной скважины" из описательного текста.
    wps_t_idx, wps_i, wps_j = np.unravel_index(int(np.argmax(d1['Wps'])), d1['Wps'].shape)
    wps_max_day = float(m1['time'][wps_t_idx])
    wps_max_xy = (float(x_mesh[wps_i]), float(y_mesh[wps_j]))
    wps_t1_t3 = [float(d1['Wps'][idx].max()) for idx in (idx1, idx3)]

    lines += [
        f'- порог начала кристаллизации T* при wp={init_Wp}: {t_onset:.1f}°C (Tm={Tm:.1f}°C); '
        f'предел растворимости при T={Twater:.0f}°C: {_w_hat(Twater):.3f}',
        f'- средняя температура пласта (вариант 1): t1 {mean_t[0]:.1f}°C, t2 {mean_t[1]:.1f}°C, '
        f't3 {mean_t[2]:.1f}°C',
        f'- обводненность на t2: {m1["eta"][idx2]:.3f}',
        f'- фронт S=0.5 по диагонали: t1 {s_front[0]:.1f} м, t2 {s_front[1]:.1f} м, t3 {s_front[2]:.1f} м',
        f'- фронт T=T* по диагонали: t1 {t_front[0]:.1f} м, t2 {t_front[1]:.1f} м, t3 {t_front[2]:.1f} м',
        f'- граница зоны кольматации (k/k0=0.99) по диагонали: t1 {k_front[0]:.1f} м, '
        f't2 {k_front[1]:.1f} м, t3 {k_front[2]:.1f} м',
        f'- давление, МПа: t1 {p_range[0][0]:.2f}-{p_range[0][1]:.2f}, t3 {p_range[1][0]:.2f}-{p_range[1][1]:.2f}',
        f'- максимум дебита нефти: {m1["q_oil_max"]:.1f} м3/сут на {m1["q_oil_max_day"]:.0f} сут; '
        f'дебит на t3: {m1["q_oil_end"]:.1f} м3/сут',
        '- минимум k/k0 (точка, м) и m/m0 там же: '
        + '; '.join(f't{n+1} {v:.3f} в ({x:.0f},{y:.0f}), m/m0={mm:.3f}'
                     for n, (x, y, v, mm) in enumerate(kolm)),
        f'- диапазон температуры на t3 (вариант 1): {t3_range1[0]:.1f}-{t3_range1[1]:.1f}°C',
        f'- максимум массовой доли взвешенного парафина: {wps_max:.4f}, '
        f'множитель Кригера-Догерти при этом {kd_mult:.3f}',
        f'- площадь с T<T* (вариант 1): t1 {cold_area[0]:.1f}%, t2 {cold_area[1]:.1f}%, '
        f't3 {cold_area[2]:.1f}%',
        f'- площадь с k/k0<0.99 на t3 (вариант 1): {kolm_area_t3:.1f}%',
        f'- водонасыщенность в охлажденной зоне на t3 (вариант 1): мин {float(s_cooled_t3.min()):.3f}, '
        f'средняя {float(s_cooled_t3.mean()):.3f}',
        f'- максимум wps по всему расчету (вариант 1): {wps_max:.4f} на {wps_max_day:.0f} сут '
        f'в точке ({wps_max_xy[0]:.0f},{wps_max_xy[1]:.0f}) м; wps на t1 {wps_t1_t3[0]:.4f}, '
        f'на t3 {wps_t1_t3[1]:.4f}',
    ]

    if 'noheat' in cases:
        m2 = cases['noheat']
        idx1_2, idx2_2, idx3_2 = m2['idx']
        picks = [int(np.argmin(np.abs(m2['time'] - t))) for t in m1['t']]
        _, d2 = read_solution_data(FILES['noheat'])

        mean_t2 = [float(d2['Temperature'][p].mean()) for p in picks]
        t3_range2 = (float(d2['Temperature'][picks[2]].min()), float(d2['Temperature'][picks[2]].max()))
        # Площадь с T<T* и максимум wps на СВОЕМ (не выровненном по варианту 1) конце расчета:
        # вариант без теплообмена работает дольше и остывает сильнее к своему собственному t3.
        cold_area2_t3 = float((d2['Temperature'][idx3_2] < t_onset).mean()) * 100.0
        # Кольматация без теплообмена: парафин выпадает впереди фронта вытеснения, в подвижной нефти
        kmin2_t3 = float(d2['k'][idx3_2].min())
        kolm_area2_t3 = float((d2['k'][idx3_2] < 0.99).mean()) * 100.0
        kolm_area2_t2 = float((d2['k'][idx2_2] < 0.99).mean()) * 100.0
        wps_mean2 = [float(d2['Wps'][p].mean()) for p in (idx1_2, idx2_2, idx3_2)]
        wps_max2 = float(d2['Wps'].max())
        phi2 = min(crystal_volume_fraction(wps_max2), 0.99 * PHI_MAX)
        kd_mult2 = (1.0 - phi2 / PHI_MAX) ** (-2.5 * PHI_MAX)

        dT_mean, dT_max, dS_max = [], [], []
        for own_idx, other_idx in zip((idx1, idx2, idx3), picks):
            dT = d1['Temperature'][own_idx] - d2['Temperature'][other_idx]
            dS = d1['Saturation'][own_idx] - d2['Saturation'][other_idx]
            dT_mean.append(float(dT.mean()))
            dT_max.append(float(dT.max()))
            dS_max.append(float(dS.max()))
        del d2

        lines += [
            f'- средняя температура (вариант 2, без теплообмена): t1 {mean_t2[0]:.1f}°C, '
            f't2 {mean_t2[1]:.1f}°C, t3 {mean_t2[2]:.1f}°C',
            f'- диапазон температуры на t3 (вариант 2): {t3_range2[0]:.1f}-{t3_range2[1]:.1f}°C',
            '- разность полей (вариант 1 − вариант 2), средняя/максимальная T (°C) и максимальная S: '
            + '; '.join(f't{n+1} dT_mean={a:.1f} dT_max={b:.1f} dS_max={c:.3f}'
                         for n, (a, b, c) in enumerate(zip(dT_mean, dT_max, dS_max))),
            f'- площадь с T<T* на своем t3 (вариант 2): {cold_area2_t3:.1f}%',
            f'- кольматация (вариант 2): минимум k/k0 на своем t3 {kmin2_t3:.3f}; площадь k/k0<0.99 на t2 '
            f'{kolm_area2_t2:.1f}%, на t3 {kolm_area2_t3:.1f}%; средняя wps на t1/t2/t3 '
            + '/'.join(f'{v:.4f}' for v in wps_mean2),
            f'- максимум wps по всему расчету (вариант 2): {wps_max2:.4f}, '
            f'множитель Кригера-Догерти при этом {kd_mult2:.3f}',
        ]

    if 'heat_nowax' in cases:
        m5 = cases['heat_nowax']
        idx3_5 = m5['idx'][2]
        _, d5 = read_solution_data(FILES['heat_nowax'])
        p_max5_t3 = float(d5['Pressure'][idx3_5].max()) * 1e-6
        del d5

        lines += [
            f'- максимум давления на своем t3 (вариант 5): {p_max5_t3:.2f} МПа '
            f'(вариант 1: {p_range[1][1]:.2f} МПа, разница {abs(p_max5_t3 - p_range[1][1]):.2f} МПа)',
        ]

    del d1
    return lines



# ======================================================================================
# Рисунки статьи в ТВТ
#
# Отдельная секция, а не правка `create_graphs_and_maps`: та строит рисунки исходной статьи
# (сравнение двух расчетов в plotly и svg), а журнал требует черно-белую графику и TIF 600 dpi,
# которого kaleido не умеет. Здесь matplotlib - он уже есть в пакете (`save_in_eps_format`),
# пишет TIF напрямую и дает общий colorbar на столбец панели 3x3.
#
# Перегенерация: python твт_статья/make_figures.py
# ======================================================================================

# На рисунок идут только три основных варианта: пять кривых по двум величинам нечитаемы,
# а варианты 4 и 5 нужны для таблиц (проверка приближения и аддитивности эффектов).
FIGURE_CASES = ('base', 'noheat', 'nowax')

# Вся графика статьи черно-белая: журнал печатает в одну краску, цветные иллюстрации
# оплачивает автор (правило 8.5), а TIF требуется в 256 оттенках серого (правило 8.10).
# Поля показаны изолиниями с подписанными значениями: подпись на линии читается сразу.
# Там, где изолиний мало или они сбиваются в узкую полосу (насыщенность, давление,
# множители ФЕС, разности полей), под них кладется заливка оттенками серого - иначе
# панель почти пуста. Заливка ограничена светлыми тонами (см. `_fill`), чтобы подписи
# изолиний оставались читаемыми. Варианты на графиках различаются типом линии и маркером, но не цветом.
#
# Строки панели 3x3 - моменты времени, столбцы - величины:
# (ключ поля, подпись, множитель, заливка)
MAP_COLUMNS = (
    ('Temperature', 'T, °C',  1.0,  True),
    ('Saturation',  'S',      1.0,  True),
    ('Pressure',    'p, МПа', 1e-6, True),
)

# Карты кольматации: множители проницаемости и пористости
COLMATATION_COLUMNS = (
    ('k', 'k/k₀', 1.0, True),
    ('m', 'm/m₀', 1.0, True),
)

PANEL_LETTERS = 'абвгдежзи'
LETTERS_POSITION = [0.03, 0.97]


def _article_figures_path():
    from paraphin.constants import root_folder
    path = root_folder / 'твт_статья' / 'figures'
    path.mkdir(parents=True, exist_ok=True)
    return path


def case_metrics(data: dict) -> dict:
    """Показатели одного расчета: характерные времена, КИН, накопленная добыча.

    `t1` - прорыв воды (первый слой с ненулевой обводненностью), `t3` - конец расчета
    (останов по eta >= max_eta), `t2` - ближайший сохраненный слой к середине интервала.
    КИН считается так же, как в `create_graphs_and_maps`: накопленная нефть к запасам.
    """
    from paraphin.constants import max_eta

    time_days = data['Time'] / day_to_sec
    eta = data['Wells']['Producer_eta']
    q_oil = np.abs(data['Wells']['Producer_oil']) * day_to_sec
    Q_oil = np.abs(data['Wells_accumulated']['Producer_Q_oil'])

    idx1 = int(np.argwhere(eta > 0.0)[0][0])
    idx3 = len(time_days) - 1
    idx2 = int(np.argmin(np.abs(time_days - 0.5 * (time_days[idx1] + time_days[idx3]))))

    return {
        'idx': (idx1, idx2, idx3),
        't': (time_days[idx1], time_days[idx2], time_days[idx3]),
        'eta_end': float(eta[idx3]),
        'reached_limit': bool(eta[idx3] >= max_eta),
        'KIN': float(Q_oil[idx3] / geological_reserves),
        'Q_oil': float(Q_oil[idx3]),
        'q_oil_max': float(q_oil.max()),
        'q_oil_max_day': float(time_days[int(np.argmax(q_oil))]),
        'q_oil_end': float(q_oil[idx3]),
        'mean_T_end': float(data['Temperature'][idx3].mean()),
        'time': time_days,
        'eta': eta,
        'q_oil': q_oil,
        'Q_oil_curve': Q_oil,
    }


def _fmt(span: float) -> str:
    """Формат подписи изолинии по размаху величины на панели.

    Число знаков подбирается так, чтобы соседние изолинии не получили одинаковую
    подпись: у температуры размах в десятки градусов, у множителя проницаемости - сотые.
    """
    if span >= 5.0:
        return '%.0f'
    return '%.2f' if span >= 0.05 else '%.3f'


def _label_all_levels(ax, cs, levels, fmt: str, x_mesh, y_mesh,
                       fontsize: float = 7, min_frac: float = 0.05) -> None:
    """Гарантирует подпись у каждого уровня, включая те, что clabel пропустил.

    Автоматическая расстановка inline-меток пропускает петли короче подписи - в панели
    3x3 (~2.3 дюйма на подрисунок) так теряются уровни у самой скважины, где изолинии
    стягиваются в тесный контур. Для пропущенных уровней подпись ставится вручную поверх
    середины самого длинного сегмента этого уровня (`ax.text`, а не второй вызов
    `clabel` - он трогает внутреннее состояние `cs` и падает на несуществующем индексе).

    У скважины петля иногда стягивается в точку меньше самой подписи (депрессионная
    воронка давления) - туда подпись ставить некуда, кроме как поверх значка скважины.
    Такие уровни (диагональ петли меньше `min_frac` диагонали панели) остаются без
    подписи - их значение и так видно по цвету заливки на общей шкале.
    """
    placed = ax.clabel(cs, inline=True, inline_spacing=2, fontsize=fontsize, fmt=fmt)
    present = {t.get_text() for t in placed}
    diag = np.hypot(x_mesh[-1] - x_mesh[0], y_mesh[-1] - y_mesh[0])
    for level, segs in zip(levels, cs.allsegs):
        label = fmt % level
        if label in present or not segs:
            continue
        longest = max(segs, key=len)
        bbox_diag = np.hypot(longest[:, 0].max() - longest[:, 0].min(), longest[:, 1].max() - longest[:, 1].min())
        if bbox_diag < min_frac * diag:
            continue
        x, y = longest[len(longest) // 2]
        ax.text(x, y, label, fontsize=fontsize, ha='center', va='center',bbox=dict(facecolor='none', edgecolor='none', pad=0.5))


def _isolines(ax, x_mesh, y_mesh, field, levels, fmt: str) -> None:
    """Изолинии с подписанными значениями - основной способ показать поле в статье."""
    cs = ax.contour(x_mesh, y_mesh, field.T, levels=levels, colors='k', linewidths=1.0)
    _label_all_levels(ax, cs, levels, fmt, x_mesh, y_mesh)


def _fill(ax, x_mesh, y_mesh, field, vmin: float, vmax: float, n: int = 32):
    """Заливка поля оттенками серого под изолинии.

    Самый темный тон - 0.55 от белого: на нем еще читаются черные изолинии и их
    подписи, а на печати в одну краску соседние ступени не сливаются в пятно.
    """
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('light_greys', ['1.0', '0.55'])
    levels = np.linspace(vmin, vmax, n)
    return ax.contourf(x_mesh, y_mesh, field.T, levels=levels, cmap=cmap, antialiased=False)


def _zero_line(ax, x_mesh, y_mesh, diff, lim: float) -> None:
    """Жирная нулевая изолиния разности - граница областей разного знака.

    Там, где оба варианта еще не отличаются (разность на уровне округления), contour
    рисовал бы вместо линии шумную сетку. Значения ниже 0.1% размаха относятся к
    отрицательной стороне: линия проходит только там, где разность действительно
    меняет знак.
    """
    tol = 1e-3 * lim
    clean = np.where(np.abs(diff) < tol, -tol, diff)
    cs = ax.contour(x_mesh, y_mesh, clean.T, levels=[0.0], colors='k', linewidths=1.8)
    _label_all_levels(ax, cs, [0.0], '%.0f', x_mesh, y_mesh)


def _save(fig, name: str) -> None:
    """Сохранение рисунка в TIF 600 dpi (требование журнала) и в PNG для просмотра.

    TIF пересохраняется через PIL в 8 бит на пиксель: правило 8.10 требует 256 оттенков
    серого, а matplotlib пишет RGB даже когда на рисунке нет ни одного цветного пикселя.
    """
    from PIL import Image

    path = _article_figures_path()
    tif = path / f'{name}.tif'
    fig.savefig(tif, dpi=600, format='tiff',
                pil_kwargs={'compression': 'tiff_lzw'}, bbox_inches='tight')
    with Image.open(tif) as img:
        gray = img.convert('L')
    gray.save(tif, compression='tiff_lzw')
    fig.savefig(path / f'{name}.png', dpi=200, bbox_inches='tight')


def _marker(number: int, n_points: int, color: str = 'k', per_curve: int = 12) -> dict:
    """Параметры маркера кривой: свой символ у каждого варианта, ~12 маркеров на кривую.

    Смещение первого маркера зависит от номера варианта: у совпадающих кривых маркеры
    иначе легли бы друг на друга.
    """
    step = max(1, n_points // per_curve)
    return dict(marker=MARKERS[number], markevery=(number * step // 5, step), markersize=6,
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.3)


def _wells(ax, x_mesh, y_mesh) -> None:
    """Отметки скважин. clip_on=False - они стоят точно в углах и иначе обрезаются рамкой."""
    ax.plot(x_mesh[0], y_mesh[0], marker='v', color='k', markersize=8, clip_on=False,
            markerfacecolor='white', markeredgewidth=1.3)
    ax.plot(x_mesh[-1], y_mesh[-1], marker='o', color='k', markersize=8, clip_on=False,
            markerfacecolor='white', markeredgewidth=1.3)


def field_panel(data: dict, metrics: dict, columns, name: str, figsize=(7.0, 7.4),
                transpose: bool = False, n_levels: int = 9) -> None:
    """Панель изолиний: строки - моменты t1, t2, t3; столбцы - величины из `columns`.

    Шкала общая по величине (панели сопоставимы между моментами времени), пределы осей
    одинаковы во всех панелях, скважины отмечены, панели пронумерованы русскими буквами.

    transpose меняет оси панели местами: величины по строкам, моменты по столбцам. Это нужно,
    когда величин мало: узкая и высокая панель, растянутая на ширину полосы, съедает почти
    целую страницу, а развернутая - вдвое меньше.
    """

    idxs = metrics['idx']
    times = metrics['t']
    n_col = len(columns)

    # constrained_layout вместо tight_layout: только он умеет резервировать место под
    # colorbar, приделанный к нескольким Axes сразу (легенда заливки ниже) - на tight_layout
    # колонка с колонками с заливкой заезжала на соседние панели.
    if transpose:
        fig, axes = plt.subplots(n_col, 3, figsize=figsize, sharex=True, sharey=True,
                                  squeeze=False, constrained_layout=True)
    else:
        fig, axes = plt.subplots(3, n_col, figsize=figsize, sharex=True, sharey=True,
                                  squeeze=False, constrained_layout=True)

    for col, (field_name, label, mult, fill) in enumerate(columns):
        stack = [data[field_name][i] * mult for i in idxs]
        vmin = min(f.min() for f in stack)
        vmax = max(f.max() for f in stack)
        # Крайние уровни совпадают с экстремумами поля и вырождаются в точку или в рамку,
        # поэтому из равномерной сетки уровней берется середина. Уровни общие на все три
        # момента времени: панели столбца сопоставимы между собой.
        levels = np.linspace(vmin, vmax, n_levels)[1:-1]
        fmt = _fmt(vmax - vmin)
        im = None

        for row, field in enumerate(stack):
            ax = axes[col, row] if transpose else axes[row, col]
            # поле хранится как [i, j] (быстрый индекс - x), contour ждет [y, x]
            if fill:
                im = _fill(ax, x_mesh, y_mesh, field, vmin, vmax)
            _isolines(ax, x_mesh, y_mesh, field, levels, fmt)
            ax.set_xlim(x_mesh[0], x_mesh[-1])
            ax.set_ylim(y_mesh[0], y_mesh[-1])
            _wells(ax, x_mesh, y_mesh)
            ax.set_aspect('equal')
            letter = PANEL_LETTERS[col * 3 + row] if transpose else PANEL_LETTERS[row * n_col + col]
            ax.text(*LETTERS_POSITION, f'({letter})', transform=ax.transAxes,
                    fontsize=9, va='top', bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
            # Название величины - в подписи панели; у столбцов/строк с заливкой рядом
            # ставится цветовая шкала (см. добавление colorbar ниже)
            if transpose:
                if col == 0:
                    ax.set_title(f't = {times[row]:.0f} сут', fontsize=9)
                if row == 0:
                    ax.set_ylabel(f'{label}\ny, м', fontsize=9)
                if col == n_col - 1:
                    ax.set_xlabel('x, м', fontsize=9)
            else:
                if row == 0:
                    ax.set_title(label, fontsize=9)
                if col == 0:
                    ax.set_ylabel(f't = {times[row]:.0f} сут\ny, м', fontsize=9)
                if row == 2:
                    ax.set_xlabel('x, м', fontsize=9)
            ax.tick_params(labelsize=8)

        # Легенда заливки - привязана к столбцу/строке, а не к каждой панели: шкала
        # общая на все три момента времени, отдельная лесенка на каждой панели была бы
        # той же самой шкалой три раза.
        if fill:
            cbar = fig.colorbar(im, ax=(axes[col, :] if transpose else axes[:, col]),
                                 format=fmt, shrink=0.85, pad=0.02, aspect=25)
            cbar.ax.tick_params(labelsize=7)

    _save(fig, name)
    plt.close(fig)


def figure_maps(data: dict, metrics: dict, name: str = 'fig1') -> None:
    """Рис. 1: поля температуры, водонасыщенности и давления на три момента времени."""
    field_panel(data, metrics, MAP_COLUMNS, name, figsize=(7.0, 7.4))


def figure_maps_short(data: dict, metrics: dict, name: str = 'fig1s') -> None:
    """Рис. 1 журнальной версии: только температура и водонасыщенность.

    Шесть подрисунков вместо девяти: журнал считает подрисунки в общий лимит, и место
    под сопоставление вариантов важнее, чем поле давления, которое описывается в тексте.
    """
    field_panel(data, metrics, MAP_COLUMNS[:2], name, figsize=(7.6, 5.4), transpose=True)


def figure_colmatation(data: dict, metrics: dict, name: str = 'fig3') -> None:
    """Рис. 3: множители проницаемости и пористости на те же три момента времени.

    Показывает то, что в тексте описано словами: зона деградации ФЕС не стоит на месте,
    а мигрирует вслед за фронтом вытеснения.
    """
    # Уровней меньше, чем на рис. 1: к концу разработки зона деградации ФЕС стягивается
    # в узкую полосу у границы, и семь подписанных изолиний в ней уже не помещаются.
    field_panel(data, metrics, COLMATATION_COLUMNS, name, figsize=(5.2, 7.4), n_levels=7)


def figure_overlay(overlay: dict, name: str = 'fig4') -> None:
    """Рис. 4: изолинии базового варианта и варианта без теплообмена в одних осях.

    Прямое сопоставление полей: одна и та же величина, один и тот же момент времени,
    две системы изолиний. Так построены рисунки исходной статьи.
    """
    if not {'base', 'noheat'} <= set(overlay):
        return

    specs = (('T', 'T, °C', 9), ('S', 'S', 9))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0), sharey=True)

    for col, (field_key, label, n_lev) in enumerate(specs):
        ax = axes[col]
        a, b = overlay['base'][field_key][2], overlay['noheat'][field_key][2]
        levels = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), n_lev)

        cs1 = ax.contour(x_mesh, y_mesh, a.T, levels=levels, colors='k', linewidths=1.6)
        cs2 = ax.contour(x_mesh, y_mesh, b.T, levels=levels, colors='k', linewidths=1.3,
                         linestyles='dashed')
        ax.clabel(cs1, levels[1::3], inline=True, fontsize=7, fmt='%.0f' if col == 0 else '%.2f')

        _wells(ax, x_mesh, y_mesh)
        ax.set_aspect('equal')
        ax.set_xlabel('x, м', fontsize=10)
        if col == 0:
            ax.set_ylabel('y, м', fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.text(*LETTERS_POSITION, f'({PANEL_LETTERS[col]})', transform=ax.transAxes, fontsize=9,
                va='top', bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
        ax.tick_params(labelsize=8)

    handles = [plt.Line2D([], [], color='k', lw=1.6, label='1'),
               plt.Line2D([], [], color='k', lw=1.3, ls='--', label='2')]
    axes[1].legend(handles=handles, title='вариант', fontsize=9, title_fontsize=9,
                   loc='center left',  bbox_to_anchor=(1.05, 0.95),  borderaxespad=0.0, framealpha=1.0)
    _save(fig, name)
    plt.close(fig)


def figure_difference(overlay: dict, times, name: str = 'fig7') -> None:
    """Рис. 7: разность полей базового варианта и варианта без теплообмена.

    Прямая мера эффекта: сколько градусов и сколько долей насыщенности теряется в каждой
    точке пласта, если кровлю и подошву считать теплоизолированными. Сравнение идет на
    одни и те же моменты физического времени, а не на конец каждого расчета: расчеты
    заканчиваются в разное время, и сопоставлять их последние слои некорректно.
    """

    if not {'base', 'noheat'} <= set(overlay):
        return

    specs = (('T', 'ΔT, °C', 0), ('S', 'ΔS', 1))
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.4), sharex=True, sharey=True,
                              constrained_layout=True)

    for row, (field_key, label, _digits) in enumerate(specs):
        diffs = [overlay['base'][field_key][i] - overlay['noheat'][field_key][i] for i in range(3)]
        lim = max(abs(d).max() for d in diffs)
        # Уровни симметричны относительно нуля и общие на все три момента времени;
        # нулевой уровень исключен - его рисует `_zero_line`
        levels = np.linspace(-lim, lim, 9)[1:-1]
        levels = levels[np.abs(levels) > 1e-9 * lim]
        fmt = _fmt(2.0 * lim)

        im = None
        for col, diff in enumerate(diffs):
            ax = axes[row, col]
            im = _fill(ax, x_mesh, y_mesh, diff, -lim, lim)
            _isolines(ax, x_mesh, y_mesh, diff, levels, fmt)
            _zero_line(ax, x_mesh, y_mesh, diff, lim)
            ax.set_xlim(x_mesh[0], x_mesh[-1])
            ax.set_ylim(y_mesh[0], y_mesh[-1])
            _wells(ax, x_mesh, y_mesh)
            ax.set_aspect('equal')
            ax.text(*LETTERS_POSITION, f'({PANEL_LETTERS[row * 3 + col]})', transform=ax.transAxes,
                    fontsize=9, va='top', bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
            if row == 0:
                ax.set_title(f't = {times[col]:.0f} сут', fontsize=9)
            if row == 1:
                ax.set_xlabel('x, м', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'{label}\ny, м', fontsize=9)
            ax.tick_params(labelsize=8)

        # Шкала общая на всю строку (один lim на все три момента) - один colorbar на строку
        cbar = fig.colorbar(im, ax=axes[row, :], format=fmt, shrink=0.85, pad=0.02, aspect=25)
        cbar.ax.tick_params(labelsize=7)

    _save(fig, name)
    plt.close(fig)


def figure_difference_compact(overlay: dict, times, name: str = 'fig8') -> None:
    """Разность полей на один момент времени - момент максимального расхождения (t2).

    Сжатая версия рис. 6 для журнальной статьи: два подрисунка вместо шести.
    """
    if not {'base', 'noheat'} <= set(overlay):
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8), sharey=True, constrained_layout=True)
    for col, (field_key, label) in enumerate((('T', 'ΔT, °C'), ('S', 'ΔS'))):
        diff = overlay['base'][field_key][1] - overlay['noheat'][field_key][1]
        lim = abs(diff).max()
        levels = np.linspace(-lim, lim, 9)[1:-1]
        levels = levels[np.abs(levels) > 1e-9 * lim]
        ax = axes[col]
        im = _fill(ax, x_mesh, y_mesh, diff, -lim, lim)
        _isolines(ax, x_mesh, y_mesh, diff, levels, _fmt(2.0 * lim))
        _zero_line(ax, x_mesh, y_mesh, diff, lim)
        ax.set_xlim(x_mesh[0], x_mesh[-1])
        ax.set_ylim(y_mesh[0], y_mesh[-1])
        _wells(ax, x_mesh, y_mesh)
        ax.set_aspect('equal')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('x, м', fontsize=10)
        if col == 0:
            ax.set_ylabel('y, м', fontsize=10)
        ax.text(*LETTERS_POSITION, f'({PANEL_LETTERS[col]})', transform=ax.transAxes, fontsize=9,
                va='top', bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
        ax.tick_params(labelsize=8)

        # У каждого подрисунка свой lim (не общий, как в fig7) - colorbar тоже свой
        cbar = fig.colorbar(im, ax=ax, format=_fmt(2.0 * lim), shrink=0.85, pad=0.03)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f't = {times[1]:.0f} сут', fontsize=10)
    _save(fig, name)
    plt.close(fig)


def figure_cumulative(cases: dict, name: str = 'fig5') -> None:
    """Рис. 5: накопленная добыча нефти по вариантам.

    Конечная точка каждой кривой - накопленная добыча, по которой считается КИН,
    поэтому различие вариантов по коэффициенту извлечения читается прямо с графика.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for key, _file, number, style, _title in ARTICLE_CASES:
        if key not in cases:
            continue
        m = cases[key]
        ax.plot(m['time'], m['Q_oil_curve'], linestyle=style, color='k',
                linewidth=LINE_WIDTH, label=str(number), **_marker(number, len(m['time'])))
        ax.plot(m['time'][-1], m['Q_oil_curve'][-1], MARKERS[number], color='k',
                markersize=7, markerfacecolor='k')

    ax.set_xlabel('t, сут', fontsize=10)
    ax.set_ylabel('Q$_о$, м$^3$', fontsize=10)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, color='0.88', linewidth=0.5)
    ax.legend(title='вариант', fontsize=9, title_fontsize=9, loc='lower right', framealpha=1.0)
    _save(fig, name)
    plt.close(fig)


def figure_pore(data: dict, metrics: dict, name: str = 'fig6') -> None:
    """Рис. 6: функция распределения пор по размерам в ячейке у нагнетательной скважины.

    Начальное распределение и распределение на моменты t1 и t3: видно, что сперва
    выбывают капилляры, удовлетворяющие условию блокирования, и лишь затем идет
    равномерное сужение остальных.
    """
    import matplotlib.pyplot as plt

    if 'plots' not in data:
        return

    fi = data['plots']['fi']
    i1, _, i3 = metrics['idx']
    t = metrics['t']

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(r * 1e6, fi_0, '-', color='0.45', linewidth=2.4,
            label='начальная')
    ax.plot(r * 1e6, fi[i1], '--', color='k', linewidth=LINE_WIDTH, label=f't = {t[0]:.0f} сут',
            **_marker(1, len(r)))
    ax.plot(r * 1e6, fi[i3], '-.', color='k', linewidth=LINE_WIDTH, label=f't = {t[2]:.0f} сут',
            **_marker(2, len(r)))

    ax.set_xlabel('r, мкм', fontsize=10)
    ax.set_ylabel('φ', fontsize=11)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, color='0.88', linewidth=0.5)
    ax.legend(fontsize=9, framealpha=1.0)
    _save(fig, name)
    plt.close(fig)


def figure_comparison(cases: dict, name: str = 'fig2') -> None:
    """Рис. 2: совмещенные кривые всех вариантов - дебит нефти и обводненность.

    Левая ось - дебит нефти, правая - обводненность. Варианты различаются только типом
    линии (журнал печатает черно-белым), обводненность вынесена серым, чтобы две группы
    кривых не путались. В легенде - номера вариантов, расшифровка в подрисуночной подписи.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax2 = ax.twinx()

    for key, _file, number, style, _title in ARTICLE_CASES:
        if key not in cases or key not in FIGURE_CASES:
            continue
        m = cases[key]
        ax.plot(m['time'], m['q_oil'], linestyle=style, color='k', linewidth=LINE_WIDTH,
                label=str(number), **_marker(number, len(m['time'])))
        # Обводненность - та же линия серой: две группы кривых не должны смешиваться
        ax2.plot(m['time'], m['eta'], linestyle=style, color='0.55', linewidth=LINE_WIDTH - 0.4,
                 **_marker(number, len(m['time']), color='0.55'))

    ax.set_xlabel('t, сут', fontsize=10)
    ax.set_ylabel('q$_о$, м$^3$/сут', fontsize=10)
    ax2.set_ylabel('$\\eta$', fontsize=10)
    ax2.set_ylim(0.0, 1.0)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, color='0.85', linewidth=0.5)
    ax.legend(title='вариант', fontsize=9, title_fontsize=9, loc='upper right', framealpha=1.0)

    _save(fig, name)
    plt.close(fig)


def _load_core_flood():
    """Результат `core_flood.py`: калибровка по опыту 1, прогноз опыта 2. None, если расчета еще нет."""
    import json
    import core_flood

    if not core_flood.RESULT.is_file():
        return None
    return json.loads(core_flood.RESULT.read_text(encoding='utf-8'))


def figure_core_flood(name: str = 'fig9') -> None:
    """Рис. 9: k/k₀ от прокачанных поровых объемов в опытах Sutton & Roberts и по моделям.

    Точки - опыт, сплошная линия - настоящая модель (опыт 1 - калибровка d_p и L_k, опыт 2 - прогноз
    с теми же значениями), штриховая - модель Ring et al. [10], пунктир - Wang & Civan [9]. У обеих
    чужих моделей параметры подобраны отдельно под каждый опыт.
    """
    from core_flood import EXPERIMENTS

    result = _load_core_flood()
    if result is None:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for number, marker, fill in ((1, 'o', 'k'), (2, 's', 'white')):
        exp, model = EXPERIMENTS[number], result['best'][str(number)]
        pv, k = np.array(exp['exp']).T
        ax.plot(pv, k, marker, color='k', markersize=6, markerfacecolor=fill, markeredgewidth=1.2,
                linestyle='none', zorder=3)
        # После закупорки счет прекращен, а в данных стоит точка (PV_END, 0): рисуем до закупорки и ставим ×
        pv_m, k_m = (model['pv'][:-1], model['k'][:-1]) if model['plugged'] else (model['pv'], model['k'])
        ax.plot(pv_m, k_m, '-', color='k', linewidth=LINE_WIDTH)
        if model['plugged']:
            ax.plot(pv_m[-1], k_m[-1], 'x', color='k', markersize=9, markeredgewidth=1.8)
        model = {'pv': pv_m, 'k': k_m}
        ax.plot(*np.array(exp['ring']).T, '--', color='k', linewidth=1.3)
        ax.plot(*np.array(exp['wang_civan']).T, ':', color='k', linewidth=1.6)
        ax.annotate(str(number), (model['pv'][-1], model['k'][-1]), xytext=(4, 0), textcoords='offset points',
                    fontsize=10, va='center')

    handles = [plt.Line2D([], [], marker='o', color='k', linestyle='none', label='опыт 1'),
               plt.Line2D([], [], marker='s', color='k', markerfacecolor='white', linestyle='none', label='опыт 2'),
               plt.Line2D([], [], color='k', linewidth=LINE_WIDTH, label='расчет'),
               plt.Line2D([], [], color='k', linewidth=1.3, linestyle='--', label='[10]'),
               plt.Line2D([], [], color='k', linewidth=1.6, linestyle=':', label='[9]')]
    ax.legend(handles=handles, fontsize=9, loc='upper right', framealpha=1.0)
    ax.set_xlabel('V/V$_п$', fontsize=10)
    ax.set_ylabel('k/k$_0$', fontsize=10)
    ax.set_xlim(0.0, 5.4)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, color='0.88', linewidth=0.5)
    _save(fig, name)
    plt.close(fig)


def core_flood_numbers() -> list:
    """Числа раздела о сопоставлении с экспериментом: растворимость (Li) и керн (Sutton & Roberts)."""
    from core_flood import EXPERIMENTS, LI_W, LI_PRECIPITATION, cloud_point, w_saturated
    from paraphin.constants import Tm, alpha, MW, M_o

    lines = ['', '## Сопоставление с экспериментом', '',
             f'Растворимость: эффективные Tm = {Tm:.1f}°C, alpha = {alpha / 1e3:.1f} кДж/моль '
             f'(MW = {MW:.0f}, M_o = {M_o:.0f}); точка помутнения нефти Li ({100 * LI_W:.2f}%) по модели '
             f'{cloud_point(LI_W, MW, M_o, Tm, alpha):.1f}°C, по ДСК 45.65°C', '',
             '| T, °C | выпало по [Li], % | по модели, % |', '|---|---|---|']
    for t, measured in LI_PRECIPITATION:
        model = 100.0 * (LI_W - float(w_saturated(LI_W, t, MW, M_o, Tm, alpha)))
        lines.append(f'| {t:g} | {measured:.1f} | {model:.1f} |')

    lines.append('')
    for number, exp in EXPERIMENTS.items():
        cp = cloud_point(exp['w'], exp['MW'], exp['M_o'], exp['Tm'], exp['dH'])
        lines.append(f'- опыт {number} Sutton & Roberts: точка помутнения модели {cp:.1f}°C, измеренная {exp["cloud"]}°C')

    result = _load_core_flood()
    if result is None:
        return lines + ['- расчета керна нет: python experiments/исходная_модель/core_flood.py']

    best1 = result['best']['1']
    lines.append(f'- калибровка по опыту 1: d_p = {best1["d_p"] * 1e6:.1f} мкм, L_k = {best1["lk"] * 1e6:.0f} мкм, '
                 f'eta керна = {best1["eta"]:.2f}, сетка {best1["ny"]} ячеек, шаг до {best1["dt"]} с')
    for number, exp in EXPERIMENTS.items():
        model = result['best'][str(number)]
        others = result['others_rms'][str(number)]
        at = ', '.join(f'{x} PV: опыт {np.interp(x, *np.array(exp["exp"]).T):.2f}, '
                       f'расчет {np.interp(x, model["pv"], model["k"]):.2f}' for x in (0.25, 1.0, 2.0, 3.0, 5.0))
        lines.append(f'- опыт {number} ({"калибровка" if number == 1 else "прогноз"}): СКО модели {model["rms"]:.3f}, '
                     f'Ring {others["ring"]:.3f}, Wang & Civan {others["wang_civan"]:.3f}; закупорка '
                     f'{model["plugged"]}; {at}')
        lines.append(f'  профиль в конце: k/k0 у входа {model["k_profile"][0]:.3f}, на выходе {model["k_profile"][-1]:.3f}; '
                     f'm/m0 у входа {model["m_profile"][0]:.3f}, на выходе {model["m_profile"][-1]:.3f}')

    grid = sorted(result['calibration'], key=lambda r: r['rms'])[:5]
    lines.append('- пять лучших точек калибровки (d_p мкм, L_k мкм, СКО): '
                 + '; '.join(f'{r["d_p"] * 1e6:.1f}, {r["lk"] * 1e6:.0f}, {r["rms"]:.3f}' for r in grid))
    return lines


def create_article_figures() -> dict:
    """Построение всех рисунков статьи и сбор чисел для текста и таблиц."""
    cases, missing, overlay = {}, [], {}
    base_times = None
    for key, file_name, _number, _style, _title in ARTICLE_CASES:
        try:
            _, data = read_solution_data(file_name)
        except ValueError:
            missing.append(file_name)
            continue
        cases[key] = case_metrics(data)

        if key == 'base':
            figure_maps(data, cases[key])
            figure_maps_short(data, cases[key])
            figure_colmatation(data, cases[key])
            figure_pore(data, cases[key])
        # Для сопоставления полей нужны два расчета сразу. Держать оба файла в памяти нельзя -
        # это сотни мегабайт, поэтому забираем только срезы на три момента времени.
        # Моменты берутся у базового варианта: расчеты заканчиваются в разное время, и
        # сравнивать последние слои разных вариантов означало бы сравнивать разные моменты.
        if key == 'base':
            base_times = cases[key]['t']
            picks = cases[key]['idx']
        elif key == 'noheat' and base_times is not None:
            own = cases[key]['time']
            picks = [int(np.argmin(np.abs(own - t))) for t in base_times]
        else:
            picks = None

        if picks is not None and key in ('base', 'noheat'):
            overlay[key] = {'T': [data['Temperature'][i].copy() for i in picks],
                            'S': [data['Saturation'][i].copy() for i in picks]}
        del data

    figure_core_flood()
    if missing:
        print('Нет файлов расчета:', ', '.join(missing))
    if cases:
        figure_comparison(cases)
        figure_overlay(overlay)
        figure_cumulative(cases)
        if base_times is not None:
            figure_difference(overlay, base_times)
            figure_difference_compact(overlay, base_times)
        print(f'Рисунки записаны в {_article_figures_path()}')

    return cases


if __name__ == '__main__':
    main()
