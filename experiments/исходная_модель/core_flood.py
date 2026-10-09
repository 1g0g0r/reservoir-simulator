"""Проверка модели по лабораторным данным: кольматация керна (Sutton & Roberts) и растворимость (Li et al.).

Кольматация. Sutton G.D., Roberts L.D. Paraffin Precipitation During Fracture Stimulation // JPT. 1974.
V. 26. No 9. P. 997 - прокачка парафинистой нефти через керн Berea ниже точки помутнения, измерено
k/k_0 от прокачанных поровых объемов. Постановка и свойства - по Ring et al. (SPE PF, 1994, табл. 1-4),
которые эти опыты моделировали; так же их моделировали Wang & Civan (JERT, 2005, табл. 1-2). Как и у
Ring, керн изотермический (21.1 C), нефть и в керне, и на входе уже содержит равновесную взвесь кристаллов.
Расход у Sutton & Roberts не приведен, берется допущение Ring.

Растворимость. Li X. et al. Cooling Damage Characterization ... in Low-Permeable and High-Waxy Oil
Reservoirs // Processes. 2024. V. 12. P. 421 - кривая выпадения парафина из нефти Жетыбая (рис. 3б).
По ней подбираются эффективные Tm и alpha в (6.1) для полевого расчета (`--fit`).

Параметры модели (d_p, L_k) у одного варианта - константы уровня модуля, numba вшивает их в машинный
код. Поэтому каждый вариант - отдельный процесс в копии пакета с поправленным `constants.py`
(`tests/_patched_copy.make_copy`), прогон - общим рабочим процессом `experiments/core_runner.py`; репозиторий и его
кеш numba не трогаются.

    python experiments/исходная_модель/core_flood.py          # калибровка d_p, L_k по опыту 1, прогноз опыта 2
    python experiments/исходная_модель/core_flood.py --fit    # подгонка Tm, alpha по кривой Li
    python experiments/исходная_модель/core_flood.py --check  # сходимость по сетке и шагу при найденных d_p, L_k
    python experiments/исходная_модель/core_flood.py --ring   # оба опыта с долей взвеси по расчету Ring
    python experiments/исходная_модель/core_flood.py --diag   # разложение расхождения с опытом по причинам
    python experiments/исходная_модель/core_flood.py --thermal  # калибровка в неизотермической постановке опыта

Результат - `outputs/data/core_flood.json`, его читает `make_figures.py`.
"""
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares

ROOT = Path(__file__).resolve().parents[2]
# В конец пути: рабочий процесс берет `paraphin` из копии (PYTHONPATH), а `tests` - из репозитория
sys.path.append(str(ROOT))
from tests._patched_copy import make_copy  # noqa: E402

RESULT = ROOT / 'outputs' / 'data' / 'core_flood.json'

R_GAS = 8.31446261815324
FT = 0.3048                       # [м]
LBFT3 = 16.018463                 # lbm/ft^3 -> кг/м^3
KCAL = 4186.8                     # ккал -> Дж
DAY = 86400.0


def _rankine_to_c(t):
    return (t - 459.67 - 32.0) / 1.8


# Керн и режим (Ring et al., 1994, табл. 2, 4; Wang & Civan, 2005, табл. 1). Сечение - как у Ring
# (квадрат 0.1221 фута); у Wang & Civan диаметр 2.5 см при том же расходе. Время в PV от сечения не зависит.
LENGTH = 1.0 * FT
SIDE = 0.1221 * FT
POROSITY = 0.25
T_CORE = _rankine_to_c(70.0 + 459.67)  # 21.1 C
T_HOT = _rankine_to_c(130.0 + 459.67)  # 54.4 C: нагрев керна и температура нагнетаемой нефти в опыте
P_OUT = 1900 * 6894.757                # противодавление, [Па]; на несжимаемую задачу не влияет
# Вязкость нефти при 21 C не приведена ни у Ring, ни у Wang & Civan. Принята типичная для легкой нефти
# (псевдокомпонент M = 104-123, 0.72-0.75 г/см^3). При заданном расходе скорость в капилляре от вязкости
# не зависит (|grad p|/mu = u/k), вязкость входит только в стоксовскую диффузию частиц D_p ~ 1/mu,
# то есть в сужение ~ D_p^(2/3), и поглощается калибровкой.
MU_OIL = 3.0e-3

# Опыты. Точки k/k_0(PV) и кривые моделей оцифрованы по рисункам с растра 300 dpi (точность ~0.01-0.02):
# опыт - Ring рис. 5-6 (совпадает с Wang & Civan рис. 1-2); Ring - кривая lambda = 0.5 ft^-1, m = 8 (значения,
# принятые ими для обоих опытов); Wang & Civan - поверхностное осаждение, alpha_P = 0.0241 и 0.0182 1/с.
EXPERIMENTS = {
    1: dict(
        title='Shannon Sand crude, Berea',
        k0=405e-3, w=0.041, MW=522.4, M_o=104.11,
        ro_o=44.69 * LBFT3, ro_p=61.08 * LBFT3,
        Tm=_rankine_to_c(628.2), dH=26.0 * KCAL, cloud=37.8, solids_ring=0.0334,
        q=0.38 * FT ** 3 / DAY,
        exp=[(0.0, 1.00), (0.25, 0.70), (0.5, 0.60), (0.75, 0.57), (1.0, 0.56), (1.25, 0.51), (1.5, 0.41),
             (1.75, 0.40), (2.0, 0.39), (2.25, 0.34), (2.5, 0.37), (2.75, 0.31), (3.0, 0.33), (3.25, 0.23),
             (3.5, 0.25), (3.75, 0.26), (4.0, 0.24), (4.25, 0.23), (4.5, 0.27), (4.75, 0.20), (5.0, 0.19)],
        ring=[(0.0, 1.0), (0.25, 0.928), (0.5, 0.870), (0.75, 0.820), (1.0, 0.765), (1.5, 0.674), (2.0, 0.592),
              (2.5, 0.516), (3.0, 0.449), (3.5, 0.389), (4.0, 0.339), (4.5, 0.291), (5.0, 0.251)],
        wang_civan=[(0.0, 1.0), (0.1, 0.891), (0.25, 0.746), (0.4, 0.694), (0.6, 0.645), (0.85, 0.577),
                    (1.15, 0.494), (1.4, 0.437), (1.6, 0.397), (1.9, 0.350), (2.15, 0.320), (2.4, 0.298),
                    (2.6, 0.284), (2.9, 0.268), (3.15, 0.258), (3.4, 0.254), (3.6, 0.250), (3.9, 0.243),
                    (4.15, 0.243), (4.4, 0.240), (4.6, 0.237), (4.9, 0.237)],
    ),
    2: dict(
        title='Muddy formation crude, Berea',
        k0=314e-3, w=0.061, MW=479.7, M_o=122.51,
        ro_o=46.71 * LBFT3, ro_p=61.08 * LBFT3,
        Tm=_rankine_to_c(621.0), dH=23.6 * KCAL, cloud=35.0, solids_ring=0.0322,
        q=0.30 * FT ** 3 / DAY,
        exp=[(0.0, 1.00), (0.25, 0.72), (0.5, 0.63), (0.75, 0.63), (1.0, 0.58), (1.25, 0.59), (1.5, 0.51),
             (1.75, 0.51), (2.0, 0.49), (2.25, 0.53), (2.5, 0.54), (2.75, 0.48), (3.0, 0.50), (3.25, 0.43),
             (3.5, 0.40), (3.75, 0.40), (4.0, 0.38), (4.25, 0.39), (4.5, 0.32), (4.75, 0.28), (5.0, 0.29)],
        ring=[(0.0, 1.0), (0.25, 0.928), (0.5, 0.873), (0.75, 0.820), (1.0, 0.770), (1.5, 0.677), (2.0, 0.596),
              (2.5, 0.521), (3.0, 0.457), (3.5, 0.401), (4.0, 0.348), (4.5, 0.301), (5.0, 0.26)],
        wang_civan=[(0.0, 1.0), (0.1, 0.899), (0.25, 0.752), (0.4, 0.708), (0.6, 0.663), (0.85, 0.615),
                    (1.15, 0.560), (1.4, 0.517), (1.6, 0.492), (1.9, 0.462), (2.15, 0.438), (2.4, 0.422),
                    (2.6, 0.412), (2.9, 0.400), (3.15, 0.392), (3.4, 0.387), (3.6, 0.384), (3.9, 0.384),
                    (4.15, 0.379), (4.4, 0.379), (4.6, 0.379), (4.9, 0.376)],
    ),
}

# Кривая выпадения парафина из нефти Жетыбая (Li et al., 2024, рис. 3б, ДСК): (T, C; выпало, % масс. нефти).
# Оцифровка с точностью ~0.5%; WAT 45.65 C по ДСК (рис. 3а), общее содержание парафина 24.93%.
LI_W = 0.2493
LI_PRECIPITATION = [(45.65, 0.0), (40.0, 1.5), (35.0, 5.0), (30.0, 9.5), (25.0, 13.0), (20.0, 16.0),
                    (10.0, 20.5), (0.0, 23.0), (-10.0, 24.2), (-20.0, 24.9)]
LI_FIT_T_MIN = 10.0  # в пласте холоднее закачиваемой воды не бывает; подгонка - по интервалу 10-45 C

# Калибровка: сетка по диаметру частиц и длине порового канала, критерий - СКО от опыта 1.
# d_p задает, какие каналы блокируются (r <= d_p/(2*gamma)), то есть начальный провал k; L_k - интенсивность
# сужения (u_r ~ L_k^(-1/3)), то есть длину зоны осаждения у входа. Доля блокируемых каналов beta не
# калибруется: при любом beta из 0.003-0.3 блокирование идет за секунды, быстрее переноса, и k/k_0 от него
# не зависит (проверено: СКО 0.335-0.338 при d_p = 5 мкм); остается полевое значение из constants.py.
D_GRID = (10e-6, 12.5e-6, 15e-6, 17.5e-6, 20e-6)
LK_GRID = (3e-5, 1e-4, 3e-4, 1e-3)
NY, DT = 60, 2.0        # ячеек вдоль керна; стартовый и наибольший шаг, [с]
PV_END = 5.2
PLUGGED = 0.02          # k/k_0, ниже которого керн считается закупоренным и счет прекращается


def w_saturated(w_sum, T, MW, M_o, Tm, alpha):
    """Предел растворимости (6.1)-(6.2) - та же формула, что `Thermo_wax.single_wp_saturated`, но с параметрами аргументами:
    `paraphin` берет их из `constants.py`, а здесь нужны параметры каждого опыта."""
    x = np.minimum(1.0, np.exp(-alpha / R_GAS * (1.0 / (T + 273.15) - 1.0 / (Tm + 273.15))))
    w_hat = x * MW / (x * MW + (1.0 - x) * M_o)
    return np.minimum(w_sum, w_hat / np.maximum(1.0 - w_hat, 1e-15) * (1.0 - w_sum))


def cloud_point(w_sum, MW, M_o, Tm, alpha):
    """Температура, при которой предел растворимости сравнивается с содержанием парафина, [C]."""
    return brentq(lambda t: w_saturated(w_sum, t, MW, M_o, Tm, alpha) - w_sum + 1e-12, -80.0, Tm - 1e-6)


def fit_solubility():
    """Эффективные Tm и alpha в (6.1) по кривой Li (МНК по точкам не холоднее `LI_FIT_T_MIN`)."""
    sys.path.insert(0, str(ROOT))
    from paraphin.constants import MW, M_o

    pts = np.array([p for p in LI_PRECIPITATION if p[0] >= LI_FIT_T_MIN])
    t, prec = pts[:, 0], pts[:, 1] / 100.0

    def residual(x):
        return (LI_W - w_saturated(LI_W, t, MW, M_o, x[0], x[1] * 1e3)) - prec

    res = least_squares(residual, [80.0, 40.0])
    tm, alpha = res.x[0], res.x[1] * 1e3
    print(f'Tm = {tm:.1f} C, alpha = {alpha / 1e3:.1f} кДж/моль (MW = {MW}, M_o = {M_o}); '
          f'макс. отклонение {100 * np.abs(res.fun).max():.2f}%, WAT модели {cloud_point(LI_W, MW, M_o, tm, alpha):.1f} C')
    return tm, alpha


def w_ring_solids(number: int) -> float:
    """Суммарная доля парафина, при которой равновесная взвесь при 21.1 C равна рассчитанной Ring.

    Содержание взвеси в опытах не измерялось. Ring (табл. 1, 3) получил 3.34 и 3.22% об. твердой фазы,
    а (6.1)-(6.2) с теми же псевдокомпонентами дают 2.7 и 4.0%: в опыте 2 взвеси на 40% больше, чем в
    опыте 1, у Ring - почти поровну. Прогон с этой долей отделяет погрешность равновесия от кинетики.
    """
    exp = EXPERIMENTS[number]
    phi = exp['solids_ring']
    target = phi * exp['ro_p'] / (phi * exp['ro_p'] + (1.0 - phi) * exp['ro_o'])  # массовая доля взвеси

    def excess(w):
        return w - float(w_saturated(w, T_CORE, exp['MW'], exp['M_o'], exp['Tm'], exp['dH'])) - target

    return brentq(excess, target, 0.5)


# --- Прогон одного варианта -------------------------------------------------------------------------

def _case_constants(exp: dict, d_p: float, lk: float, ny: int, dt: float) -> dict:
    """Константы прогона. Геометрия керна (`length`, `side`, `porosity`), его температура `T`, вязкость
    жидкой основы при ней `mu` и длительность в PV `pv_end` берутся из опыта, если заданы, иначе - керн
    Sutton & Roberts (так их задают сторонние опыты, `validation_*.py`)."""
    side, porosity, t_core = exp.get('side', SIDE), exp.get('porosity', POROSITY), exp.get('T', T_CORE)
    w_sat = float(w_saturated(exp['w'], t_core, exp['MW'], exp['M_o'], exp['Tm'], exp['dH']))
    t_end = exp.get('pv_end', PV_END) * exp.get('length', LENGTH) * side * side * porosity / exp['q']
    return {
        'Nx, Ny': f'1, {ny}',
        'X_min, X_max': f'0.0, {side!r}',
        'Y_min, Y_max': f'0.0, {exp.get("length", LENGTH)!r}',
        'h': repr(side),
        'Time_end': repr(t_end),
        'dt': repr(dt),
        'sol_time_step': repr(10.0 * t_end),   # сохранять в layers.pkl нечего, кроме обязательного первого слоя
        'Twater': repr(t_core),
        'S_min': '0.0',                        # однофазная нефть: pf_o(0) = 1, множитель (S_o - S_o*) = 1
        'S_max': '1.0',
        'heat_losses': '0',
        'init_Wp': repr(w_sat),                # равновесие при температуре керна: скрытая теплота не выделяется
        'init_Wps': repr(exp['w'] - w_sat),
        'init_k': f'{exp["k0"]!r} * darcy_to_m2',
        'init_m': repr(porosity),
        'init_T': repr(t_core),
        'ro_o': repr(exp['ro_o']),
        'ro_p': repr(exp['ro_p']),
        'D': repr(d_p),
        'Lk': repr(lk),
        'mu_o_ref': repr(exp.get('mu', MU_OIL)),
        'T_mu_ref': repr(t_core),
        'MW': repr(exp['MW']),
        'M_o': repr(exp['M_o']),
        'Tm': repr(exp['Tm']),
        'alpha': repr(exp['dH']),
    }


def run_case(number: int, d_p: float, lk: float, ny: int = NY, dt: float = DT, w: float = None,
             oil: dict = None, extra: dict = None, mode: str = 'rate', exp: dict = None) -> dict:
    """Прогон опыта `number` отдельным процессом в копии пакета с поправленным `constants.py`.

    `w` - суммарная доля парафина вместо указанной в опыте (см. `w_ring_solids`); `oil` - свойства нефти
    вместо свойств опыта (MW, M_o, ro_o, ro_p, Tm, dH); `extra` - дополнительные константы `constants.py`;
    `mode` - режим рабочего процесса (`core_runner.py`): 'rate', 'dp', 'thermal' или 'ramp'; `exp` - сторонний опыт
    вместо `EXPERIMENTS[number]` (тогда `number` - только метка временных файлов).
    """
    exp = dict(exp or EXPERIMENTS[number], **(oil or {}), **({} if w is None else {'w': w}))
    values = _case_constants(exp, d_p, lk, ny, dt)
    if mode == 'thermal':  # керн нагрет и без взвеси, парафин весь растворен
        values.update(init_T=repr(exp.get('T_hot', T_HOT)), init_Wp=repr(exp['w']), init_Wps='0.0')
    values.update(extra or {})
    root = make_copy(f'core_flood_{number}', values)
    result = run_runner(root, exp, mode)
    result.update(d_p=d_p, lk=lk, ny=ny, dt=dt, rms=rms(result, exp['exp']))
    return result


RUNNER = ROOT / 'experiments' / 'core_runner.py'


def run_runner(root: Path, exp: dict, mode: str = 'rate') -> dict:
    """Прогон опыта общим рабочим процессом керна `experiments/core_runner.py` в копии пакета `root`.

    Опыт дополняется умолчаниями керна Sutton & Roberts (давление на выходе, длина, длительность в PV, температуры,
    порог закупорки). При закупорке кривая дополняется точкой k = 0 в конце прокачки, как в прежнем рабочем процессе
    этого модуля: СКО по интерполяции иначе брала бы последнее значение до закупорки."""
    run_exp = dict(exp, P_out=exp.get('P_out', P_OUT), length=exp.get('length', LENGTH), T=exp.get('T', T_CORE),
                   T_hot=exp.get('T_hot', T_HOT), pv_end=exp.get('pv_end', PV_END), plugged=exp.get('plugged', PLUGGED))
    case, out = root / 'case.json', root / 'result.json'
    case.write_text(json.dumps({'exp': run_exp, 'mode': mode}), encoding='utf-8')
    subprocess.run([sys.executable, str(RUNNER), str(case), str(out)], cwd=root,
                   env=dict(os.environ, PYTHONPATH=str(root)), check=True)
    result = json.loads(out.read_text(encoding='utf-8'))
    if result['plugged'] is not None:
        for key, value in (('pv', run_exp['pv_end']), ('k', 0.0), ('k_harm', 0.0)):
            result[key].append(value)
    return result


def rms(result: dict, points) -> float:
    """СКО модели от опытных точек k/k_0(PV) (точка PV = 0 не считается - там 1 по определению)."""
    pv, k = np.array(result['pv']), np.array(result['k'])
    pts = np.array([p for p in points if p[0] > 0.0])
    if pts.size == 0:  # опыт без кривой k/k0(PV) (Sandyga et al.: градиент от температуры)
        return float('nan')
    return float(np.sqrt(np.mean((np.interp(pts[:, 0], pv, k) - pts[:, 1]) ** 2)))


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    for number, exp in EXPERIMENTS.items():
        cp = cloud_point(exp['w'], exp['MW'], exp['M_o'], exp['Tm'], exp['dH'])
        print(f'Опыт {number}: точка помутнения модели {cp:.1f} C, измеренная {exp["cloud"]} C')

    runs = []
    for d_p, lk in product(D_GRID, LK_GRID):
        res = run_case(1, d_p, lk)
        runs.append(res)
        at = [round(float(np.interp(x, res['pv'], res['k'])), 3) for x in (0.25, 1.0, 2.0, 3.0, 5.0)]
        print(f'd_p = {d_p * 1e6:.1f} мкм, L_k = {lk * 1e6:.0f} мкм: СКО {res["rms"]:.4f}, k/k0 при 0.25/1/2/3/5 PV = {at}, '
              f'закупорка {res["plugged"]}, ошибка расхода {res["q_error"]:.1e}', flush=True)

    best = min(runs, key=lambda r: r['rms'])
    prediction = run_case(2, best['d_p'], best['lk'])
    print(f'Лучшее: d_p = {best["d_p"] * 1e6:.1f} мкм, L_k = {best["lk"] * 1e6:.0f} мкм; опыт 1 СКО {best["rms"]:.4f}, '
          f'опыт 2 (прогноз) СКО {prediction["rms"]:.4f}')

    others = {n: {name: rms({'pv': [p for p, _ in e[name]], 'k': [k for _, k in e[name]]}, e['exp'])
                  for name in ('ring', 'wang_civan')} for n, e in EXPERIMENTS.items()}
    RESULT.write_text(json.dumps({
        'calibration': [{k: r[k] for k in ('d_p', 'lk', 'rms')} for r in runs],
        'best': {1: best, 2: prediction}, 'others_rms': others,
    }, ensure_ascii=False), encoding='utf-8')
    print(f'Записано: {RESULT}')


def check() -> None:
    """Сходимость по сетке и шагу при найденных d_p, L_k (результаты - в docs/WAX_PRECIPITATION_FINDINGS.md)."""
    sys.stdout.reconfigure(encoding='utf-8')
    best = json.loads(RESULT.read_text(encoding='utf-8'))['best']['1']
    for ny, dt in ((30, DT), (60, DT), (120, DT), (60, DT / 2)):
        res = run_case(1, best['d_p'], best['lk'], ny, dt)
        print(f'Ny = {ny}, dt = {dt} с: СКО {res["rms"]:.4f}, k/k0(1 PV) = {np.interp(1.0, res["pv"], res["k"]):.3f}, '
              f'k/k0(5 PV) = {np.interp(5.0, res["pv"], res["k"]):.3f}', flush=True)


def ring_solids() -> None:
    """Оба опыта при найденных d_p, L_k, но с долей взвеси, рассчитанной Ring (см. `w_ring_solids`)."""
    sys.stdout.reconfigure(encoding='utf-8')
    result = json.loads(RESULT.read_text(encoding='utf-8'))
    best = result['best']['1']
    result['ring_solids'] = {}
    for number in EXPERIMENTS:
        res = run_case(number, best['d_p'], best['lk'], w=w_ring_solids(number))
        result['ring_solids'][str(number)] = res
        print(f'Опыт {number}, взвесь по Ring: СКО {res["rms"]:.4f}, закупорка {res["plugged"]}', flush=True)
    RESULT.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')


DIAG = ROOT / 'outputs' / 'data' / 'core_flood_diag.json'
OIL_KEYS = ('MW', 'M_o', 'ro_o', 'ro_p', 'Tm', 'dH')


def diagnose() -> None:
    """Разложение расхождения с опытом по причинам (см. docs/WAX_PRECIPITATION_FINDINGS.md).

    Все прогоны - при найденных d_p, L_k; меняется по одному фактору. Результат - `core_flood_diag.json`
    и таблица в консоли.
    """
    sys.stdout.reconfigure(encoding='utf-8')
    best = json.loads(RESULT.read_text(encoding='utf-8'))['best']['1']
    d_p, lk = best['d_p'], best['lk']
    oil1 = {key: EXPERIMENTS[1][key] for key in OIL_KEYS}
    cases = [
        ('база', {}),
        ('без блокирования (beta = 0)', dict(extra={'betta': '0.0'})),
        ('без сужения (L_k = 1 км)', dict(extra={'Lk': '1e3'})),
        ('постоянный перепад', dict(mode='dp')),
        ('неизотермический керн', dict(mode='thermal')),
        ('sigma_r = 0.6', dict(extra={'sigma_r': '0.6'})),
        ('sigma_r = 0.8', dict(extra={'sigma_r': '0.8'})),
        ('r_m = 8 мкм', dict(extra={'r_m': '8e-6'})),
        ('r_m = 16 мкм', dict(extra={'r_m': '16e-6'})),
    ]
    table = {}
    for title, kwargs in cases:
        for number in EXPERIMENTS:
            table[f'{title} | опыт {number}'] = run_case(number, d_p, lk, **kwargs)
    table['нефть опыта 1 | опыт 2'] = run_case(2, d_p, lk, oil=oil1)

    print('| Вариант | СКО | k/k0 при 0.25/1/3/5 PV | закупорка, PV | потеря пор / m0 | пробки / потеря | Um_r2 вход кон/нач |')
    for name, res in table.items():
        at = '/'.join(f'{np.interp(x, res["pv"], res["k"]):.2f}' for x in (0.25, 1.0, 3.0, 5.0))
        loss = np.array(res['pore_loss'])
        blocked = np.sum(res['qp2_sum']) / max(np.sum(res['qp1_sum']) + np.sum(res['qp2_sum']), 1e-30)
        um = res['um_in'][-1] / res['um_in'][0] if res['um_in'] else float('nan')
        plug = f'{res["plugged"]:.2f}' if res['plugged'] else '—'
        print(f'| {name} | {res["rms"]:.3f} | {at} | {plug} | {loss.mean() / POROSITY:.2f} | {blocked:.2f} | {um:.1f} |')
    DIAG.write_text(json.dumps(table, ensure_ascii=False), encoding='utf-8')


def thermal_fit() -> None:
    """Оценка исправления «постановка опыта»: d_p, L_k по опыту 1 в неизотермической постановке ('thermal')
    и прогноз опыта 2 в ней же. Результат - ключ 'thermal_fit' в `core_flood_diag.json`."""
    sys.stdout.reconfigure(encoding='utf-8')
    runs = []
    for d_p, lk in product((12.5e-6, 15e-6, 17.5e-6), (3e-5, 1e-4, 3e-4)):
        res = run_case(1, d_p, lk, mode='thermal')
        runs.append(res)
        print(f'thermal: d_p = {d_p * 1e6:.1f} мкм, L_k = {lk * 1e6:.0f} мкм: СКО {res["rms"]:.4f}, '
              f'закупорка {res["plugged"]}', flush=True)
    best = min(runs, key=lambda r: r['rms'])
    prediction = run_case(2, best['d_p'], best['lk'], mode='thermal')
    print(f'thermal, лучшее: d_p = {best["d_p"] * 1e6:.1f} мкм, L_k = {best["lk"] * 1e6:.0f} мкм; опыт 1 СКО '
          f'{best["rms"]:.4f}, опыт 2 (прогноз) СКО {prediction["rms"]:.4f}, закупорка {prediction["plugged"]}')
    diag = json.loads(DIAG.read_text(encoding='utf-8'))
    diag['thermal_fit'] = {'calibration': [{k: r[k] for k in ('d_p', 'lk', 'rms', 'plugged')} for r in runs],
                           'best': {1: best, 2: prediction}}
    DIAG.write_text(json.dumps(diag, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--fit']:
        fit_solubility()
    elif sys.argv[1:2] == ['--check']:
        check()
    elif sys.argv[1:2] == ['--ring']:
        ring_solids()
    elif sys.argv[1:2] == ['--diag']:
        diagnose()
    elif sys.argv[1:2] == ['--thermal']:
        thermal_fit()
    else:
        main()
