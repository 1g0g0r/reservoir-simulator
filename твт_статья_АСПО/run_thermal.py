"""Влияние термических процессов на поле: элемент заводнения с кинетикой осаждения, подобранной по кернам.

    python твт_статья_АСПО/run_thermal.py              # все варианты, по два параллельно
    python твт_статья_АСПО/run_thermal.py base tiso    # выборочно
    python твт_статья_АСПО/run_thermal.py --metrics    # только показатели по готовым расчетам

Постановка - как `demo_composition.py` (пятиточечный элемент 200 x 200 x 10 м, сетка 50 x 50, 5 лет, забойные 17 и
7 МПа, нефть Жетыбая с детальным составом), но пласт холоднее: T0 = 55 C, ближе к WAT живой нефти (39-40 C). При
70 C остывала лишь зона у скважины, и парафин на показатели почти не влиял. К полной модели добавлена кинетика
кристаллизации (подбор по ступени 25 C керна Li et al. 2024). Удержание смол и асфальтенов (подбор по тому же
керну выше WAT) - отдельный фактор, вариант `ret`: в базовом варианте его нет, иначе за 18 сут оно давало
нагнетательной скважине скин 45, закачка падала в шесть раз, и различия термических вариантов пропадали.
Поровое пространство - пучок капилляров: сеть пор и горл с той же кинетикой закупоривает керн Li при 25 C, чего в
опыте нет (разбор - в статье).

Варианты меняют по одному фактору относительно базового (VARIANTS); `nowax` и `hl0_nowax` вместе с `base` и `hl0`
дают полный факторный план теплообмен x парафин прежней статьи (`old_article.md`). Каждый вариант считается в своей
копии пакета (`tests/_patched_copy.py`), полные результаты - в `outputs/data/thermal_<вариант>__*`, показатели для
статьи - в `твт_статья_АСПО/results/thermal.json`.
"""
import json
import os
import pickle
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from tests._patched_copy import make_copy  # noqa: E402

DATA = ROOT / 'outputs' / 'data'
RESULTS = HERE / 'results'
YEARS = 5.0
T0 = 55.0      # начальная температура пласта, [C]; в корневом constants.py - 70, поэтому в показателях берется отсюда
T_INJ = 20.0   # температура закачки базового варианта, [C]
MAP_YEARS = (1.0, 3.0, 5.0)  # моменты карт базового варианта
COMMON = {'Nx, Ny': '50, 50', 'Time_end': f'day_to_sec * 365 * {YEARS}', 'Pw': '170 * bar_to_pa',
          'Po': '70 * bar_to_pa', 'init_T': repr(T0), 'Twater': repr(T_INJ),
          # полная модель детального состава (`demo_composition.FULL`)
          'wax_characterization': "'scn'", 'wax_pressure': 'True', 'asphaltenes': 'True', 'gelation': 'True',
          'wax_viscosity': '1', 'pressure_viscosity': 'True',
          # кинетика кристаллизации, подтвержденная опытами
          'wax_kinetics': 'True'}
NO_WAX = '1e-4'  # «без парафина»: нулевая доля выключила бы весь блок осаждения (`solver._paraphin`)
VARIANTS = {
    'base': {},                                     # закачка 20 C, Винсом-Вестервельд, скрытая теплота, кинетика
    'tiso': {'Twater': repr(T0)},                   # изотермическая закачка: без термических процессов
    't40': {'Twater': '40.0'},                      # чуть выше WAT живой нефти: охлаждение без парафина
    't5': {'Twater': '5.0'},
    'hl0': {'heat_losses': '0'},                    # без теплообмена с кровлей и подошвой
    'hl1': {'heat_losses': '1'},                    # схема Ловерье
    'nowax': {'init_Wp': NO_WAX},                   # нефть без парафина: эффект парафина
    'hl0_nowax': {'heat_losses': '0', 'init_Wp': NO_WAX},  # четвертый угол факторного плана
    'nogel': {'gelation': 'False'},                 # без предела текучести геля
    'ret': {'adsorption': 'True', 'ads_init_equilibrium': 'True'},  # удержание смол и асфальтенов
    'nolatent': {'latent_heat_mult': '0.0'},        # без скрытой теплоты кристаллизации
    'equil': {'wax_kinetics': 'False'},             # равновесная кристаллизация
    'nopress': {'wax_pressure': 'False'},           # равновесие без давления и газа: WAT и растворимость асфальтенов
    'ltne': {'thermal_nonequilibrium': 'True'},     # отдельная температура породы
}
LABELS = {
    'base': 'базовый: 20°C, Винсом—Вестервельд',
    'tiso': f'изотермическая закачка {T0:g}°C',
    't40': 'закачка 40°C',
    't5': 'закачка 5°C',
    'hl0': 'без теплообмена с кровлей и подошвой',
    'hl1': 'теплообмен по Ловерье',
    'nowax': 'нефть без парафина',
    'hl0_nowax': 'без парафина и без теплообмена',
    'nogel': 'без геля',
    'ret': 'с удержанием смол и асфальтенов',
    'nolatent': 'без скрытой теплоты',
    'equil': 'равновесная кристаллизация',
    'nopress': 'равновесие без давления и растворенного газа',
    'ltne': 'тепловое неравновесие (LTNE)',
}

RUNNER = '''import json, sys
sys.path.insert(0, sys.argv[1])
import start
start.solve(kin=json.loads(sys.argv[2]))
'''


def field_kin() -> dict:
    """Параметры кинетики из подборов по керну Li et al. (2024): удержание выше WAT и кристаллизация при 25 C."""
    import li2024
    params = json.loads((ROOT / 'experiments' / 'params.json').read_text(encoding='utf-8'))
    ret = params['li2024'][params['li2024']['best']]['kin']
    return li2024.cold_kin(params['li2024_cold']['x'], ret)


def run(name: str, threads: int) -> None:
    root = make_copy(f'thermal_{name}', dict(COMMON, **VARIANTS[name]))
    shutil.copy2(ROOT / 'start.py', root / 'start.py')
    (root / 'run_thermal.py').write_text(RUNNER, encoding='utf-8')
    log = root / 'run.log'
    print(f'--- {name}: {root} (лог {log})', flush=True)
    env = dict(os.environ, NUMBA_NUM_THREADS=str(threads))
    with open(log, 'w', encoding='utf-8') as f:
        subprocess.run([sys.executable, 'run_thermal.py', str(root), json.dumps(field_kin())], cwd=root, env=env,
                       check=True, stdout=f, stderr=subprocess.STDOUT)
    DATA.mkdir(parents=True, exist_ok=True)
    for src in (root / 'outputs' / 'data').glob('Wp=*'):
        # двойное подчеркивание отделяет имя варианта: иначе шаблон hl0 захватывал бы и hl0_nowax
        dst = DATA / src.name.replace(src.name.split('_')[0], f'thermal_{name}_', 1)
        shutil.copy2(src, dst)
    print(f'--- {name}: готово', flush=True)


def load(name: str):
    paths = sorted(DATA.glob(f'thermal_{name}__*processed_data.pkl'))
    if not paths:
        return None
    with open(paths[0], 'rb') as f:
        return pickle.load(f)[1]


def _fronts(d, idx, t_inj):
    """Фронты по диагонали нагнетательная - добывающая, [м]: вытеснения (насыщенность выросла на 0.05 от
    начальной), охлаждения (остыло на половину перепада) и ширина теплового фронта (остыло на 10-90 % перепада)."""
    from paraphin.constants import X_max, init_S
    s = np.diagonal(d['Saturation'][idx])
    t = np.diagonal(d['Temperature'][idx])
    step = X_max * np.sqrt(2.0) / len(s)
    wet = np.nonzero(s > init_S + 0.05)[0]
    if T0 == t_inj:
        return (step * (wet[-1] + 1) if wet.size else 0.0), 0.0, 0.0
    theta = (T0 - t) / (T0 - t_inj)  # доля перепада
    cold = np.nonzero(theta > 0.5)[0]
    width = step * int(((theta > 0.1) & (theta < 0.9)).sum())
    return (step * (wet[-1] + 1) if wet.size else 0.0), (step * (cold[-1] + 1) if cold.size else 0.0), width


def _first_time(time, flags):
    """Первый момент, когда условие выполнено, [годы]; None - не выполнено ни разу."""
    hit = np.nonzero(np.asarray(flags))[0]
    return float(time[hit[0]]) if hit.size else None


def _drop(q):
    """Признак: приемистость прошла 90 % своего падения от начальной до наименьшей."""
    return q <= q[0] - 0.9 * (q[0] - q.min()) if q[0] > q.min() else np.zeros_like(q, bool)


def metrics(name: str, d) -> dict:
    """Показатели варианта: КИН, обводненность, скин, приемистость, отложения, гель, фронты, во времени и на конец."""
    from paraphin.constants import geological_reserves, P_bubble
    from paraphin.layout import N_W
    t_inj = float(VARIANTS.get(name, {}).get('Twater', T_INJ))
    time = d['Time'] / 86400.0 / 365.0
    tot = d['Totals']
    wells, acc = d['Wells'], d['Wells_accumulated']
    comp = d['Composition']
    wax_dep = sum(np.asarray(tot[f'wax {k + 1} deposited']) for k in range(N_W))
    asph_dep = np.asarray(tot['asph flocs deposited']) + np.asarray(tot['resins deposited'])
    ads = np.asarray(tot.get('asph adsorbed', 0.0 * time)) + np.asarray(tot.get('resins adsorbed', 0.0 * time))
    ads = ads - ads[0]  # прирост удержания против начального равновесного
    last = len(time) - 1
    k_year = int(np.searchsorted(time, 1.0))
    front_w, front_t, width = _fronts(d, min(k_year, last), t_inj)
    # выборка кривых для рисунков - около 200 точек
    pick = np.unique(np.linspace(0, last, min(200, last + 1)).astype(int))
    series = {
        'years': time[pick],
        'rf': np.abs(acc['Producer_Q_oil'])[pick] / geological_reserves,
        'water_cut': np.asarray(wells['Producer_eta'])[pick],
        'skin_inj': np.asarray(wells['Injector_skin'])[pick],
        'skin_prod': np.asarray(wells['Producer_skin'])[pick],
        'q_inj': np.abs(np.asarray(wells['Injector_water']))[pick] * 86400.0,
        'wax_dep': wax_dep[pick] / 1e3,
        'asph_dep': (asph_dep + ads)[pick] / 1e3,
    }
    phi = d['Gel']['Phi'][last] if 'Gel' in d else np.ones(1)  # без геля (`nogel`) поле не сохраняется
    temp = d['Temperature'][last]
    return {
        'label': LABELS[name],
        'twater': t_inj,
        'rf': float(series['rf'][-1]),
        'water_cut': float(series['water_cut'][-1]),
        'injected': float(np.abs(acc['Injector_Q_water'][-1])),
        'skin_inj': float(series['skin_inj'][-1]),
        'skin_prod': float(series['skin_prod'][-1]),
        'k_inj': float(d['k'][last][0, 0]),
        'k_prod': float(d['k'][last][-1, -1]),
        'k_min': float(d['k'][last].min()),
        'wax_dep_t': float(wax_dep[-1] / 1e3),
        'wax_dep_groups': [float(np.asarray(tot[f'wax {k + 1} deposited'])[-1] / max(wax_dep[-1], 1e-30))
                           for k in range(N_W)],
        'asph_dep_t': float(asph_dep[-1] / 1e3),
        'retained_t': float(ads[-1] / 1e3),
        'wax_dep_max': float(d['Wps dep'][last].max()),
        'gel_area': float((phi < 0.5).mean()),
        'cold_area': float((temp < T0 - 0.5 * (T0 - t_inj)).mean()) if t_inj != T0 else 0.0,
        'below_wat_area': float((temp < comp['WAT'][last] + 0.01).mean()),
        'front_water_1y': front_w,
        'front_cold_1y': front_t,
        'front_width_1y': width,
        'T_mean': float(temp.mean()),
        'p_mean': float(d['Pressure'][last].mean() / 1e6),
        'wat0': [float(comp['WAT'][0].min()), float(comp['WAT'][0].max())],
        'years_below_pb': _first_time(time, [(d['Pressure'][n] < P_bubble).mean() > 0.5 for n in range(last + 1)]),
        'years_q_drop': _first_time(time, _drop(np.abs(np.asarray(wells['Injector_water'])))),
        'asph_dep_prod': float(comp['Asph dep'][last][-1, -1]),
        'ltne_dT_max': (float(max(np.abs(d['Temperature'][n] - comp['T rock'][n]).max() for n in range(last + 1)))
                        if 'T rock' in comp else None),
        'series': {k: [float(v) for v in vals] for k, vals in series.items()},
    }


def base_maps(d) -> dict:
    """Карты базового варианта в моменты MAP_YEARS для рисунков статьи (сетка 50 x 50 - компактно)."""
    time = d['Time'] / 86400.0 / 365.0
    idx = [min(int(np.searchsorted(time, y - 1e-6)), len(time) - 1) for y in MAP_YEARS]
    maps = {'years': [round(float(time[n]), 2) for n in idx]}
    for key, field in (('T', 'Temperature'), ('S', 'Saturation'), ('k', 'k'), ('wax_dep', 'Wps dep')):
        maps[key] = [np.round(np.asarray(d[field][n], float), 5).tolist() for n in idx]
    return maps


def collect() -> dict:
    out = {'kin': field_kin(), 'init_T': T0, 'variants': {}}
    for name in LABELS:
        d = load(name)
        if d is None:
            print(f'нет расчета {name}')
            continue
        out['variants'][name] = metrics(name, d)
        if name == 'base':
            out['maps'] = base_maps(d)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'thermal.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    for name, m in out['variants'].items():
        print(f"{name:9s} КИН {100 * m['rf']:5.2f} %  закачка {m['injected'] / 1e3:6.1f}  скин наг {m['skin_inj']:8.2f} "
              f"k наг {m['k_inj']:.3f}  парафин {m['wax_dep_t']:7.1f} т  асф+смолы {m['asph_dep_t']:6.1f} т  "
              f"удерж {m['retained_t']:6.1f} т  гель {100 * m['gel_area']:4.1f} %  ниже WAT {100 * m['below_wat_area']:4.1f} %")
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    names = [a for a in sys.argv[1:] if not a.startswith('--')] or list(VARIANTS)
    if '--metrics' not in sys.argv:
        workers = int(os.environ.get('THERMAL_WORKERS', '2'))
        threads = max(1, (os.cpu_count() or 2) // workers)
        with ThreadPoolExecutor(workers) as pool:
            for fut in [pool.submit(run, n, threads) for n in names]:
                fut.result()
    collect()


if __name__ == '__main__':
    main()
