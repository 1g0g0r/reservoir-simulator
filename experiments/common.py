"""Общая часть сравнений с опытами: постановка керна, параллельные прогоны в копиях пакета, кэш, метрики.

Каждый вариант - это (флаги и константы `constants.py`, параметры кинетики `solver.kin`, опыт). Флаги и константы
compile-time, поэтому вариант считается в копии пакета `outputs/.flags_on/<имя>` (`tests/_patched_copy.py`):
первая копия с новым набором констант компилируется ~40 с, дальше ее кэш numba переиспользуется. Параметры
кинетики меняются без перекомпиляции (`core_runner.py`), поэтому подбор перебирает их в одной копии.

Результаты кэшируются в `experiments/results/cache/` по хешу (константы, параметры, опыт, исходники пакета):
повторный запуск `run_all.py` без изменений ничего не считает, а правка кода или данных пересчитывает только
затронутое.
"""
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / 'data'
RESULTS = HERE / 'results'
CACHE = RESULTS / 'cache'
FIGURES = HERE / 'figures'
sys.path.insert(0, str(ROOT))
from tests._patched_copy import make_copy  # noqa: E402

RUNNER = HERE / 'core_runner.py'
PARAMS = HERE / 'params.json'  # итоговые параметры подборов: по ним режим --quick повторяет расчет без подбора
WORKERS = int(os.environ.get('EXPERIMENTS_WORKERS', max(1, (os.cpu_count() or 2))))
NY = 60      # ячеек вдоль керна
DT = 2.0     # стартовый и наибольший шаг, [с]
DARCY = 9.869233e-13


def load(name: str) -> dict:
    return json.loads((DATA / f'{name}.json').read_text(encoding='utf-8'))


def mode_from_argv(argv=None) -> str:
    """Режим модуля опыта: 'full' - с подбором, 'quick' - только итоговые наборы из params.json, 'plot' - рисунок."""
    argv = sys.argv[1:] if argv is None else argv
    return 'plot' if '--plot' in argv else 'quick' if '--quick' in argv else 'full'


def load_params(key: str) -> dict:
    """Итоговые параметры подбора `key` из params.json; без них режим --quick невозможен."""
    params = json.loads(PARAMS.read_text(encoding='utf-8')) if PARAMS.is_file() else {}
    if key not in params:
        raise SystemExit(f'в {PARAMS.name} нет подбора {key}: сначала полный прогон (без --quick)')
    return params[key]


def save_params(key: str, value: dict) -> None:
    params = json.loads(PARAMS.read_text(encoding='utf-8')) if PARAMS.is_file() else {}
    params[key] = value
    PARAMS.write_text(json.dumps(params, ensure_ascii=False, indent=1, sort_keys=True), encoding='utf-8')


def save_results(name: str, out: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f'{name}.json').write_text(json.dumps(out, ensure_ascii=False, default=float), encoding='utf-8')


def load_results(name: str) -> dict:
    path = RESULTS / f'{name}.json'
    if not path.is_file():
        raise SystemExit(f'нет results/{name}.json: сначала python experiments/{name}.py')
    return json.loads(path.read_text(encoding='utf-8'))


# --- Постановка керна -------------------------------------------------------------------------------------------

def core_constants(exp: dict, ny: int = NY, dt: float = DT) -> dict:
    """Константы `constants.py` одномерного керна (как `experiments/исходная_модель/core_flood._case_constants`).

    exp: length, side, porosity, k0 [Д], T, q, pv_end, w, MW, M_o, Tm, dH, ro_o, ro_p, mu; опционально T_hot.
    Нефть однофазная (S_min = 0, S_max = 1), керн изотермический при T, равновесная взвесь и в керне, и на входе.
    """
    from paraphin.constants import R  # noqa: F401  (проверка, что пакет импортируется)
    import importlib.util
    spec = importlib.util.spec_from_file_location('core_flood', HERE / 'исходная_модель' / 'core_flood.py')
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    w_sat = float(cf.w_saturated(exp['w'], exp['T'], exp['MW'], exp['M_o'], exp['Tm'], exp['dH']))
    t_end = exp['pv_end'] * exp['length'] * exp['side'] ** 2 * exp['porosity'] / exp['q']
    return {
        'Nx, Ny': f'1, {ny}',
        'X_min, X_max': f'0.0, {exp["side"]!r}',
        'Y_min, Y_max': f'0.0, {exp["length"]!r}',
        'h': repr(exp['side']),
        'Time_end': repr(t_end),
        'dt': repr(dt),
        'sol_time_step': repr(10.0 * t_end),
        'Twater': repr(exp['T']),
        'S_min': '0.0',
        'S_max': '1.0',
        'heat_losses': '0',
        'init_Wp': repr(w_sat),
        'init_Wps': repr(exp['w'] - w_sat),
        'init_k': f'{exp["k0"]!r} * darcy_to_m2',
        'init_m': repr(exp['porosity']),
        'init_T': repr(exp['T']),
        'ro_o': repr(exp['ro_o']),
        'ro_p': repr(exp['ro_p']),
        'mu_o_ref': repr(exp['mu']),
        'T_mu_ref': repr(exp['T']),
        'MW': repr(exp['MW']),
        'M_o': repr(exp['M_o']),
        'Tm': repr(exp['Tm']),
        'alpha': repr(exp['dH']),
    }


# --- Прогоны ---------------------------------------------------------------------------------------------------

def _source_hash() -> str:
    h = hashlib.sha256()
    for path in sorted((ROOT / 'paraphin').rglob('*.py')):
        h.update(path.read_bytes())
    h.update(RUNNER.read_bytes())
    return h.hexdigest()[:16]


_SRC = None


def _key(constants: dict, case: dict) -> str:
    global _SRC
    if _SRC is None:
        _SRC = _source_hash()
    blob = json.dumps({'c': constants, 'case': case, 'src': _SRC}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:24]


def run_many(jobs):
    """Прогоны [(имя копии, константы, case)] параллельно (до WORKERS). Возвращает список результатов по порядку.

    Копии пакета готовятся последовательно (запись файлов), прогоны идут параллельно, по одному потоку numba на
    прогон: одномерному керну из 60 ячеек потоки не помогают, а несколько процессов со всеми потоками мешают
    друг другу (переподписка ядер замедляла шаг в разы)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    todo, results = [], [None] * len(jobs)
    for n, (copy_name, constants, case) in enumerate(jobs):
        key = _key(constants, case)
        cached = CACHE / f'{key}.json'
        if cached.is_file():
            results[n] = json.loads(cached.read_text(encoding='utf-8'))
        else:
            todo.append((n, copy_name, constants, case, cached))

    # Одна копия на набор констант; сначала по одному прогону на копию - компиляция без гонок за кэш numba
    roots = {}
    for _, copy_name, constants, _, _ in todo:
        if copy_name not in roots:
            roots[copy_name] = make_copy(copy_name, constants)

    def work(item):
        n, copy_name, constants, case, cached = item
        root = roots[copy_name]
        tag = cached.stem
        case_path, out_path = root / f'case_{tag}.json', root / f'out_{tag}.json'
        case_path.write_text(json.dumps(case, ensure_ascii=False), encoding='utf-8')
        env = dict(os.environ, PYTHONPATH=str(root), NUMBA_NUM_THREADS='1')
        proc = subprocess.run([sys.executable, str(RUNNER), str(case_path), str(out_path)], cwd=root, env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f'{copy_name}: прогон упал\n{proc.stderr[-3000:]}')
        res = json.loads(out_path.read_text(encoding='utf-8'))
        case_path.unlink()
        out_path.unlink()
        cached.write_text(json.dumps(res), encoding='utf-8')
        return n, res

    first, rest, seen = [], [], set()
    for item in todo:
        (rest if item[1] in seen else first).append(item)
        seen.add(item[1])
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for batch in (first, rest):
            for n, res in pool.map(work, batch):
                results[n] = res
    return results


# --- Метрики ---------------------------------------------------------------------------------------------------

def rms_k(res: dict, points) -> float:
    """СКО модели от опытных точек k/k0(PV); точка PV = 0 не считается (там 1 по определению). После
    закупорки модельная кривая считается нулем."""
    pv, k = np.array(res['pv']), np.array(res['k'])
    pts = np.array([p for p in points if p[0] > 0.0])
    model = np.where(pts[:, 0] <= pv[-1] + 1e-12, np.interp(pts[:, 0], pv, k), 0.0 if res['plugged'] else k[-1])
    return float(np.sqrt(np.mean((model - pts[:, 1]) ** 2)))


def noise_floor(points) -> float:
    """Собственный разброс опытных точек около гладкой кривой - нижняя граница достижимой СКО.

    Оценка по вторым разностям (метод фон Неймана): у гладкого сигнала на равномерной сетке вторая разность
    почти ноль, а у независимого шума с дисперсией s^2 ее дисперсия 6*s^2, поэтому s^2 ~ sum(d2^2)/(6*(n - 2)).
    Тренд (спад k/k0) в отличие от первых разностей оценку не завышает. Модель, которая проходит через точки
    ближе этого, подгоняется под шум оцифровки и разброс опыта."""
    pts = np.array([p for p in points if p[0] > 0.0])
    d2 = np.diff(pts[:, 1], 2)
    return float(np.sqrt(np.sum(d2 * d2) / (6.0 * len(d2))))


def interp_at(res: dict, x_points):
    pv, k = np.array(res['pv']), np.array(res['k'])
    return [float(np.interp(x, pv, k)) if x <= pv[-1] else 0.0 for x in x_points]


# --- Подбор ----------------------------------------------------------------------------------------------------

def lsq(label, jobs_of, residual_of, x0, step, lo, hi, max_nfev=12, ftol=1e-3, xtol=1e-3):
    """least_squares по прогонам с параллельным разностным якобианом.

    jobs_of(x) -> список заданий `run_many` для одной точки (например, оба опыта семейства), residual_of(results) ->
    вектор невязок по этим прогонам. Прогоны точек якобиана (n + 1 точка) идут одним `run_many`, повторные точки
    берутся из кэша. Возвращает (x, results лучшей точки, СКО)."""
    from scipy.optimize import least_squares
    step = np.asarray(step, float)

    def runs(xs):
        jobs, sizes = [], []
        for x in xs:
            js = jobs_of(np.asarray(x, float))
            jobs += js
            sizes.append(len(js))
        res, out, pos = run_many(jobs), [], 0
        for n in sizes:
            out.append(res[pos:pos + n])
            pos += n
        return out

    def fun(x):
        r = residual_of(runs([x])[0])
        print(f'  {label}: x = {np.round(x, 3).tolist()}, СКО {np.sqrt(np.mean(r ** 2)):.4f}', flush=True)
        return r

    def jac(x):
        xs = [np.asarray(x, float)] + [np.asarray(x, float) + step[n] * np.eye(len(x))[n] for n in range(len(x))]
        rs = [residual_of(r) for r in runs(xs)]
        return np.array([(rs[n + 1] - rs[0]) / step[n] for n in range(len(x))]).T

    fit = least_squares(fun, np.clip(np.asarray(x0, float), lo, hi), jac=jac, bounds=(lo, hi), max_nfev=max_nfev,
                        x_scale=step * 5, ftol=ftol, xtol=xtol)
    best = runs([fit.x])[0]
    return [float(v) for v in fit.x], best, float(np.sqrt(np.mean(residual_of(best) ** 2)))


def k_residuals(res, points):
    """Модель минус опыт k/k0 во всех точках (PV > 0), после закупорки модель - ноль (как `rms_k`)."""
    pv, k = np.array(res['pv']), np.array(res['k'])
    pts = np.array([p for p in points if p[0] > 0.0])
    model = np.where(pts[:, 0] <= pv[-1] + 1e-12, np.interp(pts[:, 0], pv, k), 0.0 if res['plugged'] else k[-1])
    return model - pts[:, 1]
