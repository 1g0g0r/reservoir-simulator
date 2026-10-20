"""Интерактивная визуализация больших расчётных данных (Dash) с ленивой загрузкой.

Слева - график (карта поля или временные ряды), под ним подпись времени и ползунок; правее - столбец кнопок видов,
еще правее - дополнительный график. Стили и клиентские колбэки - в `assets/` (style.css, viz.js). Точка входа и
порт - `visualise_solution.py` в корне проекта.

Карта - одно поле заливкой (кнопка справа) и любые поля изолиниями поверх. Настройки отображения - в
`map-settings`, от них пересобирается фигура (`_redraw`). Дополнительный график (`AUX`): величины в выбранной
щелчком ячейке во времени, профиль вдоль линии или φ(r) в ячейке на текущем слое.

Ползунок (`_on_slider`) шлет только данные слоя (Patch), и при перетаскивании тоже, но через клиентскую заслонку
`viz.gate`: в работе не больше одного запроса, новейшее положение перекрывает старые, частота кадров сама
подстраивается под скорость сервера. Простой updatemode='drag' слал бы запрос на каждый шаг: dash отбрасывает
устаревшие ответы, но сервер считает все. Подпись времени обновляется на клиенте, без сервера.

Данные: кэш pkl -> npy (float32, memory-mapped), собирается при запуске и удаляется при выходе. φ(r) приходит в
склейке `read_data_files._glue_fi` (только ячейки, где fi отошел от fi_0) или полным полем (n, Nx, Ny, Nr) - кэш
приводит его к тому же виду. Сохранение: HTML - полная анимация, SVG и EPS - картинка текущего слоя.
"""
from __future__ import annotations

import gc
import math
import re
import shutil
import subprocess
import tempfile
import threading
import traceback
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from dash import ALL, ClientsideFunction, Dash, Input, Output, Patch, State, ctx, dcc, html, no_update

from paraphin.geometry import r, fi_0
from paraphin.constants import Nx, Ny, X_min, X_max, hx, hy, Y_max, Y_min, bar_to_pa, day_to_sec, CONTOUR_PLOT, S_max
from paraphin.utils.read_data_files import (read_solution_data, convert_pkl_files, results_path, layers_file,
                                           case_name, FI_CELL, fi_curve)

ASSETS = Path(__file__).parent / 'assets'
js_path = ASSETS / 'plotly_script.js'   # plotly.js для «Сохранить в HTML»; на страницу не подключается
PDFTOPS = shutil.which('pdftops')       # EPS - из PDF через pdftops (poppler, есть в MiKTeX); без него кнопки EPS нет

# ----------------------------- Настройки ------------------------------------
CACHE_DIR = results_path.parent / f'{case_name}_vizcache'
DTYPE = np.float32
MAX_DISPLAY_CELLS = 250_000             # максимум точек сетки для экрана (None — без прореживания)
MAX_MARKS = 7                           # подписей под слайдером, как в оригинале
JSON_DECIMALS = 5
# Размеры фигур; контейнеры под них - в assets/style.css (+10 px)
GRAPH_W, GRAPH_H = 720, 800             # ширина карты; высота временных рядов
MAP_H = 660                             # высота карты: квадратная область уже GRAPH_H, под ней была пустая полоса
WIDE_W = 950                            # временные ряды шире карты
CTRL_SHIFT = 50                         # сдвиг подписи времени, ползунка и настроек вправо от края фигуры, px
AUX_H = 330                             # высота дополнительного графика
AUX_FI_H = MAP_H - 40                   # φ(r) - во всю высоту карты (колонка начинается на 40 px ниже, вровень с кнопками)
AUX_MAX_W = 1100                        # доп. график растягивается по свободному месту справа, но не шире
AUX_EXPORT_W = 900                      # ширина доп. графика в SVG/EPS (на экране она по месту)
SERIES_GROUPS = ('Wells', 'Wells_accumulated', 'Totals', 'Other params')
SERIES_TITLES = {'Wells': 'Скважины', 'Wells_accumulated': 'Накопленная добыча', 'Totals': 'Массы компонентов',
                 'Other params': 'Прочие ряды'}

# Разделы столбца кнопок: поле попадает в первый подходящий. Имена полей - из `save_data_fields`.
_WAX_GROUP = re.compile(r'^Wax \d+( susp| dep)?$')
FIELD_SECTIONS = (
    ('Пласт', lambda n: n in ('Pressure', 'Saturation', 'Temperature', 'm', 'k', 'm conductive', 'Wo', 'T rock',
                              'Free gas')),
    ('Парафин', lambda n: n in ('Wp', 'Wps', 'Wps dep', 'WAT', 'Supersaturation', 'Wax dep rate', 'Wall cryst rate')
                          or bool(_WAX_GROUP.match(n))),
    ('Асфальтены и смолы', lambda n: n.startswith(('Asph', 'Resins', 'Adsorbed', 'Floc')) or n in ('CII',
                                                                                                    'Wettability omega')),
    ('Гель', lambda n: n in ('Phi', 'mu_o', 'mu_p', 'tau_y') or n.startswith('Gel')),
    ('Прочее', lambda n: True),
)

x_mesh = np.linspace(X_min + hx / 2, X_max - hx / 2, Nx)
y_mesh = np.linspace(Y_min + hy / 2, Y_max - hy / 2, Ny)
# Знаков после запятой в координатах ячейки: столько, чтобы соседние центры различались (2.7 м - целые, керн - мм)
_XY_DEC = max(0, -math.floor(math.log10(min(hx, hy))))

PALETTES = ('Jet', 'Turbo', 'Viridis', 'Plasma', 'RdBu', 'Greys')
LINE_COLORS = ('black', 'white', 'red', 'magenta', 'cyan', 'lime', 'blue', 'green', 'orange', 'gray')
OV_COLORS = LINE_COLORS[:6]             # цвета изолиний по умолчанию - по очереди
PROFILES = ('diag', 'x', 'y')           # значения `aux` для профиля: диагональ, вдоль X, вдоль Y
NEEDS_CELL = ('cell', 'fi')             # режимы доп. графика по выбранной ячейке

# Настройки отображения (`map-settings`); `ov` - стиль изолиний по имени поля, `cell` - (i, j) выбранной ячейки
DEFAULT_SETTINGS = {
    'overlays': [], 'mode': 'fill' if CONTOUR_PLOT else 'heatmap', 'cmap': 'Jet', 'ncont': 10, 'labels': True,
    'range': 'all', 'zmin': None, 'zmax': None, 'log': False, 'ov': {},
    'aux': 'none', 'cell': None,
    'series_log': False, 'series_norm': False,
}
# Ключ настройки -> id элемента управления (свойство 'value'); `_collect` собирает из них `map-settings`
SETTING_INPUTS = {'overlays': 'overlays', 'mode': 'set-mode', 'cmap': 'set-cmap', 'log': 'set-log',
                  'ncont': 'set-ncont', 'labels': 'set-labels', 'range': 'set-range', 'zmin': 'set-zmin',
                  'zmax': 'set-zmax', 'aux': 'aux-mode', 'series_log': 'series-log', 'series_norm': 'series-norm'}
_FLAGS = ('log', 'labels', 'series_log', 'series_norm')   # чекбоксы: список значений -> bool


def _day_fmt(t) -> str:
    """Формат времени как в оригинале: f'{round(time[i], 1)} день'."""
    return f'{round(float(t), 1)} день'


def _cell_label(ci: int, cj: int) -> str:
    """Подпись ячейки на доп. графике - координаты ее центра в метрах, а не индексы."""
    return f'x = {x_mesh[ci]:.{_XY_DEC}f} м, y = {y_mesh[cj]:.{_XY_DEC}f} м'


def _sanitize(name: str) -> str:
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in name)


def _safe_minmax(a: np.ndarray) -> tuple[float, float]:
    with np.errstate(invalid='ignore'):
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return (lo, hi) if (math.isfinite(lo) and math.isfinite(hi)) else (0.0, 1.0)


def _pad(lo: float, hi: float) -> list[float]:
    """Диапазон оси с запасом 3 %."""
    pad = 0.03 * (hi - lo) or 0.03 * abs(hi) or 1.0
    return [lo - pad, hi + pad]


# ============================ Слой данных ====================================
def _load_raw() -> dict:
    """Чтение исходных данных (тяжёлая операция — только при сборе кэша)."""
    name = f'{case_name}_processed_data.pkl'
    try:
        # Файл слоев остается на диске, только если расчет не дошел до склейки: значит данные
        # свежее обработанного файла и их надо собрать заново
        if layers_file.is_file():
            convert_pkl_files()
        n_times, data = read_solution_data(name)
    except ValueError:
        convert_pkl_files()
        n_times, data = read_solution_data(name)
    print('Временных слоев:', n_times)
    return data


def _series_row(g: str, k: str, v: np.ndarray) -> tuple[bool, np.ndarray, str, str]:
    """Кривая `k` группы рядов `g` для экрана: (видна ли по умолчанию, значения, ось, единицы)."""
    if g == 'Wells':
        visible = not np.allclose(v, v.flat[0], rtol=0, atol=1e-9)
        if 'eta' in k:
            return visible, v, 'y2', ''
        if 'bhp' in k:
            return visible, v / bar_to_pa, 'y', 'бар'   # `save_fields` пишет забойное давление в Па
        return visible, np.abs(v) * day_to_sec, 'y', 'м^3/день'
    if g == 'Wells_accumulated':
        return not np.all(np.isclose(v, 0.0)), np.abs(v), 'y', 'м^3'
    if g == 'Totals':   # массы в нефти на порядки больше отложений: отложения и адсорбция - на правой оси
        return bool(np.any(v != 0.0)), v, 'y' if 'in oil' in k else 'y2', 'кг'
    return True, v, 'y', ''


class SolutionStore:
    """Метаданные — в RAM, массивы — memory-mapped npy в CACHE_DIR; срезы читаются по требованию."""

    def __init__(self):
        self._lock = threading.RLock()
        self._mmaps: dict[str, np.ndarray] = {}
        self._series: dict[str, np.ndarray] = {}
        self._fi: tuple | None = None   # (ячейка, ее кривые fi) - последняя запрошенная

        raw = _load_raw()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        time = np.asarray(raw.pop('Time'), dtype=np.float64) / day_to_sec
        self.meta = {'n_times': len(time), 'time': [round(float(t), 6) for t in time]}
        _prepare(raw)
        self.meta['fields'] = self._cache_fields(raw)
        self.meta['series'] = self._cache_series(raw)
        self.meta['has_plots'] = self._cache_fi(raw.get('plots') or {})
        del raw
        gc.collect()

        # Сетка и прореживание для экрана
        ny, nx = (next(iter(self.meta['fields'].values()))['shape'][1:] if self.meta['fields'] else (Ny, Nx))
        self.stride = 1 if MAX_DISPLAY_CELLS is None else max(1, math.ceil(math.sqrt(nx * ny / MAX_DISPLAY_CELLS)))
        self.time = np.asarray(self.meta['time'], dtype=np.float64)
        self.x, self.y = x_mesh[::self.stride], y_mesh[::self.stride]

    # ------------------------------- кэш ------------------------------------
    @staticmethod
    def _cache_fields(raw: dict) -> dict:
        """3D-поля - потоковой записью в npy, полный куб в RAM не дублируется."""
        fields = {}
        for name in [k for k in raw if k not in SERIES_GROUPS and k != 'plots']:
            arr = np.asarray(raw.pop(name))
            if arr.ndim != 3:
                print(f'  поле "{name}" пропущено (ожидался куб n×Ny×Nx, ndim={arr.ndim})')
                continue
            fname = f'field__{_sanitize(name)}.npy'
            # Поле расчета - [i=x, j=y], а plotly читает z как [строка=y][столбец=x]: в кэш кладется транспонированным
            shape = (arr.shape[0], arr.shape[2], arr.shape[1])
            mm = np.lib.format.open_memmap(CACHE_DIR / fname, mode='w+', dtype=DTYPE, shape=shape)
            for i in range(arr.shape[0]):
                mm[i] = arr[i].T
            mm.flush()
            del mm
            pos = arr[arr > 0]
            fields[name] = {'file': fname, 'shape': list(shape), **dict(zip(('zmin', 'zmax'), _safe_minmax(arr))),
                            'zpos': float(pos.min()) if pos.size else 1e-12}   # низ логарифмической шкалы
            print(f'  поле "{name}": {arr.shape} -> {fname}')
            del arr
        return fields

    def _cache_series(self, raw: dict) -> dict:
        """Временные ряды по группам: одна матрица (кривая x момент) на группу."""
        n_times, series = self.meta['n_times'], {}
        for g in SERIES_GROUPS:
            names, axes, units, vis, rows = [], [], [], [], []
            for k, v in (raw.get(g) or {}).items():
                v = np.asarray(v, dtype=np.float64).reshape(-1)
                if v.size == 1:
                    v = np.full(n_times, float(v[0]))
                if v.size != n_times:
                    print(f'  {g}/{k}: длина {v.size} != {n_times}, пропущено')
                    continue
                visible, y, ax, unit = _series_row(g, k, v)
                names.append(k); axes.append(ax); units.append(unit); vis.append(visible)
                rows.append(y.astype(DTYPE))
            if names:
                fname = f'series__{_sanitize(g)}.npy'
                np.save(CACHE_DIR / fname, np.asarray(rows, dtype=DTYPE))
                series[g] = {'file': fname, 'names': names, 'axes': axes, 'units': units, 'visible': vis}
                print(f'  {g}: {len(names)} кривых')
        return series

    @staticmethod
    def _cache_fi(plots: dict) -> bool:
        """fi(r) - в виде склейки `read_data_files._glue_fi`: 'cells' - ячейки, где fi хоть раз отошел от fi_0,
        'fi' - (n, n_cells, Nr). Полное поле (n, Nx, Ny, Nr) приводится к тому же виду."""
        if 'fi' not in plots:
            return False
        if 'cells' not in plots:
            fi = np.asarray(plots['fi'])
            fi = fi.reshape(fi.shape[0], -1, fi.shape[-1])
            cells = np.flatnonzero((fi != fi_0.astype(fi.dtype)).any(axis=(0, 2)))
            plots = {'cells': cells, 'fi': fi[:, cells]}
        np.save(CACHE_DIR / 'fi_cells.npy', np.asarray(plots['cells']))
        np.save(CACHE_DIR / 'fi.npy', np.asarray(plots['fi'], dtype=DTYPE))
        return True

    # ---------------------------- доступ к данным ----------------------------
    def _mmap(self, fname: str) -> np.ndarray:
        with self._lock:
            if fname not in self._mmaps:   # с диска читаются только нужные страницы
                self._mmaps[fname] = np.load(CACHE_DIR / fname, mmap_mode='r')
            return self._mmaps[fname]

    def _field(self, name: str) -> np.ndarray:
        return self._mmap(self.meta['fields'][name]['file'])

    @staticmethod
    def _round(a: np.ndarray) -> np.ndarray:
        return a if JSON_DECIMALS is None else np.round(a.astype(np.float64, copy=False), JSON_DECIMALS)

    def get_field(self, name: str, i: int) -> np.ndarray:
        """Срез поля в момент i (с прореживанием, без округления), z[j, i]."""
        return np.asarray(self._field(name)[i, ::self.stride, ::self.stride], dtype=np.float64)

    def get_cell(self, name: str, i: int, j: int) -> np.ndarray:
        """Величина в ячейке (i, j) во все моменты: из memmap читается только этот столбец."""
        return np.asarray(self._field(name)[:, j, i], dtype=np.float64)

    def get_profile(self, name: str, t: int, ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
        """Значения в ячейках (ii[k], jj[k]) в момент t."""
        return np.asarray(self._field(name)[t][jj, ii], dtype=np.float64)

    def get_series(self, group: str) -> np.ndarray:
        with self._lock:
            if group not in self._series:
                self._series[group] = np.load(CACHE_DIR / self.meta['series'][group]['file']).astype(np.float64)
            return self._series[group]

    def get_fi(self, cell) -> np.ndarray:
        """Кривые fi(r) в ячейке `cell` (i, j) во все моменты, (n_times, Nr); без ячейки - в `FI_CELL`. Из memmap
        читается только эта ячейка (`fi_curve`)."""
        cell = tuple(cell or FI_CELL)
        with self._lock:
            if self._fi is None or self._fi[0] != cell:
                plots = {'cells': self._mmap('fi_cells.npy'), 'fi': self._mmap('fi.npy')}
                self._fi = cell, np.asarray(fi_curve(plots, *cell), dtype=np.float64)
            return self._fi[1]


def _prepare(raw: dict) -> None:
    """Исходные данные - к виду для экрана: группы состава и геля - в общий список полей, скин - в прочие ряды,
    давление - в бар, взвесь Wps - на нефтяную фазу."""
    # Детальный состав нефти и гель (группы парафина, асфальтены, `gelation`): поля - картами, массы ('Totals') - рядами
    for group in ('Composition', 'Gel'):
        raw.update(raw.pop(group, {}))
    # Скин-фактор - безразмерный и бывает отрицательным: не к дебитам, а в прочие ряды
    wells = raw.get('Wells') or {}
    for k in [k for k in wells if k.endswith('_skin')]:
        raw.setdefault('Other params', {})[k] = wells.pop(k)
    if 'Pressure' in raw:
        raw['Pressure'] = np.asarray(raw['Pressure']) / bar_to_pa
    if 'plots' in raw and 'Wps' in raw and 'Saturation' in raw:
        raw['Wps'] = np.asarray(raw['Wps']) * (S_max - np.asarray(raw['Saturation']))


_STORE: SolutionStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> SolutionStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = SolutionStore()
    return _STORE


# ============================ Карта ==========================================
def _axis_dict() -> dict:
    """Оси в стиле оригинала: чёрная сетка, чёрная рамка, шрифт подписи 18."""
    return dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1, title_font=dict(size=18))


def _map_axes() -> tuple[dict, dict]:
    """Оси карты: квадратная область, диапазон - autorange по трассе.

    constrain='domain' обязателен на ОБЕИХ осях: по умолчанию ('range') ось,
    подогнанная под scaleanchor, РАСШИРЯЕТ свой диапазон — из-за этого сетка
    выходила за границы области данных.

    Явного диапазона нет: Heatmap дотягивает крайние ячейки до границ домена (X_min..X_max), Contour
    занимает только отрезок от первой до последней точки сетки, и диапазоны у них разные. Явный
    диапазон plotly при переключении Contour <-> Heatmap (Plotly.react) применял только к оси x, а ось y
    со scaleanchor оставляла прежний: контур «приподнимался» над осью x на пол-ячейки до двойного
    щелчка. Autorange пересчитывается при react на обеих осях (проверено в Edge).
    """
    xax = _axis_dict() | dict(autorange=True, constrain='domain')
    # Квадратная область уже высоты фигуры: constrain='domain' по умолчанию ставит ее посередине, ниже кнопок
    yax = _axis_dict() | dict(autorange=True, scaleanchor='x', scaleratio=1, constrain='domain', constraintoward='top')
    return xax, yax


def _hover(name: str) -> str:
    return f'X: %{{x}}<br>Y: %{{y}}<br>{name}: %{{z}}<extra></extra>'


def _overlays(st: dict, base: str, store: SolutionStore) -> list[str]:
    """Поля изолиний поверх заливки (без самой заливки)."""
    return [n for n in st['overlays'] if n != base and n in store.meta['fields']]


def _ov_style(st: dict, name: str) -> dict:
    """Стиль изолиний поля; цвет по умолчанию - по месту поля в списке наложенных."""
    k = st['overlays'].index(name) if name in st['overlays'] else 0
    s = st['ov'].get(name) or {}
    return {'color': s.get('color') or OV_COLORS[k % len(OV_COLORS)], 'n': s.get('n') or 5,
            'width': s.get('width') or 2, 'labels': s.get('labels', True)}


def _levels(lo: float, hi: float, n: int) -> dict:
    """n изолиний внутри [lo, hi]. Уровни явные: автоуровни plotly прыгают от кадра к кадру."""
    size = (hi - lo) / (n + 1) if hi > lo else 1e-12
    return dict(start=lo + size, end=hi - size if hi > lo else lo, size=size)


def _layer(store: SolutionStore, name: str, t: int, st: dict, is_base: bool) -> tuple[np.ndarray, float, float]:
    """Срез поля для экрана (lg - только у заливки) и диапазон шкалы по настройке."""
    info = store.meta['fields'][name]
    z = store.get_field(name, t)
    log = is_base and st['log']
    if log:
        z = np.log10(np.maximum(z, info['zpos']))
    if is_base and st['range'] == 'manual' and st['zmin'] is not None and st['zmax'] is not None:
        lo, hi = float(st['zmin']), float(st['zmax'])   # в единицах шкалы, то есть в lg при логарифмической
    elif st['range'] == 'layer':
        lo, hi = _safe_minmax(z)
    else:
        lo, hi = info['zmin'], info['zmax']
        if log:
            lo, hi = math.log10(max(lo, info['zpos'])), math.log10(max(hi, info['zpos']))
    return store._round(z), lo, hi


def _trace_data(store: SolutionStore, base: str, overlays: list[str], t: int, st: dict) -> list[dict]:
    """Что меняется у трасс карты от слоя к слою: z, а при шкале по текущему слою - и диапазон с уровнями."""
    out = []
    for k, name in enumerate([base] + overlays):
        z, lo, hi = _layer(store, name, t, st, k == 0)
        d = {'z': z}
        if st['range'] == 'layer':
            d.update(zmin=lo, zmax=hi)
            if k > 0 or st['mode'] != 'heatmap':
                d['contours'] = _levels(lo, hi, st['ncont'] if k == 0 else _ov_style(st, name)['n'])
        out.append(d)
    return out


def _line_cells(st: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ячейки профиля (ii, jj) и расстояние вдоль линии, м. Вдоль X - строка, вдоль Y - столбец ячейки, выбранной
    щелчком по карте (до щелчка - нулевые)."""
    ci, cj = st['cell'] or (0, 0)
    if st['aux'] == 'x':
        jj = int(min(max(cj, 0), Ny - 1))
        return np.arange(Nx), np.full(Nx, jj), x_mesh - x_mesh[0]
    if st['aux'] == 'y':
        ii = int(min(max(ci, 0), Nx - 1))
        return np.full(Ny, ii), np.arange(Ny), y_mesh - y_mesh[0]
    f = np.linspace(0.0, 1.0, max(Nx, Ny))   # диагональ (0, 0) -> (Nx-1, Ny-1)
    ii, jj = np.rint(f * (Nx - 1)).astype(int), np.rint(f * (Ny - 1)).astype(int)
    return ii, jj, np.hypot(x_mesh[ii] - x_mesh[0], y_mesh[jj] - y_mesh[0])


def _field_bounds(c: np.ndarray, mode: str):
    """Прижатие координаты к области поля по оси с узлами `c`: у Heatmap - края крайних ячеек, у Contour - их центры."""
    h = (c[1] - c[0]) / 2 if mode == 'heatmap' and len(c) > 1 else 0.0
    lo, hi = float(c[0] - h), float(c[-1] + h)
    return lambda v: min(max(float(v), lo), hi)


def _map_marks(store: SolutionStore, st: dict) -> list:
    """Отметка выбранной ячейки или линия профиля поверх карты.

    К маркерам и shape plotly добавляет в autorange запас (5 % и размер маркера, у shape - толщину линии), и после
    щелчка по крайней ячейке карта съезжала на ~12 м из 200. Поэтому обе - Scatter из одних линий по замкнутому
    пути с прозрачной заливкой 'tozerox': без маркеров снимается запас по x, а замкнутый путь с x-заливкой снимает
    его и по y, не притягивая ось к нулю (`calcAxisExpansion` в plotly.js). Концы прижаты к границам поля."""
    mark = dict(mode='lines', fill='tozerox', fillcolor='rgba(0,0,0,0)', showlegend=False, hoverinfo='skip',
                cliponaxis=False)
    if st['aux'] in NEEDS_CELL and st['cell']:
        ci, cj = st['cell']
        cx, cy, d = x_mesh[ci], y_mesh[cj], 1.5 * max(hx, hy)
        xc, yc = _field_bounds(store.x, st['mode']), _field_bounds(store.y, st['mode'])
        xs = [xc(cx - d), xc(cx + d), None, cx, cx, None, xc(cx - d)]   # последняя точка замыкает путь
        ys = [cy, cy, None, yc(cy - d), yc(cy + d), None, cy]
        return [go.Scatter(x=xs, y=ys, line=dict(color=color, width=width), **mark)   # черный крест с белой каймой
                for width, color in ((5, 'white'), (2.5, 'black'))]
    if st['aux'] in PROFILES:
        ii, jj, _ = _line_cells(st)
        x0, y0, x1, y1 = x_mesh[ii[0]], y_mesh[jj[0]], x_mesh[ii[-1]], y_mesh[jj[-1]]
        return [go.Scatter(x=[x0, x1, x0], y=[y0, y1, y0],   # туда и обратно - замкнутый путь
                           line=dict(color='black', width=2, dash='dash'), **mark)]
    return []


def _animation_slider(store: SolutionStore, base: str, overlays: list[str], i: int, st: dict) -> dict:
    """Ползунок plotly со всеми слоями - для сохранения в HTML."""
    idx = list(range(1 + len(overlays)))
    steps = []
    for j in range(store.meta['n_times']):
        ds = _trace_data(store, base, overlays, j, st)
        args = {'z': [d['z'] for d in ds]}
        if st['range'] == 'layer':
            args['zmin'], args['zmax'] = [d['zmin'] for d in ds], [d['zmax'] for d in ds]
            for key in ('start', 'end', 'size'):
                args[f'contours.{key}'] = [d['contours'][key] if 'contours' in d else None for d in ds]
        steps.append(dict(method='restyle', args=[args, idx], label=_day_fmt(store.time[j])))
    return dict(active=min(i, len(steps) - 1), currentvalue={'prefix': 'Время: '}, steps=steps)


def _map_figure(base: str, i: int, store: SolutionStore, st: dict, animate: bool = False) -> go.Figure:
    """Карта: поле `base` заливкой (контур, линии или Heatmap) и изолинии полей `st['overlays']` поверх."""
    overlays = _overlays(st, base, store)
    z, lo, hi = _layer(store, base, i, st, True)
    common = dict(x=store.x, y=store.y, z=z, name=base, colorscale=st['cmap'], zmin=lo, zmax=hi,
                  hovertemplate=_hover(base), colorbar=dict(title='lg' if st['log'] else None))  # имя поля - на кнопке
    if st['mode'] == 'heatmap':
        traces = [go.Heatmap(**common)]
    else:
        traces = [go.Contour(**common, autocontour=False,
                             contours=dict(coloring=st['mode'], showlabels=st['labels'],
                                           labelfont=dict(size=12, color='black'), **_levels(lo, hi, st['ncont'])),
                             line=dict(width=2) if st['mode'] == 'lines' else None)]
    for name in overlays:
        s = _ov_style(st, name)
        z_o, lo_o, hi_o = _layer(store, name, i, st, False)
        traces.append(go.Contour(
            x=store.x, y=store.y, z=z_o, name=name, autocontour=False,
            contours=dict(coloring='lines', showlabels=s['labels'], labelfont=dict(size=12, color=s['color']),
                          **_levels(lo_o, hi_o, s['n'])),
            line=dict(width=s['width']), colorscale=[[0, s['color']], [1, s['color']]],
            showscale=False, showlegend=True, hovertemplate=_hover(name)))
    # Отметки - после трасс полей: Patch по ползунку правит data[k] по номеру поля
    fig = go.Figure(data=traces + _map_marks(store, st))

    xax, yax = _map_axes()
    # uirevision меняется вместе с режимом: у Contour и Heatmap разные диапазоны, и масштаб пользователя
    # переносить между ними нельзя. Между полями в одном режиме uirevision не меняется - масштаб сохраняется.
    fig.update_layout(plot_bgcolor='white', uirevision=f"map-{'heatmap' if st['mode'] == 'heatmap' else 'contour'}",
                      xaxis=xax, yaxis=yax, legend=dict(orientation='h', x=0, y=1.01, yanchor='bottom'),
                      margin=dict(t=65, b=50), width=GRAPH_W, height=MAP_H)  # верх карты - вровень с первой кнопкой
    if animate:
        fig.update_layout(sliders=[_animation_slider(store, base, overlays, i, st)])
    return fig


# ============================ Дополнительный график ===========================
# Режим -> (построить фигуру, Patch по слою). Построение получает поля карты `names` (заливка, затем изолинии).
def _full_range(store: SolutionStore, names: list[str], log: bool) -> list[float]:
    """Диапазон оси по всему расчету для полей `names` с запасом; при log - в lg, как ждет plotly."""
    infos = [store.meta['fields'][n] for n in names]
    lo, hi = min(f['zmin'] for f in infos), max(f['zmax'] for f in infos)
    if log:
        pos = min(f['zpos'] for f in infos)
        lo, hi = math.log10(max(lo, pos)), math.log10(max(hi, pos))
    return _pad(lo, hi)


def _field_axes(fig: go.Figure, names: list[str], store: SolutionStore, st: dict) -> None:
    """Оси графика полей карты: заливка слева, изолинии справа. Шкалы - по всему расчету (по всем ячейкам и слоям):
    у профиля autorange подстраивался бы под каждый слой и кривые «прыгали», у ячейки - под ее собственный размах,
    и разные ячейки было бы не сравнить."""
    yax = _axis_dict() | {'range': _full_range(store, names[:1], st['log'])}
    if st['log']:
        yax['type'] = 'log'
    yax2 = dict(side='right', overlaying='y')
    if len(names) > 1:
        yax2['range'] = _full_range(store, names[1:], False)
    fig.update_layout(yaxis=yax, yaxis2=yax2, legend=dict(orientation='h', x=0, y=-0.25), margin=dict(t=40, b=10))


def _hint(fig: go.Figure) -> None:
    fig.add_annotation(text='Щелкните по карте, чтобы выбрать ячейку', showarrow=False, font=dict(size=15))


def _aux_cell(fig, names, i, store, st):
    """Поля карты в выбранной ячейке во времени; пунктир - текущий момент."""
    if st['cell']:
        ci, cj = st['cell']
        for k, nm in enumerate(names):
            fig.add_trace(go.Scatter(x=store.time, y=store.get_cell(nm, ci, cj), mode='lines', name=nm,
                                     yaxis='y2' if k else 'y', hovertemplate='x: %{x} день<br>y: %{y}<br>'))
        fig.add_shape(type='line', xref='x', yref='paper', x0=store.time[i], x1=store.time[i], y0=0, y1=1,
                      line=dict(color='gray', width=1, dash='dot'))
        fig.update_layout(title=dict(text=_cell_label(ci, cj), font=dict(size=14)), xaxis_title='t, день')
    else:
        _hint(fig)
    _field_axes(fig, names, store, st)


def _patch_cell(names, t, store, st) -> Patch:
    patch = Patch()
    patch['layout']['shapes'][0]['x0'] = patch['layout']['shapes'][0]['x1'] = float(store.time[t])
    return patch


def _aux_profile(fig, names, i, store, st):
    """Поля карты вдоль линии профиля на текущем слое."""
    ii, jj, s = _line_cells(st)
    for k, nm in enumerate(names):
        fig.add_trace(go.Scatter(x=s, y=store.get_profile(nm, i, ii, jj), mode='lines', name=nm,
                                 yaxis='y2' if k else 'y', hovertemplate='x: %{x} м<br>y: %{y}<br>'))
    fig.update_layout(xaxis_title='Расстояние вдоль линии, м')
    _field_axes(fig, names, store, st)


def _patch_profile(names, t, store, st) -> Patch:
    patch, (ii, jj, _) = Patch(), _line_cells(st)
    for k, name in enumerate(names):
        patch['data'][k]['y'] = store.get_profile(name, t, ii, jj).tolist()
    return patch


def _aux_fi(fig, names, i, store, st):
    """φ(r) в выбранной ячейке на текущем слое и φ₀; шкала - по всем слоям ячейки, иначе кривая «прыгала» бы."""
    yax = _axis_dict()
    if st['cell']:
        ci, cj = st['cell']
        fi = store.get_fi(st['cell'])
        # φ первой: ее правит Patch по ползунку, φ₀ неподвижна
        for name, y, width in (('φ', fi[i], 4), ('φ₀', fi_0, 3)):
            fig.add_trace(go.Scatter(x=r, y=y, mode='lines', name=name, hovertemplate='x: %{x}<br>y: %{y}<br>',
                                     line=dict(width=width)))
        yax['range'] = _pad(min(0.0, float(fi.min()), float(fi_0.min())), max(float(fi.max()), float(fi_0.max())))
        fig.update_layout(title=dict(text=f'φ(r), {_cell_label(ci, cj)}', font=dict(size=16)),
                          xaxis_title='r, м', yaxis_title='φ')
    else:
        _hint(fig)
    fig.update_layout(yaxis=yax, legend=dict(x=1.0, y=1.0, xanchor='right', font=dict(size=16)),
                      margin=dict(t=40, b=50))


def _patch_fi(names, t, store, st) -> Patch:
    patch = Patch()
    patch['data'][0]['y'] = store.get_fi(st['cell'])[t].tolist()
    return patch


AUX = {'cell': (_aux_cell, _patch_cell), 'fi': (_aux_fi, _patch_fi), **{p: (_aux_profile, _patch_profile) for p in PROFILES}}


def _aux_figure(base: str, i: int, store: SolutionStore, st: dict) -> go.Figure:
    """Дополнительный график по режиму `st['aux']`; размер задает контейнер (`_aux_style`)."""
    fig = go.Figure()
    AUX[st['aux']][0](fig, [base] + _overlays(st, base, store), i, store, st)
    fig.update_layout(plot_bgcolor='white', uirevision=f"aux-{st['aux']}", xaxis=_axis_dict())
    return fig


# ============================ Временные ряды и виды ===========================
def _series_figure(g: str, store: SolutionStore, st: dict) -> go.Figure:
    """Временные ряды группы `g`; логарифмическая ось и нормировка - по настройкам."""
    s = store.meta['series'][g]
    M = store.get_series(g)
    traces = []
    for j, nm in enumerate(s['names']):
        if not s['visible'][j]:
            continue
        y = M[j]
        if st['series_norm']:
            nz = y[y != 0.0]
            y = y / nz[0] if nz.size else y
        unit = '' if st['series_norm'] or not s['units'][j] else ' ' + s['units'][j]
        if st['series_log']:
            y = np.where(y != 0.0, np.abs(y), np.nan)   # ноль на логарифмической оси - разрыв кривой
        traces.append(go.Scatter(x=store.time, y=y, mode='lines', name=nm, hovertemplate=f'x: %{{x}} день<br>y: %{{y}}{unit}<br>',
                                 **({'yaxis': 'y2'} if s['axes'][j] == 'y2' else {})))

    fig = go.Figure(data=traces)
    if not traces:
        fig.add_annotation(text='Нет данных для отображения', showarrow=False, font=dict(size=18))
    y_type = 'log' if st['series_log'] else 'linear'
    fig.update_layout(plot_bgcolor='white', uirevision=f"series:{g}:{y_type}:{st['series_norm']}",
                      xaxis=_axis_dict(), yaxis=_axis_dict() | {'type': y_type},
                      yaxis2=dict(side='right', overlaying='y', type=y_type),
                      legend=dict(x=1.05, y=1.0), width=WIDE_W, height=GRAPH_H)
    return fig


def _view(view_id: str | None) -> dict | None:
    """Вид по id кнопки: 'field:<поле>' - карта поля заливкой, 'series:<группа>' - временные ряды."""
    if view_id is None:
        return None
    kind, name = view_id.split(':', 1)
    is_map = kind == 'field'
    return {'id': view_id, 'kind': 'map' if is_map else 'series', 'base': name if is_map else None,
            'group': None if is_map else name}


def _figure_for(view_id: str, i: int, store: SolutionStore, st: dict, animate: bool = False) -> go.Figure:
    v = _view(view_id)
    return (_map_figure(v['base'], i, store, st, animate) if v['kind'] == 'map'
            else _series_figure(v['group'], store, st))


def _empty_figure(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=20))
    fig.update_layout(width=GRAPH_W, height=GRAPH_H)
    return fig


def _image(fig: go.Figure, fmt: str) -> bytes:
    """Картинка фигуры: SVG - kaleido; EPS - kaleido в PDF, затем `pdftops -eps` (сам kaleido 0.2 pdftops из MiKTeX
    не находит). Только первая страница: у широких фигур kaleido дописывает вторую, пустую, а EPS - одностраничный."""
    if fmt == 'svg':
        return pio.to_image(fig, format='svg')
    with tempfile.TemporaryDirectory() as d:
        pdf, eps = Path(d, 'fig.pdf'), Path(d, 'fig.eps')
        pdf.write_bytes(pio.to_image(fig, format='pdf'))
        subprocess.run([PDFTOPS, '-eps', '-f', '1', '-l', '1', str(pdf), str(eps)], check=True, capture_output=True)
        return eps.read_bytes()


# ============================ Интерфейс ======================================
def _view_entries(meta: dict) -> list[tuple[str, str, str]]:
    """Кнопки по разделам: (раздел, id вида, подпись). Один порядок на столбец кнопок и на список их классов в
    `_select_view` (wildcard-выход сопоставляет значения кнопкам по порядку)."""
    out, names = [], list(meta['fields'])
    for title, fits in FIELD_SECTIONS:
        out += [(title, f'field:{n}', n) for n in names if fits(n)]
        names = [n for n in names if not fits(n)]
    return out + [('Графики', f'series:{g}', SERIES_TITLES[g]) for g in SERIES_GROUPS if g in meta['series']]


def _view_ids(meta: dict) -> list[str]:
    return [vid for _, vid, _ in _view_entries(meta)]


def _btn_class(active: bool) -> str:
    return 'plt-btn active' if active else 'plt-btn'


def _view_buttons(meta: dict, current: str | None) -> list:
    """Вертикальный столбец кнопок в стиле plotly, с заголовками разделов."""
    els, section = [], None
    for title, vid, label in _view_entries(meta):
        if title != section:
            els.append(html.Div(title, className='plt-section'))
            section = title
        els.append(html.Button(label, id={'type': 'view-btn', 'index': vid}, n_clicks=0,
                               className=_btn_class(vid == current)))
    return els


def _export_buttons(graph: str) -> html.Div:
    """Кнопки картинки графика `graph` ('main' / 'aux'): SVG, и EPS, если есть pdftops."""
    return html.Div([html.Button(fmt.upper(), id={'type': 'export', 'graph': graph, 'fmt': fmt}, n_clicks=0,
                                 className='plt-save', title=f'Скачать график в {fmt.upper()}')
                     for fmt in ('svg', 'eps') if fmt == 'svg' or PDFTOPS], className='plt-export')


def _marks(store: SolutionStore) -> dict:
    n = len(store.time)
    if n <= 1:
        return {0: _day_fmt(store.time[0])}
    idx = np.unique(np.linspace(0, n - 1, min(n, MAX_MARKS)).astype(int))
    return {int(i): _day_fmt(store.time[i]) for i in idx}


def _row(label: str, *children) -> html.Div:
    return html.Div([html.Span(label, className='plt-lbl', title=label), *children], className='plt-row')


def _check(cid, label: str, on: bool) -> dcc.Checklist:
    return dcc.Checklist(id=cid, options=[{'label': f' {label}', 'value': 'on'}], value=['on'] if on else [],
                         inline=True)


def _num(cid, value, **kw) -> dcc.Input:
    # debounce: фигура пересобирается по Enter или уходу из поля, а не на каждую цифру
    return dcc.Input(id=cid, type='number', value=value, debounce=True, className='plt-num', **kw)


def _radio(cid, value, options: dict) -> dcc.RadioItems:
    """Переключатель в строку. Вариант переносится на новую строку целиком, а не разрывается («вдоль» / «Y»):
    nowrap здесь и неразрывные пробелы в подписях (labelStyle dash 4 вешает на внутренний span)."""
    return dcc.RadioItems(id=cid, value=value, inline=True, inputStyle={'marginRight': '4px'},
                          labelStyle={'marginRight': '10px', 'whiteSpace': 'nowrap', 'display': 'inline-block'},
                          options=[{'label': lb, 'value': v} for v, lb in options.items()])


def _ov_row(name: str, st: dict) -> html.Div:
    """Строка стиля изолиний одного наложенного поля: цвет, число линий, толщина, подписи."""
    s = _ov_style(st, name)
    return _row(name,
                dcc.Dropdown(id={'type': 'ov-color', 'index': name}, options=list(LINE_COLORS), value=s['color'],
                             clearable=False, style={'width': '110px'}),
                _num({'type': 'ov-n', 'index': name}, s['n'], min=1, max=100, step=1, placeholder='линий'),
                _num({'type': 'ov-width', 'index': name}, s['width'], min=0.5, max=10, step=0.5, placeholder='толщина'),
                _check({'type': 'ov-labels', 'index': name}, 'подписи', s['labels']))


def _map_panel(store: SolutionStore, st: dict) -> html.Details:
    """Настройки карты: изолинии поверх, заливка, шкала, дополнительный график."""
    aux = {'none': 'нет', 'cell': 'ячейка', 'diag': 'диагональ', 'x': 'вдоль X', 'y': 'вдоль Y'}
    if store.meta['has_plots']:
        aux['fi'] = 'φ(r)'
    return html.Details([
        html.Summary('Настройки карты', className='plt-section', style={'cursor': 'pointer'}),
        _row('Изолинии поверх', dcc.Dropdown(id='overlays', options=list(store.meta['fields']), value=st['overlays'],
                                             multi=True, placeholder='поля...', style={'flex': 1})),
        html.Div([_ov_row(n, st) for n in st['overlays']], id='ov-rows'),
        _row('Заливка', _radio('set-mode', st['mode'], {'fill': 'контур', 'lines': 'линии', 'heatmap': 'Heatmap'})),
        _row('Палитра', dcc.Dropdown(id='set-cmap', options=list(PALETTES), value=st['cmap'], clearable=False,
                                     style={'width': '130px'}),
             _check('set-log', 'логарифмическая', st['log'])),
        _row('Изолиний', _num('set-ncont', st['ncont'], min=1, max=100, step=1),
             _check('set-labels', 'подписи', st['labels'])),
        _row('Шкала', _radio('set-range', st['range'], {'all': 'весь расчет', 'layer': 'текущий слой',
                                                        'manual': 'вручную'}),
             _num('set-zmin', st['zmin'], placeholder='min'), _num('set-zmax', st['zmax'], placeholder='max')),
        # Выбор режима сам включает график справа от кнопок; ячейку и линию профиля выбирает щелчок по карте
        _row('Доп. график', _radio('aux-mode', st['aux'], aux)),
    ], open=True, id='map-panel')


def _series_panel(st: dict) -> html.Div:
    return html.Div(_row('Ось Y', _check('series-log', 'логарифмическая', st['series_log']),
                         _check('series-norm', 'нормировать на начальное', st['series_norm'])), id='series-panel')


def _aux_style(view: dict | None, st: dict) -> dict:
    """Дополнительный график - третьей колонкой, вровень с кнопками (paddingTop как у их столбца).

    Ширина - по свободному месту справа от кнопок (`aux-graph` responsive), от GRAPH_W до AUX_MAX_W, с отступом
    от правого края окна; высота - по режиму: φ(r) во всю высоту карты."""
    on = isinstance(view, dict) and view['kind'] == 'map' and st['aux'] != 'none'
    h = AUX_FI_H if st['aux'] == 'fi' else AUX_H
    return {'paddingTop': '40px', 'flex': f'1 1 {GRAPH_W}px', 'minWidth': f'{GRAPH_W}px',
            'maxWidth': f'{AUX_MAX_W}px', 'marginRight': '30px', 'height': f'{h}px',
            **({} if on else {'display': 'none'})}


# Стили и клиентские колбэки - из assets/ (подключает Dash). plotly_script.js там же, но на страницу не идет:
# иначе поверх plotly.js dcc.Graph грузилась бы вторая копия, 4.6 МБ.
app = Dash(__name__, assets_folder=str(ASSETS), assets_ignore=r'plotly_script\.js')
app.title = 'Визуализация'
app.config.suppress_callback_exceptions = True

# Спиннер только на первичной загрузке данных: target_components сужает dcc.Loading до вывода ui-root.children.
# Без него dash>=3 подписывает спиннер на любой вложенный компонент, и он срабатывал бы на каждое отпускание
# ползунка и клик по кнопке, пряча график (мелькание белого фона).
app.layout = html.Div([
    dcc.Interval(id='boot', interval=50, max_intervals=1),
    dcc.Download(id='download'),
    dcc.Loading(html.Div(id='ui-root'), type='circle', target_components={'ui-root': 'children'}),
])


@app.callback(Output('ui-root', 'children'), Input('boot', 'n_intervals'))
def _init(_n):
    """Инициализация: страница открывается сразу, тяжелые данные грузятся под спиннером."""
    try:
        store = get_store()
    except Exception:
        return html.Div([html.H3('Ошибка загрузки данных'), html.Pre(traceback.format_exc())])

    ids = _view_ids(store.meta)
    view = _view(ids[0] if ids else None)
    kind = view['kind'] if view else 'series'
    n_times = store.meta['n_times']
    slider_off = view is None or kind == 'series' or n_times <= 1
    st = dict(DEFAULT_SETTINGS)
    graph_config = {'displaylogo': False, 'toImageButtonOptions': {'format': 'svg'}}   # «камера» - в SVG

    return html.Div([
        html.Div([
            # ------- левая колонка: график, подпись времени, ползунок, настройки -------
            html.Div([
                html.Div(dcc.Graph(id='main-graph', responsive=False, config=graph_config,
                                   figure=_figure_for(view['id'], 0, store, st) if view else _empty_figure('Нет данных')),
                         id='graph-container'),
                html.Div([   # правее левого края фигуры, ближе к области графика
                    html.Div(f'Время: {_day_fmt(store.time[0])}' if kind != 'series' else '', id='time-label',
                             className='plt-time-label'),
                    html.Div([
                        # во всю колонку до кнопки ▶: ползунок заканчивается у правого края фигуры
                        html.Div(dcc.Slider(id='time-slider', min=0, max=max(0, n_times - 1), step=1, value=0,
                                            marks=_marks(store), updatemode='mouseup', allow_direct_input=False,
                                            disabled=slider_off),
                                 className='plt-slider', style={'flex': 1, 'minWidth': 0}),
                        html.Button('▶', id='btn-play', n_clicks=0, disabled=slider_off, className='plt-btn',
                                    style={'width': '28px', 'padding': '5px 0', 'textAlign': 'center', 'flexShrink': 0,
                                           'marginLeft': 'auto', 'transform': 'translateX(30px)'}),
                    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'width': '100%'}),
                    dcc.Interval(id='play-interval', interval=300, disabled=True),
                    _map_panel(store, st),
                    _series_panel(st),
                ], style={'paddingLeft': f'{CTRL_SHIFT}px'}),
            ], id='graph-column'),

            # ------- столбец кнопок видов (с детальным составом их за сорок - прокрутка в высоту графика) -------
            html.Div(_view_buttons(store.meta, view and view['id']) + [
                html.Button('Сохранить в HTML', id='btn-save', n_clicks=0, className='plt-save'),
                _export_buttons('main'),
            ], style={'width': '185px', 'display': 'flex', 'flexDirection': 'column', 'gap': '2px',
                      'paddingTop': '40px', 'flexShrink': 0, 'maxHeight': f'{GRAPH_H}px', 'overflowY': 'auto'}),

            # ------- справа от кнопок: доп. график; responsive - размер ему задает контейнер (`_aux_style`) -------
            html.Div([_export_buttons('aux'),
                      dcc.Graph(id='aux-graph', figure=go.Figure(), responsive=True, style={'height': '100%'},
                                config=graph_config)],
                     id='aux-container', style=_aux_style(view, st)),
        ], id='ui-body', className=f'view-{kind}',
            style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '30px', 'alignItems': 'flex-start'}),

        dcc.Store(id='view-state', data=view),
        dcc.Store(id='map-settings', data=st),
        dcc.Store(id='times-store', data=store.meta['time']),
        dcc.Store(id='t-live'),                    # слой, который показать: {'t', 'seq'} (`viz.gate`)
        dcc.Store(id='live-busy', data=False),     # запрос слоя в работе (running у `_on_slider`)
        dcc.Download(id='download-img'),
    ], style={'padding': '0 10px'})


app.clientside_callback(
    ClientsideFunction('viz', 'timeLabel'),
    Output('time-label', 'children'),
    Input('time-slider', 'drag_value'), Input('time-slider', 'value'), Input('view-state', 'data'),
    State('times-store', 'data'),
    prevent_initial_call=True,
)
app.clientside_callback(
    ClientsideFunction('viz', 'gate'),
    Output('t-live', 'data'),
    Input('time-slider', 'drag_value'), Input('time-slider', 'value'), Input('live-busy', 'data'),
    prevent_initial_call=True,
)


@app.callback(
    Output('view-state', 'data'),
    Output('time-slider', 'disabled'),
    Output('btn-play', 'disabled'),
    Output('play-interval', 'disabled', allow_duplicate=True),
    Output({'type': 'view-btn', 'index': ALL}, 'className'),
    Output('ui-body', 'className'),
    Input({'type': 'view-btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def _select_view(clicks):
    """Кнопка вида. Фигуру строит `_redraw` по новому view-state, раскладку переключает класс `view-<kind>`.

    Wildcard-выход (классы кнопок) всегда получает список длиной с число кнопок: одиночный no_update для него
    недопустим (InvalidCallbackReturnValue)."""
    ids = _view_ids(get_store().meta)
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or not any(clicks or []):   # холостое срабатывание при создании кнопок
        return (no_update,) * 4 + ([no_update] * len(ids), no_update)
    view = _view(trig['index'])
    slider_off = view['kind'] == 'series' or get_store().meta['n_times'] <= 1
    # вид без ползунка глушит и автопрокрутку, иначе play-interval тикал бы в фоне вхолостую
    return (view, slider_off, slider_off, True if slider_off else no_update,
            [_btn_class(vid == view['id']) for vid in ids], f"view-{view['kind']}")


@app.callback(
    Output('ov-rows', 'children'),
    Input('overlays', 'value'),
    State('map-settings', 'data'),
    prevent_initial_call=True,
)
def _ov_rows(overlays, st):
    """Строки стиля изолиний; у оставшихся в списке полей стиль сохраняется (берется из map-settings)."""
    st = st | {'overlays': overlays or []}
    return [_ov_row(n, st) for n in st['overlays']]


@app.callback(
    Output('map-settings', 'data'),
    inputs=dict(**{k: Input(cid, 'value') for k, cid in SETTING_INPUTS.items()},
                click=Input('main-graph', 'clickData'),
                **{f'ov_{p}': Input({'type': f'ov-{p}', 'index': ALL}, 'value') for p in ('color', 'n', 'width', 'labels')}),
    state=dict(ov_ids=State({'type': 'ov-color', 'index': ALL}, 'id'), prev=State('map-settings', 'data')),
    prevent_initial_call=True,
)
def _collect(click, ov_color, ov_n, ov_width, ov_labels, ov_ids, prev, **s):
    """Все элементы управления -> один словарь настроек (map-settings); `s` - значения по `SETTING_INPUTS`."""
    cell = prev['cell']
    if ctx.triggered_id == 'main-graph':
        # Щелчок по карте выбирает ячейку графика ячейки и φ(r), а у профиля вдоль X / Y - ее строку / столбец
        if s['aux'] not in ('cell', 'fi', 'x', 'y') or not click:
            return no_update
        p = click['points'][0]
        cell = [int(np.abs(x_mesh - p['x']).argmin()), int(np.abs(y_mesh - p['y']).argmin())]
    ov = dict(prev['ov'])
    for cid, c, n, w, lb in zip(ov_ids, ov_color, ov_n, ov_width, ov_labels):
        ov[cid['index']] = {'color': c, 'n': int(n) if n else None, 'width': w, 'labels': bool(lb)}
    return s | {k: bool(s[k]) for k in _FLAGS} | {
        'overlays': s['overlays'] or [], 'ncont': int(min(max(s['ncont'] or DEFAULT_SETTINGS['ncont'], 1), 100)),
        'ov': ov, 'cell': cell}


@app.callback(
    Output('main-graph', 'figure', allow_duplicate=True),
    Output('aux-graph', 'figure', allow_duplicate=True),
    Output('aux-container', 'style'),
    Input('view-state', 'data'),
    Input('map-settings', 'data'),
    State('time-slider', 'value'),
    prevent_initial_call=True,
)
def _redraw(view, st, t):
    """Смена вида или настроек - фигура заново (карта ~75x75 собирается за миллисекунды)."""
    if not isinstance(view, dict):
        return _empty_figure('Нет данных для отображения'), no_update, _aux_style(None, st)
    store, t = get_store(), int(t or 0)
    aux_on = view['kind'] == 'map' and st['aux'] != 'none'
    aux = _aux_figure(view['base'], t, store, st) if aux_on else no_update
    return _figure_for(view['id'], t, store, st), aux, _aux_style(view, st)


@app.callback(
    Output('main-graph', 'figure', allow_duplicate=True),
    Output('aux-graph', 'figure', allow_duplicate=True),
    Input('t-live', 'data'),
    State('view-state', 'data'),
    State('map-settings', 'data'),
    running=[(Output('live-busy', 'data'), True, False)],
    prevent_initial_call=True,
)
def _on_slider(live, view, st):
    """Слой по положению ползунка (перетаскивание, отпускание, автопрокрутка) из заслонки: Patch только с данными слоя."""
    if not isinstance(view, dict) or view['kind'] == 'series' or not live:
        return no_update, no_update
    store, t = get_store(), int(live['t'])
    base = view['base']
    overlays = _overlays(st, base, store)
    patch = Patch()
    for k, d in enumerate(_trace_data(store, base, overlays, t, st)):
        for key, v in d.items():
            if key == 'contours':   # поштучно: целиком contours затер бы coloring и подписи
                for kk, vv in v.items():
                    patch['data'][k]['contours'][kk] = vv
            else:
                patch['data'][k][key] = v.tolist() if key == 'z' else v

    aux = no_update
    if st['aux'] in AUX and (st['cell'] or st['aux'] not in NEEDS_CELL):
        aux = AUX[st['aux']][1]([base] + overlays, t, store, st)
    return patch, aux


@app.callback(
    Output('play-interval', 'disabled', allow_duplicate=True),
    Output('btn-play', 'children'),
    Output('time-slider', 'value'),
    Input('btn-play', 'n_clicks'),
    Input('play-interval', 'n_intervals'),
    State('play-interval', 'disabled'),
    State('time-slider', 'value'),
    State('time-slider', 'max'),
    State('time-slider', 'disabled'),
    prevent_initial_call=True,
)
def _play(_n_clicks, _n_intervals, is_paused, value, max_value, slider_off):
    """Кнопка ▶/⏸ — просто крутит time-slider.value через dcc.Interval; сам сдвиг
    подхватывает заслонка ползунка и передает в _on_slider через t-live."""
    if ctx.triggered_id == 'btn-play':
        if slider_off:
            return no_update, no_update, no_update
        return (not is_paused), ('▶' if is_paused else '⏸'), no_update

    value = int(value or 0)
    if value >= max_value:  # доехали до конца — стоп на последнем кадре
        return True, '▶', no_update
    return no_update, no_update, value + 1


@app.callback(
    Output('download', 'data'),
    Input('btn-save', 'n_clicks'),
    State('view-state', 'data'),
    State('time-slider', 'value'),
    State('map-settings', 'data'),
    prevent_initial_call=True,
)
def _save_html(n, view, t, st):
    """Сохранение в HTML: полная анимация карты с текущими настройками; временные ряды - статичный график."""
    if not n or not isinstance(view, dict):
        return no_update
    fig = _figure_for(view['id'], int(t or 0), get_store(), st, animate=view['kind'] == 'map')
    html_str = fig.to_html(include_plotlyjs=str(js_path), full_html=True)
    return dcc.send_bytes(html_str.encode('utf-8'), filename=f"{_sanitize(view['base'] or view['group'])}.html")


@app.callback(
    Output('download-img', 'data'),
    Input({'type': 'export', 'graph': ALL, 'fmt': ALL}, 'n_clicks'),
    State('view-state', 'data'),
    State('time-slider', 'value'),
    State('map-settings', 'data'),
    prevent_initial_call=True,
)
def _export(_clicks, view, t, st):
    """Картинка (SVG / EPS) основного или доп. графика на текущем слое - та же фигура, что на экране, без
    масштаба, выбранного мышью (его сохраняет «камера» в панели графика)."""
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or not ctx.triggered[0]['value'] or not isinstance(view, dict):
        return no_update
    store, t = get_store(), int(t or 0)
    name = view['base'] or view['group']
    if trig['graph'] == 'aux':
        if view['kind'] != 'map' or st['aux'] == 'none':
            return no_update
        fig = _aux_figure(view['base'], t, store, st)
        fig.update_layout(width=AUX_EXPORT_W, height=AUX_FI_H if st['aux'] == 'fi' else AUX_H)
        name += f"_{st['aux']}"
    else:
        fig = _figure_for(view['id'], t, store, st)
    if view['kind'] == 'map':
        name += f'_{store.time[t]:.1f}d'
    return dcc.send_bytes(_image(fig, trig['fmt']), filename=f"{_sanitize(name)}.{trig['fmt']}")
