"""Интерактивная визуализация больших расчётных данных (Dash) с ленивой загрузкой.

Интерфейс повторяет исходный plotly-файл: график слева, вертикальный столбец
кнопок выбора поля справа, под графиком подпись "Время: ... день" и ползунок.

Отсутствие мерцания и минимум запросов:
1. НЕТ автообновления (никаких dcc.Interval, polling и dcc.Loading вокруг графика).
2. Слайдер в режиме mouseup: при перетаскивании — 0 запросов к серверу,
   при отпускании — ровно 1 запрос, и в нём только массив z/y текущего поля (Patch).
3. Подпись времени обновляется на клиенте (clientside callback) — без сервера.
4. Все контейнеры фиксированного размера — интерфейс не "вздрагивает".
5. Сетка в пределах расчётной области: явные range осей + constrain='domain'.
6. Wildcard-выход (кнопки) всегда получает СПИСОК значений — иначе Dash падает
   с InvalidCallbackReturnValue (что ломало обновление по слайдеру).

Карта - одно поле заливкой (кнопка справа) и любые поля изолиниями поверх (список под ползунком);
«Sat and Temp» - готовый набор из них. Настройки отображения - в `map-settings`, от них пересобирается
фигура (`_redraw`); ползунок шлет только данные слоя (`_on_slider`). Справа от кнопок - дополнительный график:
величины в выбранной щелчком ячейке во времени или профиль вдоль линии.

Данные: кэш pkl -> npy (float32, memory-mapped). Сохранение в HTML — полная анимация.
"""
from __future__ import annotations

import gc
import json
import math
import os
import re
import shutil
import threading
import traceback
import webbrowser
from collections import OrderedDict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import dash
from dash import Dash, Input, Output, State, dcc, html, no_update, ctx, Patch, ALL

assert hasattr(dash, 'Patch'), 'Требуется dash>=2.9: pip install -U dash'

from paraphin.geometry import r, fi_0
from paraphin.constants import Nx, Ny, X_min, X_max, hx, hy, Y_max, Y_min, bar_to_pa, day_to_sec, CONTOUR_PLOT, S_max
from paraphin.utils.read_data_files import (read_solution_data, convert_pkl_files, results_path, layers_file,
                                           case_name)

js_path = Path(__file__).parent / 'plotly_script.js'

# ----------------------------- Настройки ------------------------------------
CACHE_VERSION = 8
CACHE_DIR = results_path.parent / f'{case_name}_vizcache'
DTYPE = np.float32
MAX_DISPLAY_CELLS = 250_000             # максимум точек сетки для экрана (None — без прореживания)
MAX_MARKS = 7                           # подписей под слайдером, как в оригинале
JSON_DECIMALS = 5
GRAPH_W, GRAPH_H = 720, 800             # размер фигуры карты (фиксированный, квадратная область)
# Высота карты: квадратная область уже GRAPH_H, и под ней оставалась пустая полоса до ползунка
MAP_H = 660
CTRL_SHIFT = 50                         # сдвиг подписи времени, ползунка и настроек вправо от края фигуры, px
WIDE_W = 950                            # 1D-графики (временные ряды, φ(r)) шире карты — слайдер уже них не растягивается
SKIP_FIELDS = {}
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

# Юникод, а не LaTeX: dcc.Graph без mathjax=True показывает формулу сырым текстом
names_converter = {'fi': 'φ', 'fi_o': 'φ₀'}

PALETTES = ('Jet', 'Turbo', 'Viridis', 'Plasma', 'RdBu', 'Greys')
LINE_COLORS = ('black', 'white', 'red', 'magenta', 'cyan', 'lime', 'blue', 'green', 'orange', 'gray')
OV_COLORS = LINE_COLORS[:6]             # цвета изолиний по умолчанию - по очереди
AUX_H = 330                             # высота дополнительного графика под картой
PROFILES = ('diag', 'x', 'y')           # значения `aux` для профиля: диагональ, вдоль X при j, вдоль Y при i

# Настройки отображения (`map-settings`); `ov` - стиль изолиний по имени поля, `cell` - (i, j) выбранной ячейки
DEFAULT_SETTINGS = {
    'overlays': [], 'mode': 'fill' if CONTOUR_PLOT else 'heatmap', 'cmap': 'Jet', 'ncont': 10, 'labels': True,
    'range': 'all', 'zmin': None, 'zmax': None, 'log': False, 'ov': {},
    'aux': 'none', 'cell': None, 'line_idx': 0,
    'series_log': False, 'series_norm': False,
}


def _hover(name: str) -> str:
    return f'X: %{{x}}<br>Y: %{{y}}<br>{name}: %{{z}}<extra></extra>'

_FONT = "'Open Sans', Verdana, Arial, sans-serif"
_BTN_SAVE = {'backgroundColor': '#2196F3', 'color': 'white', 'fontWeight': 'bold',
             'border': '1px solid #1d84c9', 'padding': '7px 11px', 'borderRadius': '3px',
             'cursor': 'pointer', 'width': '100%', 'marginTop': '10px', 'fontSize': '13px',
             'fontFamily': _FONT, 'textAlign': 'center'}
_BTN_ACTIVE = {'backgroundColor': '#e5ecf6'}
_BTN_INACTIVE = {'backgroundColor': '#ffffff'}

# CSS подключается через html.Style — гарантированно применяется (dcc.Markdown вырезает <style>).
_CSS = f"""
.plt-btn{{width:100%;text-align:left;padding:5px 11px;background:#fff;border:1px solid #e2e6ec;
          border-radius:3px;color:#2a3f5f;font-family:{_FONT};font-size:13px;font-weight:400;
          cursor:pointer;box-shadow:none;}}
.plt-btn:hover{{background:#f3f6fb;border-color:#d5dbe5;}}
.plt-btn:focus,.plt-btn:focus-visible{{outline:none;box-shadow:none;}}
.plt-btn:active{{background:#eef2f9;}}
.plt-section{{font-family:{_FONT};font-size:12px;font-weight:bold;color:#7f8c9d;padding:8px 2px 2px;}}
.plt-row{{display:flex;align-items:center;gap:8px;margin:4px 0;font-family:{_FONT};font-size:13px;color:#2a3f5f;}}
.plt-lbl{{width:120px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}}
.plt-num{{width:64px;}}

.plt-time-label{{font-family:{_FONT};font-size:13px;font-weight:bold;color:#2a3f5f;
                 height:22px;padding-left:2px;font-variant-numeric:tabular-nums;white-space:nowrap;}}

/* Слайдер в стиле plotly: тонкая серая рельса, светлый круглый бегунок, без фиолетового.
   dash>=3 рендерит dcc.Slider поверх @radix-ui/react-slider (классы dash-slider-*,
   не rc-slider-*) и не поставляет для него никакого CSS — Radix-примитивы
   безликие ("headless"), всю визуальную часть (цвет/размер/форму) задаёт
   потребитель; позиционирование (left/transform по значению) Radix считает сам
   через инлайн-стили, поэтому здесь только цвет/размер/форма, без position. */
.plt-slider{{min-height:52px;}}
.plt-slider .dash-slider-track{{background-color:#d8d8d8 !important;height:3px;border-radius:2px;}}
.plt-slider .dash-slider-range{{background-color:#9a9a9a !important;border-radius:2px;}}
.plt-slider .dash-slider-thumb{{width:14px;height:14px;border-radius:50%;
                               border:1px solid #9a9a9a !important;background-color:#f6f6f6 !important;
                               box-shadow:none !important;cursor:pointer;}}
.plt-slider .dash-slider-thumb:hover,
.plt-slider .dash-slider-thumb:focus,
.plt-slider .dash-slider-thumb:active,
.plt-slider .dash-slider-thumb:focus-visible{{border-color:#8a8a8a !important;background-color:#efefef !important;
                                        box-shadow:none !important;outline:none !important;}}
.plt-slider .dash-slider-dot{{display:none;}}                  /* точки меток — как в plotly, их нет */
.plt-slider .dash-slider-mark{{font-size:11px;color:#2a3f5f;font-family:{_FONT};}}
/* скрыть всплывающий счётчик шагов у бегунка при наведении/перетаскивании */
.plt-slider .dash-slider-tooltip{{display:none !important;}}
"""


def _day_fmt(t) -> str:
    """Формат времени как в оригинале: f'{round(time[i], 1)} день'."""
    return f'{round(float(t), 1)} день'


# ============================ Слой данных ====================================
def _load_raw():
    """Чтение исходных данных (тяжёлая операция — только при сборе кэша)."""
    try:
        # Файл слоев остается на диске, только если расчет не дошел до склейки: значит данные
        # свежее обработанного файла и их надо собрать заново
        if layers_file.is_file():
            convert_pkl_files()
        _n_times, data = read_solution_data(f'{case_name}_processed_data.pkl')
    except ValueError:
        convert_pkl_files()
        _n_times, data = read_solution_data(f'{case_name}_processed_data.pkl')
    print('Временных слоев:', _n_times)
    return _n_times, data


def _safe_minmax(a: np.ndarray) -> tuple[float, float]:
    with np.errstate(invalid='ignore'):
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return (lo, hi) if (math.isfinite(lo) and math.isfinite(hi)) else (0.0, 1.0)


def _sanitize(name: str) -> str:
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in name)


class SolutionStore:
    """Метаданные — в RAM, массивы — memory-mapped npy; срезы читаются по требованию."""

    def __init__(self):
        self._lock = threading.RLock()
        self._mmaps: dict[str, np.ndarray] = {}
        self._series_cache: dict[str, np.ndarray] = {}
        self._plots_cache: OrderedDict | None = None

        # Создание кеша кривых
        _n, raw = _load_raw()
        self._build_cache(raw)
        del raw
        gc.collect()

        self.meta = json.loads((CACHE_DIR / 'meta.json').read_text(encoding='utf-8'))
        self.time = np.asarray(self.meta['time'], dtype=np.float64)
        self.x = np.asarray(self.meta['x_display'])
        self.y = np.asarray(self.meta['y_display'])
        self.stride = self.meta['stride']

    # ------------------------------- кэш ------------------------------------
    def _source_stat(self):
        name = f'{case_name}_processed_data.pkl'
        for base in (results_path.parent, Path.cwd()):
            p = base / name
            if p.is_file():
                st = p.stat()
                return {'size': st.st_size, 'mtime': st.st_mtime}
        return None


    def _build_cache(self, raw: dict):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        meta = {'version': CACHE_VERSION, 'source': self._source_stat()}

        time = np.asarray(raw.pop('Time'), dtype=np.float64) / day_to_sec
        n_times = int(len(time))
        meta['n_times'] = n_times
        meta['time'] = [round(float(t), 6) for t in time]

        # Детальный состав нефти и гель (группы парафина, асфальтены, `gelation`): поля - картами, массы ('Totals') - рядами
        for group in ('Composition', 'Gel'):
            if group in raw:
                raw.update(raw.pop(group))
        # Скин-фактор - безразмерный и бывает отрицательным: не к дебитам, а в прочие ряды
        wells = raw.get('Wells') or {}
        for k in [k for k in wells if k.endswith('_skin')]:
            raw.setdefault('Other params', {})[k] = wells.pop(k)

        if 'Pressure' in raw:
            raw['Pressure'] = np.asarray(raw['Pressure']) / bar_to_pa
        if 'plots' in raw and 'Wps' in raw and 'Saturation' in raw:
            raw['Wps'] = np.asarray(raw['Wps']) * (S_max - np.asarray(raw['Saturation']))

        # --- 3D-поля: потоковая запись на диск, полный куб в RAM не дублируется ---
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
                            # нижняя граница логарифмической шкалы
                            'zpos': float(pos.min()) if pos.size else 1e-12}
            print(f'  поле "{name}": {arr.shape} -> {fname}')
            del arr
        meta['fields'] = fields

        # --- сетка и прореживание для экрана ---
        ny, nx = (next(iter(fields.values()))['shape'][1:] if fields else (Ny, Nx))
        s = 1 if MAX_DISPLAY_CELLS is None else max(1, math.ceil(math.sqrt(nx * ny / MAX_DISPLAY_CELLS)))
        meta['stride'] = s
        meta['x_display'] = list(np.asarray(x_mesh[::s], dtype=float))
        meta['y_display'] = list(np.asarray(y_mesh[::s], dtype=float))

        # --- временные ряды ---
        series = {}
        for g in SERIES_GROUPS:
            d = raw.get(g)
            if not d:
                continue
            names, axes, units, vis, rows = [], [], [], [], []
            for k, v in d.items():
                v = np.asarray(v, dtype=np.float64).reshape(-1)
                if v.size == 1:
                    v = np.full(n_times, float(v[0]))
                if v.size != n_times:
                    print(f'  {g}/{k}: длина {v.size} != {n_times}, пропущено')
                    continue
                if g == 'Wells':
                    visible = not np.allclose(v, v.flat[0], rtol=0, atol=1e-9)
                    if 'eta' in k:
                        y, ax, unit = v, 'y2', ''
                    elif 'bhp' in k:
                        y, ax, unit = v / bar_to_pa, 'y', 'бар'  # `save_fields` пишет забойное давление в Па
                    else:
                        y, ax, unit = np.abs(v) * day_to_sec, 'y', 'м^3/день'
                elif g == 'Wells_accumulated':
                    visible, y, ax, unit = not np.all(np.isclose(v, 0.0)), np.abs(v), 'y', 'м^3'
                elif g == 'Totals':
                    # Массы в нефти на порядки больше отложений: отложения и адсорбция - на правой оси
                    visible, y, ax, unit = bool(np.any(v != 0.0)), v, 'y' if 'in oil' in k else 'y2', 'кг'
                else:
                    visible, y, ax, unit = True, v, 'y', ''
                names.append(k); axes.append(ax); units.append(unit); vis.append(visible)
                rows.append(y.astype(DTYPE))
            if names:
                fname = f'series__{_sanitize(g)}.npy'
                np.save(CACHE_DIR / fname, np.asarray(rows, dtype=DTYPE))
                series[g] = {'file': fname, 'names': names, 'axes': axes, 'units': units, 'visible': vis}
                print(f'  {g}: {len(names)} кривых')
        meta['series'] = series

        # --- графики phi(r) ---
        plots = {}
        for k, v in (raw.get('plots') or {}).items():
            v = np.asarray(v, dtype=DTYPE)
            fname = f'plots__{_sanitize(k)}.npy'
            np.save(CACHE_DIR / fname, v)
            plots[k] = {'file': fname, 'shape': list(v.shape)}
        meta['plots'] = plots
        meta['has_plots'] = bool(plots)

        (CACHE_DIR / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')

    # ---------------------------- доступ к данным ----------------------------
    def _mmap(self, fname: str) -> np.ndarray:
        with self._lock:
            mm = self._mmaps.get(fname)
            if mm is None:
                mm = np.load(CACHE_DIR / fname, mmap_mode='r')   # с диска читаются только нужные страницы
                self._mmaps[fname] = mm
            return mm

    @staticmethod
    def _round(a: np.ndarray) -> np.ndarray:
        return a if JSON_DECIMALS is None else np.round(a.astype(np.float64, copy=False), JSON_DECIMALS)

    def get_field(self, name: str, i: int) -> np.ndarray:
        """Срез поля в момент i (с прореживанием, без округления), z[j, i]."""
        mm = self._mmap(self.meta['fields'][name]['file'])
        s = self.stride
        return np.asarray(mm[i, ::s, ::s], dtype=np.float64)

    def get_cell(self, name: str, i: int, j: int) -> np.ndarray:
        """Величина в ячейке (i, j) во все моменты: из memmap читается только этот столбец."""
        return np.asarray(self._mmap(self.meta['fields'][name]['file'])[:, j, i], dtype=np.float64)

    def get_profile(self, name: str, t: int, ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
        """Значения в ячейках (ii[k], jj[k]) в момент t."""
        return np.asarray(self._mmap(self.meta['fields'][name]['file'])[t][jj, ii], dtype=np.float64)

    def get_series(self, group: str) -> np.ndarray:
        with self._lock:
            M = self._series_cache.get(group)
            if M is None:
                M = np.asarray(np.load(CACHE_DIR / self.meta['series'][group]['file']), dtype=np.float64)
                self._series_cache[group] = M
            return M

    def get_plot_data(self) -> 'OrderedDict[str, np.ndarray]':
        with self._lock:
            if self._plots_cache is None:
                data = OrderedDict()
                for k, info in self.meta['plots'].items():
                    data[k] = np.asarray(np.load(CACHE_DIR / info['file']), dtype=np.float64)
                if 'fi_o' not in data and 'fi' in data:
                    data['fi_o'] = np.repeat(fi_0[np.newaxis, :], data['fi'].shape[0], axis=0)
                self._plots_cache = data
            return self._plots_cache


_STORE: SolutionStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> SolutionStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = SolutionStore()
    return _STORE


# ============================ Построение фигур ===============================
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
    xax = _axis_dict()
    xax.update(autorange=True, constrain='domain')
    yax = _axis_dict()
    # Квадратная область уже высоты фигуры: constrain='domain' по умолчанию ставит ее посередине, ниже кнопок
    yax.update(autorange=True, scaleanchor='x', scaleratio=1, constrain='domain', constraintoward='top')
    return xax, yax


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
    """Ячейки профиля (ii, jj) и расстояние вдоль линии, м."""
    if st['aux'] == 'x':
        jj = int(min(max(st['line_idx'] or 0, 0), Ny - 1))
        ii = np.arange(Nx)
        return ii, np.full(Nx, jj), x_mesh - x_mesh[0]
    if st['aux'] == 'y':
        ii = int(min(max(st['line_idx'] or 0, 0), Nx - 1))
        jj = np.arange(Ny)
        return np.full(Ny, ii), jj, y_mesh - y_mesh[0]
    f = np.linspace(0.0, 1.0, max(Nx, Ny))   # диагональ (0, 0) -> (Nx-1, Ny-1)
    ii, jj = np.rint(f * (Nx - 1)).astype(int), np.rint(f * (Ny - 1)).astype(int)
    return ii, jj, np.hypot(x_mesh[ii] - x_mesh[0], y_mesh[jj] - y_mesh[0])


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

    fig = go.Figure(data=traces)
    # Отметка ячейки и линия профиля идут после трасс полей: Patch по ползунку правит data[k] по номеру поля
    if st['aux'] == 'cell' and st['cell']:
        ci, cj = st['cell']
        fig.add_trace(go.Scatter(x=[x_mesh[ci]], y=[y_mesh[cj]], mode='markers', showlegend=False, hoverinfo='skip',
                                 marker=dict(symbol='x', size=13, color='black', line=dict(width=1, color='white'))))
    elif st['aux'] in PROFILES:
        ii, jj, _ = _line_cells(st)
        fig.add_shape(type='line', x0=x_mesh[ii[0]], y0=y_mesh[jj[0]], x1=x_mesh[ii[-1]], y1=y_mesh[jj[-1]],
                      line=dict(color='black', width=2, dash='dash'))

    xax, yax = _map_axes()
    # uirevision меняется вместе с режимом: у Contour и Heatmap разные диапазоны, и масштаб пользователя
    # переносить между ними нельзя. Между полями в одном режиме uirevision не меняется - масштаб сохраняется.
    fig.update_layout(plot_bgcolor='white', uirevision=f"map-{'heatmap' if st['mode'] == 'heatmap' else 'contour'}",
                      xaxis=xax, yaxis=yax, legend=dict(orientation='h', x=0, y=1.01, yanchor='bottom'),
                      margin=dict(t=65, b=50), width=GRAPH_W, height=MAP_H)  # верх карты - вровень с первой кнопкой

    if animate:   # полная анимация для сохранения в HTML
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
        fig.update_layout(sliders=[dict(active=min(i, len(steps) - 1),
                                        currentvalue={'prefix': 'Время: '}, steps=steps)])
    return fig


def _aux_figure(base: str, i: int, store: SolutionStore, st: dict) -> go.Figure:
    """Дополнительный график под картой: поля карты в выбранной ячейке во времени или профиль вдоль линии."""
    names = [base] + _overlays(st, base, store)
    fig = go.Figure()
    if st['aux'] == 'cell':
        if not st['cell']:
            fig.add_annotation(text='Щелкните по карте, чтобы выбрать ячейку', showarrow=False, font=dict(size=15))
        else:
            ci, cj = st['cell']
            for k, nm in enumerate(names):
                fig.add_trace(go.Scatter(x=store.time, y=store.get_cell(nm, ci, cj), mode='lines', name=nm,
                                         yaxis='y2' if k else 'y', hovertemplate='x: %{x} день<br>y: %{y}<br>'))
            fig.add_shape(type='line', xref='x', yref='paper', x0=store.time[i], x1=store.time[i], y0=0, y1=1,
                          line=dict(color='gray', width=1, dash='dot'))
            fig.update_layout(title=dict(text=f'Ячейка ({ci}, {cj})', font=dict(size=14)), xaxis_title='t, день')
    else:
        ii, jj, s = _line_cells(st)
        for k, nm in enumerate(names):
            fig.add_trace(go.Scatter(x=s, y=store.get_profile(nm, i, ii, jj), mode='lines', name=nm,
                                     yaxis='y2' if k else 'y', hovertemplate='x: %{x} м<br>y: %{y}<br>'))
        fig.update_layout(xaxis_title='Расстояние вдоль линии, м')
    yax = _axis_dict()
    if st['log']:
        yax['type'] = 'log'
    fig.update_layout(plot_bgcolor='white', uirevision=f"aux-{st['aux']}", xaxis=_axis_dict(), yaxis=yax,
                      yaxis2=dict(side='right', overlaying='y'), legend=dict(orientation='h', x=0, y=-0.25),
                      margin=dict(t=40, b=10), width=GRAPH_W, height=AUX_H)
    return fig


def _series_figure(view: dict, store: SolutionStore, st: dict) -> go.Figure:
    """Скважины / прочие параметры — оформление как в оригинале; логарифмическая ось и нормировка - по настройкам."""
    g = view['group']
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
            ht = 'x: %{x} день<br>y: %{y}<br>'
        elif g == 'Wells':
            ht = 'x: %{x} день<br>y: %{y}<br>' if s['axes'][j] == 'y2' else 'x: %{x} день<br>y: %{y} м^3/день<br>'
        elif g == 'Wells_accumulated':
            ht = 'x: %{x} день<br>y: %{y} м^3<br>'
        elif g == 'Totals':
            ht = 'x: %{x} день<br>y: %{y} кг<br>'
        else:
            ht = 'x: %{x}<br>y: %{y}<br>'
        if st['series_log']:
            y = np.where(y != 0.0, np.abs(y), np.nan)   # ноль на логарифмической оси - разрыв кривой
        kw = {'yaxis': 'y2'} if s['axes'][j] == 'y2' else {}
        traces.append(go.Scatter(x=store.time, y=y, mode='lines', name=nm, hovertemplate=ht, **kw))

    fig = go.Figure(data=traces)
    if not traces:
        fig.add_annotation(text='Нет данных для отображения', showarrow=False, font=dict(size=18))
    y_type = 'log' if st['series_log'] else 'linear'
    fig.update_layout(plot_bgcolor='white', uirevision=f"series:{g}:{y_type}:{st['series_norm']}",
                      xaxis=_axis_dict(), yaxis=_axis_dict() | {'type': y_type},
                      yaxis2=dict(side='right', overlaying='y', type=y_type),
                      legend=dict(x=1.05, y=1.0),
                      width=WIDE_W, height=GRAPH_H)
    return fig


def _fi_figure(view: dict, i: int, store: SolutionStore, animate: bool = False) -> go.Figure:
    """Графики phi(r) — оформление как в _visualize_plots_fi оригинала."""
    data = store.get_plot_data()
    traces = []
    for nm, vals in data.items():
        if nm in ('Ur', 'Ub'):
            traces.append(go.Scatter(x=r, y=vals[i], mode='lines', name=nm, yaxis='y2',
                                     hovertemplate='x: %{x}<br>y: %{y}<br>', line=dict(width=3)))
        else:
            traces.append(go.Scatter(x=r, y=vals[i], mode='lines', name=names_converter.get(nm, nm),
                                     hovertemplate='x: %{x}<br>y: %{y}<br>', line=dict(width=4)))

    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis_title='r, м', yaxis_title=names_converter['fi'],
        plot_bgcolor='white', uirevision='fi', margin=dict(t=0, b=0),
        xaxis=_axis_dict(), yaxis=_axis_dict(),
        yaxis2=dict(side='right', overlaying='y'),
        legend=dict(x=1.01, y=0.8, font=dict(size=18)),
        width=WIDE_W, height=600,
    )
    fig.add_shape(type='rect', xref='paper', yref='paper',
                  x0=0, y0=0, x1=1, y1=1, line=dict(color='black', width=1))
    fig.add_hline(y=0, line=dict(color='black', width=1))
    fig.update_traces(marker=dict(size=8, line=dict(width=1)))

    if animate:
        steps = [dict(method='update', args=[{'y': [v[j] for v in data.values()]}],
                      label=_day_fmt(store.time[j]))
                 for j in range(store.meta['n_times'])]
        fig.update_layout(sliders=[dict(active=min(i, len(steps) - 1),
                                        currentvalue={'prefix': 'Время: '}, steps=steps)])
    return fig


def _view_def(view_id: str, store: SolutionStore) -> dict:
    """Вид по id кнопки. У карты `base` - поле заливкой; «Sat and Temp» - насыщенность с изолиниями температуры."""
    if view_id == 'sattemp':
        return {'kind': 'map', 'id': view_id, 'base': 'Saturation'}
    if view_id.startswith('field:'):
        return {'kind': 'map', 'id': view_id, 'base': view_id.split(':', 1)[1]}
    if view_id.startswith('series:'):
        return {'kind': 'series', 'id': view_id, 'group': view_id.split(':', 1)[1]}
    if view_id == 'fi':
        return {'kind': 'fi', 'id': view_id, 'sources': list(store.get_plot_data().keys())}
    raise ValueError(view_id)


def _figure_for(view_id: str, i: int, store: SolutionStore, st: dict, animate: bool = False) -> go.Figure:
    v = _view_def(view_id, store)
    if v['kind'] == 'map':
        return _map_figure(v['base'], i, store, st, animate)
    if v['kind'] == 'fi':
        return _fi_figure(v, i, store, animate)
    return _series_figure(v, store, st)


# ============================ Интерфейс ======================================
def _view_entries(meta: dict) -> list[tuple[str, str, str]]:
    """Кнопки по разделам: (раздел, id вида, подпись). Один порядок на столбец кнопок и на список их стилей в
    `_select_view` (wildcard-выход сопоставляет стили кнопкам по порядку)."""
    out = []
    names = [n for n in meta['fields'] if n not in SKIP_FIELDS]
    for title, fits in FIELD_SECTIONS:
        out += [(title, f'field:{n}', n) for n in names if fits(n)]
        names = [n for n in names if not fits(n)]
    out += [('Графики', f'series:{g}', SERIES_TITLES[g]) for g in SERIES_GROUPS if g in meta['series']]
    if {'Saturation', 'Temperature'} <= set(meta['fields']):
        out.append(('Графики', 'sattemp', 'Sat and Temp'))
    if meta.get('has_plots'):
        out.append(('Графики', 'fi', 'Графики φ(r)'))
    return out


def _valid_view_ids(meta: dict) -> list[str]:
    return [vid for _, vid, _ in _view_entries(meta)]


def _view_buttons(meta: dict, current: str | None) -> list:
    """Вертикальный столбец кнопок в стиле plotly, с заголовками разделов."""
    els, section = [], None
    for title, vid, label in _view_entries(meta):
        if title != section:
            els.append(html.Div(title, className='plt-section'))
            section = title
        els.append(html.Button(label, id={'type': 'view-btn', 'index': vid}, n_clicks=0, className='plt-btn',
                               style=_BTN_ACTIVE if vid == current else _BTN_INACTIVE))
    return els


def _resolve_view(view_id: str | None, store: SolutionStore) -> str | None:
    valid = _valid_view_ids(store.meta)
    return view_id if view_id in valid else (valid[0] if valid else None)


def _payload(view_id: str | None, store: SolutionStore) -> dict | None:
    if view_id is None:
        return None
    v = _view_def(view_id, store)
    return {'id': v['id'], 'kind': v['kind'], 'base': v.get('base'), 'sources': v.get('sources', []),
            'group': v.get('group')}


def _marks(store: SolutionStore) -> dict:
    n = len(store.time)
    if n <= 1:
        return {0: _day_fmt(store.time[0])}
    idx = np.unique(np.linspace(0, n - 1, min(n, MAX_MARKS)).astype(int))
    return {int(i): _day_fmt(store.time[i]) for i in idx}


def _empty_figure(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=20))
    fig.update_layout(width=GRAPH_W, height=GRAPH_H)
    return fig


def _graph_width(kind: str | None) -> int:
    """Карта — квадратная (GRAPH_W), временные ряды и φ(r) — шире (WIDE_W), слайдер под ними уже."""
    return WIDE_W if kind in ('series', 'fi') else GRAPH_W


def _container_style(kind: str | None) -> dict:
    h = MAP_H if kind == 'map' else GRAPH_H
    return {'height': f'{h + 10}px', 'width': f'{_graph_width(kind) + 10}px', 'overflow': 'hidden'}


def _column_style(kind: str | None) -> dict:
    """Левая колонка (график+слайдер) — та же ширина, что и graph-container: иначе при
    переключении на более широкий вид (fi/series) колонка остаётся старой ширины и кнопки
    справа не сдвигаются, а виджет графика просто наезжает на них поверх."""
    return {'width': f'{_graph_width(kind) + 10}px', 'flexShrink': 0}


def _shown(on: bool) -> dict:
    return {} if on else {'display': 'none'}


def _row(label: str, *children) -> html.Div:
    return html.Div([html.Span(label, className='plt-lbl', title=label), *children], className='plt-row')


def _check(cid, label: str, on: bool) -> dcc.Checklist:
    return dcc.Checklist(id=cid, options=[{'label': f' {label}', 'value': 'on'}], value=['on'] if on else [],
                         inline=True)


def _num(cid, value, **kw) -> dcc.Input:
    # debounce: фигура пересобирается по Enter или уходу из поля, а не на каждую цифру
    return dcc.Input(id=cid, type='number', value=value, debounce=True, className='plt-num', **kw)


def _ov_row(name: str, st: dict) -> html.Div:
    """Строка стиля изолиний одного наложенного поля: цвет, число линий, толщина, подписи."""
    s = _ov_style(st, name)
    return _row(name,
                dcc.Dropdown(id={'type': 'ov-color', 'index': name}, options=list(LINE_COLORS), value=s['color'],
                             clearable=False, style={'width': '110px'}),
                _num({'type': 'ov-n', 'index': name}, s['n'], min=1, max=100, step=1, placeholder='линий'),
                _num({'type': 'ov-width', 'index': name}, s['width'], min=0.5, max=10, step=0.5, placeholder='толщина'),
                _check({'type': 'ov-labels', 'index': name}, 'подписи', s['labels']))


def _map_panel(store: SolutionStore, st: dict, kind: str | None) -> html.Details:
    """Настройки карты: изолинии поверх, заливка, шкала, дополнительный график."""
    # Вариант переключателя переносится на новую строку целиком, а не разрывается («вдоль» / «Y»): nowrap здесь
    # и неразрывные пробелы \u00a0 в подписях (labelStyle dash 4 вешает на внутренний span, его могут перебить стили)
    radio = dict(inline=True, inputStyle={'marginRight': '4px'},
                 labelStyle={'marginRight': '10px', 'whiteSpace': 'nowrap', 'display': 'inline-block'})
    return html.Details([
        html.Summary('Настройки карты', className='plt-section', style={'cursor': 'pointer'}),
        _row('Изолинии поверх', dcc.Dropdown(id='overlays', options=list(store.meta['fields']), value=st['overlays'],
                                             multi=True, placeholder='поля...', style={'flex': 1})),
        html.Div([_ov_row(n, st) for n in st['overlays']], id='ov-rows'),
        _row('Заливка', dcc.RadioItems(id='set-mode', value=st['mode'], **radio, options=[
            {'label': 'контур', 'value': 'fill'}, {'label': 'линии', 'value': 'lines'},
            {'label': 'Heatmap', 'value': 'heatmap'}])),
        _row('Палитра', dcc.Dropdown(id='set-cmap', options=list(PALETTES), value=st['cmap'], clearable=False,
                                     style={'width': '130px'}),
             _check('set-log', 'логарифмическая', st['log'])),
        _row('Изолиний', _num('set-ncont', st['ncont'], min=1, max=100, step=1),
             _check('set-labels', 'подписи', st['labels'])),
        _row('Шкала', dcc.RadioItems(id='set-range', value=st['range'], **radio, options=[
            {'label': 'весь расчет', 'value': 'all'}, {'label': 'текущий слой', 'value': 'layer'},
            {'label': 'вручную', 'value': 'manual'}]),
             _num('set-zmin', st['zmin'], placeholder='min'), _num('set-zmax', st['zmax'], placeholder='max')),
        # Один переключатель: выбор линии профиля сам включает график (справа от кнопок). Подписи короткие -
        # все варианты помещаются в одну строку; что они значат - в строке ниже
        _row('Доп. график', dcc.RadioItems(id='aux-mode', value=st['aux'], **radio, options=[
            {'label': 'нет', 'value': 'none'}, {'label': 'ячейка', 'value': 'cell'},
            {'label': 'диагональ', 'value': 'diag'}, {'label': 'вдоль\u00a0X', 'value': 'x'},
            {'label': 'вдоль\u00a0Y', 'value': 'y'}])),
        _row('Номер j / i', _num('line-idx', st['line_idx'], min=0, step=1),
             html.Span('ячейка - щелчок по карте; профиль вдоль X - строка j, вдоль Y - столбец i')),
    ], open=True, id='map-panel', style=_shown(kind == 'map'))


def _series_panel(st: dict, kind: str | None) -> html.Div:
    return html.Div(_row('Ось Y', _check('series-log', 'логарифмическая', st['series_log']),
                         _check('series-norm', 'нормировать на начальное', st['series_norm'])),
                    id='series-panel', style=_shown(kind == 'series'))


def _aux_style(view: dict | None, st: dict) -> dict:
    """Дополнительный график - третьей колонкой, вровень с кнопками (paddingTop как у их столбца)."""
    on = isinstance(view, dict) and view['kind'] == 'map' and st['aux'] != 'none'
    return {'paddingTop': '40px', 'flexShrink': 0} | _shown(on)


app = Dash(__name__)
app.title = f'Визуализация'
app.config.suppress_callback_exceptions = True

# Стили встраиваем в <head> страницы через index_string — работает во всех версиях Dash
# (html.Style есть только в новых версиях, а dcc.Markdown вырезает <style>).
app.index_string = f'''<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>{_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>'''

# Минимальный статический layout: спиннер только на первичной загрузке данных.
# target_components сужает dcc.Loading ровно до вывода ui-root.children (загрузка
# при старте): без него dash>=3 подписывает спиннер на ЛЮБОЙ вложенный компонент,
# и он срабатывает на каждое отпускание слайдера/клик по кнопке (колбэки шага и вида
# тоже пишут в потомков ui-root), пряча график и слайдер (visibility:hidden) —
# отсюда мелькание белого фона страницы.
app.layout = html.Div([
    dcc.Interval(id='boot', interval=50, max_intervals=1),
    dcc.Download(id='download'),
    dcc.Loading(html.Div(id='ui-root'), type='circle',
                target_components={'ui-root': 'children'}),
])


@app.callback(Output('ui-root', 'children'), Input('boot', 'n_intervals'))
def _init(_n):
    """Инициализация: страница открывается сразу, тяжелые данные грузятся под спиннером."""
    try:
        store = get_store()
    except Exception:
        return html.Div([html.H3('Ошибка загрузки данных'), html.Pre(traceback.format_exc())])

    view_id = _resolve_view(None, store)
    n_times = store.meta['n_times']
    view = _payload(view_id, store)
    kind = view['kind'] if view else 'series'
    slider_disabled = view_id is None or kind == 'series' or n_times <= 1
    st = dict(DEFAULT_SETTINGS)

    return html.Div([
        html.Div([
            # ------- левая колонка: график, подпись времени, ползунок -------
            html.Div([
                html.Div(
                    dcc.Graph(id='main-graph',
                              figure=_figure_for(view_id, 0, store, st) if view_id else _empty_figure('Нет данных'),
                              responsive=False,
                              config={'displaylogo': False}),
                    id='graph-container', style=_container_style(kind)),
                # Подпись, ползунок и настройки - правее левого края фигуры, ближе к области графика
                html.Div([
                html.Div(id='time-label',
                         children=f'Время: {_day_fmt(store.time[0])}' if kind != 'series' else '',
                         className='plt-time-label'),
                html.Div(
                    [html.Div(dcc.Slider(id='time-slider', min=0, max=max(0, n_times - 1), step=1, value=0,
                                         marks=_marks(store), updatemode='mouseup', allow_direct_input=False,
                                         disabled=slider_disabled),
                              className='plt-slider', style={'width': f'{GRAPH_W}px'}),
                     html.Button( '▶', id='btn-play', n_clicks=0, disabled=slider_disabled, className='plt-btn',
                                  style={'width': '28px', 'padding': '5px 0', 'textAlign': 'center', 'flexShrink': 0,
                                         'marginLeft': 'auto', 'transform': 'translateX(30px)'}),
                     ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'width': '100%'}
                ),
                dcc.Interval(id='play-interval', interval=300, disabled=True),
                _map_panel(store, st, kind),
                _series_panel(st, kind),
                ], style={'paddingLeft': f'{CTRL_SHIFT}px'}),
            ], id='graph-column', style=_column_style(kind)),

            # ------- правая колонка: вертикальный столбец кнопок -------
            html.Div(
                _view_buttons(store.meta, view_id) + [
                    html.Button('Сохранить в HTML', id='btn-save', n_clicks=0, style=_BTN_SAVE),
                ],
                # С детальным составом кнопок за сорок: столбец прокручивается в пределах высоты графика
                style={'width': '185px', 'display': 'flex', 'flexDirection': 'column',
                       'gap': '2px', 'paddingTop': '40px', 'flexShrink': 0,
                       'maxHeight': f'{GRAPH_H}px', 'overflowY': 'auto'}),

            # ------- справа от кнопок, сверху: ячейка во времени или профиль -------
            html.Div(dcc.Graph(id='aux-graph', figure=go.Figure(), responsive=False,
                               config={'displaylogo': False}),
                     id='aux-container', style=_aux_style(view, st)),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '30px', 'alignItems': 'flex-start'}),

        dcc.Store(id='view-state', data=view),
        dcc.Store(id='map-settings', data=st),
        dcc.Store(id='times-store', data=store.meta['time']),
    ], style={'padding': '0 10px'})


# --- клиентский колбэк: живая подпись времени при перетаскивании (БЕЗ запросов к серверу) ---
app.clientside_callback(
    """function(dv, v, view, times) {
        if (!view || view.kind === 'series') return '';
        const idx = (dv !== null && dv !== undefined) ? dv : ((v !== null && v !== undefined) ? v : 0);
        const t = (times && times[idx] !== undefined) ? times[idx] : 0;
        return 'Время: ' + Number(t).toFixed(1) + ' день';
    }""",
    Output('time-label', 'children'),
    Input('time-slider', 'drag_value'),
    Input('time-slider', 'value'),
    Input('view-state', 'data'),
    State('times-store', 'data'),
    prevent_initial_call=True,
)


@app.callback(
    Output('view-state', 'data'),
    Output('time-slider', 'disabled'),
    Output('btn-play', 'disabled'),
    Output('play-interval', 'disabled', allow_duplicate=True),
    Output('graph-container', 'style'),
    Output('graph-column', 'style'),
    Output({'type': 'view-btn', 'index': ALL}, 'style'),
    Output('overlays', 'value'),
    Output('map-panel', 'style'),
    Output('series-panel', 'style'),
    Input({'type': 'view-btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def _select_view(clicks):
    """Кнопка вида. Фигуру строит `_redraw` по новому view-state.

    ВАЖНО: wildcard-выход (стили кнопок) должен ВСЕГДА получать список длиной
    с число кнопок — одиночный no_update для него недопустим (InvalidCallbackReturnValue).
    """
    store = get_store()
    valid = _valid_view_ids(store.meta)
    trig_id = ctx.triggered_id
    # защита от холостого срабатывания при создании кнопок (n_clicks == 0 / не кнопка)
    if (not (isinstance(trig_id, dict) and trig_id.get('type') == 'view-btn')
            or all((c or 0) <= 0 for c in (clicks or []))):
        return (no_update,) * 6 + ([no_update] * len(valid), no_update, no_update, no_update)

    view_id = _resolve_view(trig_id['index'], store)
    view = _payload(view_id, store)
    kind = view['kind'] if view else None
    slider_off = kind in (None, 'series') or store.meta['n_times'] <= 1
    # переключение на вид без ползунка глушит и автопрокрутку, иначе play-interval
    # продолжал бы тикать в фоне вхолостую; между картой и φ(r) автовоспроизведение не трогаем
    stop_play = True if slider_off else no_update
    overlays = ['Temperature'] if view_id == 'sattemp' else no_update   # готовый набор «Sat and Temp»
    return (view, slider_off, slider_off, stop_play, _container_style(kind), _column_style(kind),
            [_BTN_ACTIVE if vid == view_id else _BTN_INACTIVE for vid in valid],
            overlays, _shown(kind == 'map'), _shown(kind == 'series'))


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
    Input('overlays', 'value'),
    Input('set-mode', 'value'),
    Input('set-cmap', 'value'),
    Input('set-log', 'value'),
    Input('set-ncont', 'value'),
    Input('set-labels', 'value'),
    Input('set-range', 'value'),
    Input('set-zmin', 'value'),
    Input('set-zmax', 'value'),
    Input({'type': 'ov-color', 'index': ALL}, 'value'),
    Input({'type': 'ov-n', 'index': ALL}, 'value'),
    Input({'type': 'ov-width', 'index': ALL}, 'value'),
    Input({'type': 'ov-labels', 'index': ALL}, 'value'),
    Input('aux-mode', 'value'),
    Input('line-idx', 'value'),
    Input('main-graph', 'clickData'),
    Input('series-log', 'value'),
    Input('series-norm', 'value'),
    State({'type': 'ov-color', 'index': ALL}, 'id'),
    State('map-settings', 'data'),
    prevent_initial_call=True,
)
def _collect(overlays, mode, cmap, log, ncont, labels, rng, zmin, zmax, ov_c, ov_n, ov_w, ov_l,
             aux, line_idx, click, s_log, s_norm, ov_ids, prev):
    """Все элементы управления -> один словарь настроек (map-settings)."""
    cell = prev['cell']
    if ctx.triggered_id == 'main-graph':
        if aux != 'cell' or not click:   # щелчок по карте нужен только графику ячейки
            return no_update
        p = click['points'][0]
        cell = [int(np.abs(x_mesh - p['x']).argmin()), int(np.abs(y_mesh - p['y']).argmin())]
    ov = dict(prev['ov'])
    for cid, c, n, w, lb in zip(ov_ids, ov_c, ov_n, ov_w, ov_l):
        ov[cid['index']] = {'color': c, 'n': int(n) if n else None, 'width': w, 'labels': bool(lb)}
    return {
        'overlays': overlays or [], 'mode': mode, 'cmap': cmap, 'log': bool(log),
        'ncont': int(min(max(ncont or DEFAULT_SETTINGS['ncont'], 1), 100)), 'labels': bool(labels),
        'range': rng, 'zmin': zmin, 'zmax': zmax, 'ov': ov,
        'aux': aux, 'cell': cell, 'line_idx': int(line_idx or 0),
        'series_log': bool(s_log), 'series_norm': bool(s_norm),
    }


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
        return _empty_figure('Нет данных для отображения'), no_update, _shown(False)
    store, t = get_store(), int(t or 0)
    fig = _figure_for(view['id'], t, store, st)
    aux_on = view['kind'] == 'map' and st['aux'] != 'none'
    aux = _aux_figure(view['base'], t, store, st) if aux_on else no_update
    return fig, aux, _aux_style(view, st)


@app.callback(
    Output('main-graph', 'figure', allow_duplicate=True),
    Output('aux-graph', 'figure', allow_duplicate=True),
    Input('time-slider', 'value'),
    State('view-state', 'data'),
    State('map-settings', 'data'),
    prevent_initial_call=True,
)
def _on_slider(t, view, st):
    """Отпускание ползунка (в т.ч. автопрокрутка): единственный запрос, Patch только с данными слоя."""
    if not isinstance(view, dict) or view['kind'] == 'series':
        return no_update, no_update
    store, t = get_store(), int(t or 0)
    patch = Patch()
    if view['kind'] == 'fi':
        data = store.get_plot_data()
        for k, name in enumerate(view['sources']):
            patch['data'][k]['y'] = data[name][t].tolist()
        return patch, no_update

    base = view['base']
    overlays = _overlays(st, base, store)
    for k, d in enumerate(_trace_data(store, base, overlays, t, st)):
        for key, v in d.items():
            if key == 'contours':   # поштучно: целиком contours затер бы coloring и подписи
                for kk, vv in v.items():
                    patch['data'][k]['contours'][kk] = vv
            else:
                patch['data'][k][key] = v.tolist() if key == 'z' else v

    aux = no_update
    if st['aux'] in PROFILES:
        aux = Patch()
        ii, jj, _ = _line_cells(st)
        for k, name in enumerate([base] + overlays):
            aux['data'][k]['y'] = store.get_profile(name, t, ii, jj).tolist()
    elif st['aux'] == 'cell' and st['cell']:
        aux = Patch()
        aux['layout']['shapes'][0]['x0'] = aux['layout']['shapes'][0]['x1'] = float(store.time[t])
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
    подхватывает существующий колбэк _on_slider (у него уже есть Input на time-slider.value)."""
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
def _save_html(n, payload, t, st):
    """Сохранение в HTML: всегда полная анимация текущего вида с текущими настройками
    (Results.html / fi_func.html; для временных рядов — статичный график)."""
    if not n or not isinstance(payload, dict):
        return no_update
    store = get_store()
    t = int(t or 0)
    animate = payload['kind'] in ('map', 'fi')
    fig = _figure_for(payload['id'], t, store, st, animate=animate)

    if payload['kind'] == 'fi':
        fname = 'fi_func.html'
    elif payload['kind'] == 'map':
        fname = f'{fig.data[0].name}.html'
    else:
        fname = f'{_sanitize(payload["group"])}.html'

    html_str = fig.to_html(include_plotlyjs=str(js_path), full_html=True)
    return dcc.send_bytes(html_str.encode('utf-8'), filename=fname)


def visualize_solution(port: int = 8050, debug: bool = False, open_browser: bool = True):
    """Запуск интерактивной визуализации; страница открывается в браузере, когда сервер уже слушает порт.

    Чтобы открыть с другого устройства, запустите с host='0.0.0.0', и на другом устройстве откройте http://<IP>:<порт>.
    Разрешите порт в брандмауэре. 0.0.0.0 открывает доступ всем в локальной сети.
    """
    # С debug перезагрузчик запускает модуль второй раз (WERKZEUG_RUN_MAIN) - вкладка открывается один раз
    if open_browser and os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        threading.Timer(1.0, webbrowser.open, args=(f'http://127.0.0.1:{port}/',)).start()
    try:
        runner = getattr(app, 'run', None) or app.run_server
        runner(host='127.0.0.1', port=port, debug=debug)
    finally:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)



if __name__ == '__main__':
    visualize_solution()
    """
    попробовать сделать динамическое отображение трехмерных данных fi 
    (на двумерной карте выбирать ячейку, при нажатии на которую открывается одномерный график со слайдером)
    """
