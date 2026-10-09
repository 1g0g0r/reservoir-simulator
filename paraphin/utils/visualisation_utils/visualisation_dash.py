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

Данные: кэш pkl -> npy (float32, memory-mapped). Сохранение в HTML — полная анимация.
"""
from __future__ import annotations

import gc
import json
import math
import re
import shutil
import threading
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import dash
from dash import Dash, Input, Output, State, dcc, html, no_update, ctx, Patch, ALL

assert hasattr(dash, 'Patch'), 'Требуется dash>=2.9: pip install -U dash'

from paraphin.geometry import r, fi_0
from paraphin.constants import (Nx, Ny, X_min, X_max, hx, hy, Y_max, Y_min, results_path, layers_file, js_path,
                                bar_to_pa, day_to_sec, CONTOUR_PLOT, S_max, case_name)
from paraphin.utils.read_data_files import read_solution_data, convert_pkl_files

# ----------------------------- Настройки ------------------------------------
CACHE_VERSION = 7
CACHE_DIR = results_path.parent / f'{case_name}_vizcache'
DTYPE = np.float32
MAX_DISPLAY_CELLS = 250_000             # максимум точек сетки для экрана (None — без прореживания)
MAX_MARKS = 7                           # подписей под слайдером, как в оригинале
JSON_DECIMALS = 5
GRAPH_W, GRAPH_H = 720, 800             # размер фигуры карты (фиксированный, квадратная область)
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

names_converter = {'fi': '$$\\varphi$$', 'fi_o': '$$\\varphi_0$$'}
_HOVER_MAP = 'X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>'

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
            mm = np.lib.format.open_memmap(CACHE_DIR / fname, mode='w+', dtype=DTYPE, shape=arr.shape)
            for i in range(arr.shape[0]):
                mm[i] = arr[i]
            mm.flush()
            del mm
            fields[name] = {'file': fname, 'shape': list(arr.shape), **dict(zip(('zmin', 'zmax'), _safe_minmax(arr)))}
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
        """Срез поля в момент i (с прореживанием и округлением)."""
        mm = self._mmap(self.meta['fields'][name]['file'])
        s = self.stride
        return self._round(np.ascontiguousarray(mm[i, ::s, ::s]))

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
    yax.update(autorange=True, scaleanchor='x', scaleratio=1, constrain='domain')
    return xax, yax


def _map_figure(view: dict, i: int, store: SolutionStore, animate: bool = False) -> go.Figure:
    """Поле данных — вид как в оригинале (Jet, подписи контуров), квадратная область."""
    if view.get('sattemp'):
        traces = [
            go.Contour(x=store.x, y=store.y, z=store.get_field('Saturation', i), colorscale='Jet',
                       name='Saturation', contours=dict(coloring='fill', showlabels=True),
                       hovertemplate=_HOVER_MAP),
            go.Contour(x=store.x, y=store.y, z=store.get_field('Temperature', i), name='Temperature',
                       contours=dict(coloring='lines', showlabels=True,
                                     start=25 * 1.001, end=70 * 0.99, size=10),
                       line=dict(width=3), colorscale=[[0, 'black'], [1, 'black']],
                       showscale=False, showlegend=True, hovertemplate=_HOVER_MAP),
        ]
    else:
        name = view['sources'][0]
        info = store.meta['fields'][name]
        common = dict(x=store.x, y=store.y, z=store.get_field(name, i), name=name, colorscale='Jet',
                      zmin=info['zmin'], zmax=info['zmax'], hovertemplate=_HOVER_MAP)
        if CONTOUR_PLOT:
            traces = [go.Contour(**common, contours=dict(coloring='fill', showlabels=True,
                                                         labelfont=dict(size=12, color='black')))]
        else:
            traces = [go.Heatmap(**common)]

    fig = go.Figure(data=traces)
    xax, yax = _map_axes()
    # uirevision меняется вместе с режимом: у Contour и Heatmap разные диапазоны, и масштаб пользователя
    # переносить между ними нельзя. Между полями (Pressure/Saturation/...) в одном режиме uirevision
    # не меняется, так что масштаб/зум сохраняется.
    uirevision = 'map-sattemp' if view.get('sattemp') else f"map-{'contour' if CONTOUR_PLOT else 'heatmap'}"
    fig.update_layout(plot_bgcolor='white', uirevision=uirevision,
                      xaxis=xax, yaxis=yax, legend=dict(x=1.05, y=1.0),
                      width=GRAPH_W, height=GRAPH_H)

    if animate:   # полная анимация для сохранения в HTML
        steps = [dict(method='update',
                      args=[{'z': [store.get_field(nm, j) for nm in view['sources']]}],
                      label=_day_fmt(store.time[j]))
                 for j in range(store.meta['n_times'])]
        fig.update_layout(sliders=[dict(active=min(i, len(steps) - 1),
                                        currentvalue={'prefix': 'Время: '}, steps=steps)])
    return fig


def _series_figure(view: dict, store: SolutionStore) -> go.Figure:
    """Скважины / прочие параметры — оформление как в оригинале."""
    g = view['group']
    s = store.meta['series'][g]
    M = store.get_series(g)
    traces = []
    for j, nm in enumerate(s['names']):
        if not s['visible'][j]:
            continue
        if g == 'Wells':
            ht = 'x: %{x} день<br>y: %{y}<br>' if s['axes'][j] == 'y2' else 'x: %{x} день<br>y: %{y} м^3/день<br>'
        elif g == 'Wells_accumulated':
            ht = 'x: %{x} день<br>y: %{y} м^3<br>'
        elif g == 'Totals':
            ht = 'x: %{x} день<br>y: %{y} кг<br>'
        else:
            ht = 'x: %{x}<br>y: %{y}<br>'
        kw = {'yaxis': 'y2'} if s['axes'][j] == 'y2' else {}
        traces.append(go.Scatter(x=store.time, y=M[j], mode='lines', name=nm, hovertemplate=ht, **kw))

    fig = go.Figure(data=traces)
    if not traces:
        fig.add_annotation(text='Нет данных для отображения', showarrow=False, font=dict(size=18))
    fig.update_layout(plot_bgcolor='white', uirevision=f'series:{g}',
                      xaxis=_axis_dict(), yaxis=_axis_dict(),
                      yaxis2=dict(side='right', overlaying='y'),
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
    if view_id == 'sattemp':
        return {'kind': 'map', 'id': view_id, 'sources': ['Saturation', 'Temperature'], 'sattemp': True}
    if view_id.startswith('field:'):
        return {'kind': 'map', 'id': view_id, 'sources': [view_id.split(':', 1)[1]], 'sattemp': False}
    if view_id.startswith('series:'):
        return {'kind': 'series', 'id': view_id, 'group': view_id.split(':', 1)[1]}
    if view_id == 'fi':
        return {'kind': 'fi', 'id': view_id, 'sources': list(store.get_plot_data().keys())}
    raise ValueError(view_id)


def _figure_for(view_id: str, i: int, store: SolutionStore, animate: bool = False) -> go.Figure:
    v = _view_def(view_id, store)
    if v['kind'] == 'map':
        return _map_figure(v, i, store, animate)
    if v['kind'] == 'fi':
        return _fi_figure(v, i, store, animate)
    return _series_figure(v, store)


# ============================ Интерфейс ======================================
def _view_entries(meta: dict) -> list[tuple[str, str, str]]:
    """Кнопки по разделам: (раздел, id вида, подпись). Один порядок на столбец кнопок и на список их стилей в
    `_update` (wildcard-выход сопоставляет стили кнопкам по порядку)."""
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
    return {'id': v['id'], 'kind': v['kind'], 'sources': v.get('sources', []), 'group': v.get('group')}


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
    return {'height': f'{GRAPH_H + 10}px', 'width': f'{_graph_width(kind) + 10}px', 'overflow': 'hidden'}


def _column_style(kind: str | None) -> dict:
    """Левая колонка (график+слайдер) — та же ширина, что и graph-container: иначе при
    переключении на более широкий вид (fi/series) колонка остаётся старой ширины и кнопки
    справа не сдвигаются, а виджет графика просто наезжает на них поверх."""
    return {'width': f'{_graph_width(kind) + 10}px', 'flexShrink': 0}


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
# и он срабатывает на каждое отпускание слайдера/клик по кнопке (callback _update
# тоже пишет в потомков ui-root), пряча график и слайдер (visibility:hidden) —
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
    kind = _view_def(view_id, store)['kind'] if view_id else 'series'
    slider_disabled = view_id is None or kind == 'series' or n_times <= 1

    return html.Div([
        html.Div([
            # ------- левая колонка: график, подпись времени, ползунок -------
            html.Div([
                html.Div(
                    dcc.Graph(id='main-graph',
                              figure=_figure_for(view_id, 0, store) if view_id else _empty_figure('Нет данных'),
                              responsive=False,
                              config={'displaylogo': False}),
                    id='graph-container', style=_container_style(kind)),
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
                        html.Button(f"Contour: {'вкл' if CONTOUR_PLOT else 'выкл'}", id='btn-contour', n_clicks=0,
                                    className='plt-btn', style={'width': 'auto', 'whiteSpace': 'nowrap', 'flexShrink': 0,
                                                                'padding': '5px 11px', 'transform': 'translateX(40px)'}),
                     ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'width': '100%'}
                ),
                dcc.Interval(id='play-interval', interval=300, disabled=True),
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
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '30px', 'alignItems': 'flex-start'}),

        dcc.Store(id='view-state', data=_payload(view_id, store)),
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
    Output('main-graph', 'figure', allow_duplicate=True),
    Output('view-state', 'data'),
    Output('time-slider', 'disabled'),
    Output('btn-play', 'disabled'),
    Output('play-interval', 'disabled', allow_duplicate=True),
    Output('graph-container', 'style'),
    Output('graph-column', 'style'),
    Output({'type': 'view-btn', 'index': ALL}, 'style'),
    Input({'type': 'view-btn', 'index': ALL}, 'n_clicks'),
    Input('time-slider', 'value'),
    State('view-state', 'data'),
    prevent_initial_call=True,
)
def _update(clicks, t, prev):
    """Единый колбэк: отпускание ползунка -> Patch (только данные), кнопка -> лёгкая фигура.

    ВАЖНО: wildcard-выход (стили кнопок) должен ВСЕГДА получать список длиной
    с число кнопок — одиночный no_update для него недопустим (InvalidCallbackReturnValue).
    """
    store = get_store()
    t = int(t or 0)
    n_btn = len(_valid_view_ids(store.meta))
    noop_btns = [no_update] * n_btn

    trig_props = {p['prop_id'] for p in ctx.triggered}
    slider_fired = 'time-slider.value' in trig_props
    btn_fired = any(p.startswith('{') for p in trig_props)

    # --- отпускание ползунка (в т.ч. автопрокрутка): единственный запрос, обновляем только z/y ---
    if slider_fired and not btn_fired:
        if not isinstance(prev, dict) or prev['kind'] == 'series':
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, noop_btns
        patch = Patch()
        if prev['kind'] == 'map':
            for k, name in enumerate(prev['sources']):
                patch['data'][k]['z'] = store.get_field(name, t).tolist()
        else:  # fi
            data = store.get_plot_data()
            for k, name in enumerate(prev['sources']):
                patch['data'][k]['y'] = data[name][t].tolist()
        return patch, no_update, no_update, no_update, no_update, no_update, no_update, noop_btns

    # --- кнопка вида ---
    if btn_fired:
        trig_id = ctx.triggered_id
        # защита от холостого срабатывания при создании кнопок (n_clicks == 0 / не кнопка)
        if not (isinstance(trig_id, dict) and trig_id.get('type') == 'view-btn'):
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, noop_btns
        if all((c or 0) <= 0 for c in (clicks or [])):
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, noop_btns

        valid = _valid_view_ids(store.meta)
        view_id = _resolve_view(trig_id['index'], store)
        if view_id is None:
            return (_empty_figure('Нет данных для отображения'), None, True, True, True,
                    _container_style(None), _column_style(None), [_BTN_INACTIVE] * len(valid))

        v = _view_def(view_id, store)
        slider_off = v['kind'] == 'series' or store.meta['n_times'] <= 1
        # переключение на вид без ползунка глушит и автопрокрутку, иначе play-interval
        # продолжал бы тикать в фоне вхолостую; между картой и φ(r) автовоспроизведение не трогаем
        stop_play = True if slider_off else no_update
        return (_figure_for(view_id, t, store),
                _payload(view_id, store),
                slider_off, slider_off, stop_play,
                _container_style(v['kind']), _column_style(v['kind']),
                [_BTN_ACTIVE if vid == view_id else _BTN_INACTIVE for vid in valid])

    return no_update, no_update, no_update, no_update, no_update, no_update, no_update, noop_btns


@app.callback(
    Output('main-graph', 'figure', allow_duplicate=True),
    Output('btn-contour', 'children'),
    Input('btn-contour', 'n_clicks'),
    State('view-state', 'data'),
    State('time-slider', 'value'),
    prevent_initial_call=True,
)
def _toggle_contour(_n, payload, t):
    """Contour (заливка с изолиниями) <-> Heatmap для карт — тот же CONTOUR_PLOT,
    что и в constants.py, просто переключаемый в рантайме без перезапуска."""
    global CONTOUR_PLOT
    CONTOUR_PLOT = not CONTOUR_PLOT
    label = f"Contour: {'вкл' if CONTOUR_PLOT else 'выкл'}"
    if not isinstance(payload, dict) or payload['kind'] != 'map':
        return no_update, label
    return _figure_for(payload['id'], int(t or 0), get_store()), label


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
    подхватывает существующий колбэк _update (у него уже есть Input на time-slider.value)."""
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
    prevent_initial_call=True,
)
def _save_html(n, payload, t):
    """Сохранение в HTML: всегда полная анимация текущего вида
    (Results.html / fi_func.html; для временных рядов — статичный график)."""
    if not n or not isinstance(payload, dict):
        return no_update
    store = get_store()
    t = int(t or 0)
    animate = payload['kind'] in ('map', 'fi')
    fig = _figure_for(payload['id'], t, store, animate=animate)

    if payload['kind'] == 'fi':
        fname = 'fi_func.html'
    elif payload['kind'] == 'map':
        fname = f'{fig.data[0].name}.html'
    else:
        fname = f'{_sanitize(payload["group"])}.html'

    html_str = fig.to_html(include_plotlyjs=str(js_path), full_html=True)
    return dcc.send_bytes(html_str.encode('utf-8'), filename=fname)


def visualize_solution(port: int = 8050, debug: bool = False):
    """Запуск интерактивной визуализации.

    Чтобы открыть с другого устройства, запустите с host='0.0.0.0', и на другом устройстве откройте http://<IP>:<порт>.
    Разрешите порт в брандмауэре. 0.0.0.0 открывает доступ всем в локальной сети.
    """
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
