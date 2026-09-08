"""Интерактивная визуализация больших расчётных данных (Dash) с ленивой загрузкой.

Оптимизации:
1. Кэш на диске: pkl -> npy (float32, memory-mapped). Повторные запуски мгновенны,
   в RAM держится только текущий срез времени, а не весь куб данных.
2. Лёгкая фигура: строится только текущее поле в текущий момент времени;
   слайдер времени обновляет z/y точечно через dash.Patch (без пересылки layout).
3. Даунсэмплинг сетки для экрана (MAX_DISPLAY_CELLS).
4. Сохранение в HTML — всегда полная анимация текущего вида (с именами файлов
   как в исходной версии: Results.html / fi_func.html).

Оформление фигур соответствует исходным plotly-графикам (размеры, сетка,
легенды, контуры, подписи слайдера).
"""
from __future__ import annotations

import gc
import json
import math
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

from paraphin import r, fi_0
from paraphin.constants import (Nx, Ny, X_min, X_max, hx, hy, Y_max, Y_min, results_path, layers_file, js_path,
                                bar_to_pa, day_to_sec, CONTOUR_PLOT, S_max, case_name)
from paraphin.utils.read_data_files import read_solution_data, convert_pkl_files

# ----------------------------- Настройки ------------------------------------
CACHE_VERSION = 6                       # инкрементируется при изменении формата кэша
CACHE_DIR = results_path.parent / f'{case_name}_vizcache'
DTYPE = np.float32                      # тип хранения на диске (x2 экономия RAM/диска)
MAX_DISPLAY_CELLS = 250_000             # максимум точек сетки для экрана (None — без прореживания)
MAX_MARKS = 12                          # максимум подписей на слайдере времени
JSON_DECIMALS = 5                       # округление данных перед отправкой в браузер (None — откл.)
SKIP_FIELDS = {'Pressure', 'Wps', 'Wo', 'Wp', 'Wps dep', 'qp', 'mu_o', 'mu_w'}
SERIES_GROUPS = ('Wells', 'Wells_accumulated', 'Other params')

x_mesh = np.linspace(X_min + hx / 2, X_max - hx / 2, Nx)
y_mesh = np.linspace(Y_min + hy / 2, Y_max - hy / 2, Ny)

names_converter = {'fi': '$$\\varphi$$', 'fi_o': '$$\\varphi_0$$'}
_HOVER_MAP = 'X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>'

_BTN_ACTIVE = {'fontWeight': 'bold', 'backgroundColor': '#4CAF50', 'color': 'white',
               'border': 'none', 'padding': '7px 14px', 'borderRadius': '4px', 'cursor': 'pointer'}
_BTN_INACTIVE = {'fontWeight': 'normal', 'backgroundColor': '#e0e0e0', 'color': 'black',
                 'border': 'none', 'padding': '7px 14px', 'borderRadius': '4px', 'cursor': 'pointer'}
_BTN_SAVE = {'backgroundColor': '#2196F3', 'color': 'white', 'fontWeight': 'bold',
             'border': 'none', 'padding': '7px 14px', 'borderRadius': '4px'}
_BTN_HEADER = {'fontWeight': 'bold', 'color': '#555', 'fontSize': '13px', 'margin': '0 4px'}


def _day_fmt(t) -> str:
    """Формат подписи времени как в оригинале: f'{round(time[i], 1)} день'."""
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

        if not self._cache_actual():
            print('Кэш визуализации отсутствует или устарел — однократный сбор...')
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
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

    def _cache_actual(self) -> bool:
        meta_file = CACHE_DIR / 'meta.json'
        if not meta_file.is_file():
            return False
        try:
            meta = json.loads(meta_file.read_text(encoding='utf-8'))
        except Exception:
            return False
        if meta.get('version') != CACHE_VERSION:
            return False
        src = self._source_stat()
        return src is None or meta.get('source') == src

    def _build_cache(self, raw: dict):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        meta = {'version': CACHE_VERSION, 'source': self._source_stat()}

        time = np.asarray(raw.pop('Time'), dtype=np.float64) / day_to_sec
        n_times = int(len(time))
        meta['n_times'] = n_times
        meta['time'] = [round(float(t), 6) for t in time]

        # Приводим к "экранному" виду один раз, при сборе кэша
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

        # --- временные ряды (маленькие, приводим к экранному виду заранее) ---
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
                    visible = not (np.all(np.isclose(v, 0.0)) or np.all(np.isclose(v, 1.0)))
                    if 'eta' in k:
                        y, ax, unit = v, 'y2', ''
                    else:
                        y, ax, unit = np.abs(v) * day_to_sec, 'y', 'м^3/день'
                elif g == 'Wells_accumulated':
                    visible, y, ax, unit = not np.all(np.isclose(v, 0.0)), np.abs(v), 'y', 'м^3'
                else:
                    visible, y, ax, unit = True, np.abs(v), 'y', ''
                names.append(k); axes.append(ax); units.append(unit); vis.append(visible)
                rows.append(y.astype(DTYPE))
            if names:
                fname = f'series__{_sanitize(g)}.npy'
                np.save(CACHE_DIR / fname, np.asarray(rows, dtype=DTYPE))
                series[g] = {'file': fname, 'names': names, 'axes': axes, 'units': units, 'visible': vis}
                print(f'  {g}: {len(names)} кривых')
        meta['series'] = series

        # --- графики phi(r) (маленькие 2D-массивы) ---
        plots = {}
        for k, v in (raw.get('plots') or {}).items():
            v = np.asarray(v, dtype=DTYPE)
            fname = f'plots__{_sanitize(k)}.npy'
            np.save(CACHE_DIR / fname, v)
            plots[k] = {'file': fname, 'shape': list(v.shape)}
        meta['plots'] = plots
        meta['has_plots'] = bool(plots)

        (CACHE_DIR / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
        print('Кэш собран:', CACHE_DIR)

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


def _map_figure(view: dict, i: int, store: SolutionStore, animate: bool = False) -> go.Figure:
    """Поле данных — оформление как в _visualize_fields оригинала (1000×800, Jet, подписи контуров)."""
    if view.get('sattemp'):
        # Вид 'Sat and Temp' — точно как в оригинале: заливка насыщенности + чёрные изолинии температуры
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

    # Layout оригинала: белый фон, чёрная сетка, легенда справа, размер 1000×800.
    # scaleanchor/scaleratio — сохранение отношения сторон (по доп. требованию).
    xax = _axis_dict()
    xax['constrain'] = 'domain'
    yax = _axis_dict()
    yax.update(scaleanchor='x', scaleratio=1)
    fig.update_layout(plot_bgcolor='white', uirevision='map',
                      xaxis=xax, yaxis=yax, legend=dict(x=1.05, y=1.0),
                      width=1000, height=800)

    if animate:   # полная анимация для сохранения в HTML — слайдер как в оригинале
        steps = [dict(method='update',
                      args=[{'z': [store.get_field(nm, j) for nm in view['sources']]}],
                      label=_day_fmt(store.time[j]))
                 for j in range(store.meta['n_times'])]
        fig.update_layout(sliders=[dict(active=min(i, len(steps) - 1),
                                        currentvalue={'prefix': 'Время: '}, steps=steps)])
    return fig


def _series_figure(view: dict, store: SolutionStore) -> go.Figure:
    """Скважины / прочие параметры — оформление как в оригинале (часть _visualize_fields)."""
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
                      width=1000, height=800)
    return fig


def _fi_figure(view: dict, i: int, store: SolutionStore, animate: bool = False) -> go.Figure:
    """Графики phi(r) — оформление как в _visualize_plots_fi оригинала (height=600, margin t=0 b=0)."""
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
        height=600,
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
def _valid_view_ids(meta: dict) -> list[str]:
    """Упорядоченный список доступных видов (порядок совпадает с порядком кнопок)."""
    ids = [f'field:{n}' for n in meta['fields'] if n not in SKIP_FIELDS]
    if {'Saturation', 'Temperature'} <= set(meta['fields']):
        ids.append('sattemp')
    ids += [f'series:{g}' for g in SERIES_GROUPS if g in meta['series']]
    if meta.get('has_plots'):
        ids.append('fi')
    return ids


def _view_buttons(meta: dict, current: str | None) -> list:
    """Кнопки выбора отображения, сгруппированные заголовками."""
    def btn(vid: str, label: str) -> html.Button:
        return html.Button(label, id={'type': 'view-btn', 'index': vid}, n_clicks=0,
                           style=_BTN_ACTIVE if vid == current else _BTN_INACTIVE)

    els = []
    fields = [n for n in meta['fields'] if n not in SKIP_FIELDS]
    if fields:
        els.append(html.Span('Поля:', style=_BTN_HEADER))
        els += [btn(f'field:{n}', n) for n in fields]
        if {'Saturation', 'Temperature'} <= set(meta['fields']):
            els.append(btn('sattemp', 'Sat and Temp'))
    groups = [g for g in SERIES_GROUPS if g in meta['series']]
    if groups:
        els.append(html.Span('Скважины / прочее:', style=_BTN_HEADER))
        els += [btn(f'series:{g}', g) for g in groups]
    if meta.get('has_plots'):
        els.append(html.Span('φ(r):', style=_BTN_HEADER))
        els.append(btn('fi', 'Графики φ(r)'))
    return els


def _resolve_view(view_id: str | None, store: SolutionStore) -> str | None:
    valid = _valid_view_ids(store.meta)
    return view_id if view_id in valid else (valid[0] if valid else None)


def _payload(view_id: str | None, store: SolutionStore) -> dict | None:
    if view_id is None:
        return None
    v = _view_def(view_id, store)
    return {'id': v['id'], 'kind': v['kind'], 'sources': v.get('sources', []), 'group': v.get('group')}


def _slider_style(kind: str, n_times: int = 2) -> dict:
    if kind == 'series' or n_times <= 1:
        return {'display': 'none'}
    return {'display': 'flex', 'alignItems': 'center', 'gap': '16px',
            'margin': '4px auto 0', 'maxWidth': '1000px', 'minHeight': '36px'}


def _marks(store: SolutionStore) -> dict:
    n = len(store.time)
    if n <= 1:
        return {'0': ''}
    idx = np.unique(np.linspace(0, n - 1, min(n, MAX_MARKS)).astype(int))
    return {int(i): _day_fmt(store.time[i]) for i in idx}


def _time_label(store: SolutionStore, i) -> str:
    """Текущее значение слайдера в формате currentvalue оригинала: 'Время: 123.4 день'."""
    return f'Время: {_day_fmt(store.time[int(i or 0)])}'


def _empty_figure(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=20))
    fig.update_layout(width=1000, height=800)
    return fig


app = Dash(__name__)
app.title = f'Визуализация — {case_name}'
app.config.suppress_callback_exceptions = True   # кнопки создаются колбэком инициализации

app.layout = html.Div([
    dcc.Interval(id='boot', interval=50, max_intervals=1),
    html.Div(
        dcc.Loading(html.Div(id='ui-root'), type='circle'),
        style={'margin': '30px'}
    ),
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
    show_slider = view_id is not None and kind != 'series' and n_times > 1

    return html.Div([
        html.H3(f'Визуализация решения — {case_name}', style={'textAlign': 'center', 'margin': '8px 0'}),

        html.Div(_view_buttons(store.meta, view_id),
                 style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '6px', 'alignItems': 'center',
                        'justifyContent': 'center', 'margin': '6px 0'}),

        html.Div([
            html.Button('💾 Сохранить анимацию в HTML', id='btn-save', n_clicks=0, style=_BTN_SAVE),
            dcc.Download(id='download'),
        ], style={'display': 'flex', 'justifyContent': 'center', 'margin': '6px 0'}),

        dcc.Loading(
            html.Div(
                dcc.Graph(id='main-graph',
                          figure=_figure_for(view_id, 0, store) if view_id else _empty_figure('Нет данных'),
                          responsive=False,                      # фиксированный размер фигуры, как в оригинале
                          style={'width': 'fit-content'}),
                style={'display': 'flex', 'justifyContent': 'center', 'overflowX': 'auto'},
            ),
            type='circle'),

        html.Div([
            html.Div(id='time-label',
                     children=_time_label(store, 0) if show_slider else '',
                     style={'minWidth': '160px', 'fontWeight': 'bold'}),
            dcc.Slider(id='time-slider', min=0, max=max(0, n_times - 1), step=1, value=0,
                       marks=_marks(store), disabled=n_times <= 1, tooltip={'placement': 'bottom'}),
        ], id='slider-box',
           style=_slider_style(kind, n_times) if show_slider else {'display': 'none'}),

        dcc.Store(id='view-state', data=_payload(view_id, store)),
    ], style={'maxWidth': '1600px', 'margin': '0 auto', 'padding': '0 10px'})


@app.callback(
    Output('main-graph', 'figure'),
    Output('view-state', 'data'),
    Output('slider-box', 'style'),
    Output('time-label', 'children'),
    Output({'type': 'view-btn', 'index': ALL}, 'style'),
    Input({'type': 'view-btn', 'index': ALL}, 'n_clicks'),
    Input('time-slider', 'value'),
    State('view-state', 'data'),
    prevent_initial_call=True,
)
def _update(clicks, t, prev):
    """Кнопки вида и слайдер времени: в браузер уходит только изменившееся."""
    store = get_store()
    t = int(t or 0)
    valid = _valid_view_ids(store.meta)

    # Защита от срабатывания при динамическом создании кнопок
    if all(c is None for c in (clicks or [])) and ctx.triggered_id != 'time-slider':
        return no_update, no_update, no_update, no_update, no_update

    trig = ctx.triggered_id

    # --- нажата кнопка вида: пересобираем лёгкую фигуру из 1-2 трейсов ---
    if isinstance(trig, dict) and trig.get('type') == 'view-btn':
        view_id = _resolve_view(trig['index'], store)
        if view_id is None:
            return (_empty_figure('Нет данных для отображения'), None, {'display': 'none'}, '',
                    [_BTN_INACTIVE] * len(valid))
        v = _view_def(view_id, store)
        return (_figure_for(view_id, t, store), _payload(view_id, store), _slider_style(v['kind'], store.meta['n_times']),
                _time_label(store, t) if v['kind'] != 'series' else '',
                [_BTN_ACTIVE if vid == view_id else _BTN_INACTIVE for vid in valid])

    # --- движение слайдера времени: точечное обновление z/y через Patch ---
    if not isinstance(prev, dict) or prev['kind'] == 'series':
        return no_update, no_update, no_update, no_update, no_update

    patch = Patch()
    if prev['kind'] == 'map':
        for k, name in enumerate(prev['sources']):
            patch['data'][k]['z'] = store.get_field(name, t).tolist()
    else:  # fi
        data = store.get_plot_data()
        for k, name in enumerate(prev['sources']):
            patch['data'][k]['y'] = data[name][t].tolist()
    return patch, no_update, no_update, _time_label(store, t), no_update


@app.callback(
    Output('download', 'data'),
    Input('btn-save', 'n_clicks'),
    State('view-state', 'data'),
    State('time-slider', 'value'),
    prevent_initial_call=True,
)
def _save_html(n, payload, t):
    """Сохранение в HTML: всегда полная анимация текущего вида, с оригинальными
    именами файлов (Results.html / fi_func.html) и оригинальным встроенным слайдером Plotly."""
    if not n or not isinstance(payload, dict):
        return no_update
    store = get_store()
    t = int(t or 0)
    animate = payload['kind'] in ('map', 'fi')
    fig = _figure_for(payload['id'], t, store, animate=animate)

    if payload['kind'] == 'fi':
        fname = 'fi_func.html'
    elif payload['kind'] == 'map':
        fname = 'Results.html'
    else:
        fname = f'{_sanitize(payload["group"])}.html'

    html_str = fig.to_html(include_plotlyjs=str(js_path), full_html=True)
    return dcc.send_bytes(html_str.encode('utf-8'), filename=fname)


def visualize_solution(port: int = 8050, debug: bool = False):
    """Запуск интерактивной визуализации: http://127.0.0.1:<port>."""
    runner = getattr(app, 'run', None) or app.run_server
    runner(host='127.0.0.1', port=port, debug=debug)


if __name__ == '__main__':
    visualize_solution(debug=True)