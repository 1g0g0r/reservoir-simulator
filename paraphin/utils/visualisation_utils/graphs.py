"""Создание графиков и двумерных полей в формате .svg из предобработанных данных расчета."""
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from .read_data_files import read_solution_data
from paraphin.constants import pictures_path, bar_to_pa, day_to_sec, S_min, init_T, Twater


@dataclass
class FieldSettings:
    """Настройки для визуализации поля."""
    name: str
    start: float
    end: float
    step: float


@dataclass
class PlotSettings:
    """Настройки для создания графиков."""
    name_plot1: str
    name_plot2: str
    right_axis_title: str
    left_axis_title: str


class DataVisualizer:
    """Базовый класс для визуализации данных."""

    def __init__(self, output_path: Path = None):
        """
        Инициализация визуализатора.

        Args:
            output_path: Путь для сохранения изображений
        """
        self.output_path = output_path or pictures_path
        self.data = None
        self.data_wp = None

    def load_data(self, data_path: str = 'Wp=0.0_processed_data.pkl',
                  data_wp_path: str = 'Wp=0.05_processed_data.pkl'):
        """Загрузка и подготовка данных."""
        from . import x_mesh, y_mesh

        self.x_mesh = x_mesh
        self.y_mesh = y_mesh

        _, self.data = read_solution_data(data_path)
        _, self.data_wp = read_solution_data(data_wp_path)

        # Конвертация единиц измерения
        self.data['Pressure'] /= bar_to_pa
        self.data_wp['Pressure'] /= bar_to_pa
        self.data['Time'] /= day_to_sec
        self.data_wp['Time'] /= day_to_sec

    def _prepare_output_directory(self):
        """Создание директории для сохранения изображений."""
        self.output_path.mkdir(parents=True, exist_ok=True)


class FieldVisualizer(DataVisualizer):
    """Класс для визуализации полей данных."""

    def create_fields(self, field_settings: List[FieldSettings], time_idx: int = None,
                      time_idx_wp: int = None):
        """
        Создание визуализаций для полей.

        Args:
            field_settings: Список настроек полей
            time_idx: Индекс времени для данных без Wp
            time_idx_wp: Индекс времени для данных с Wp
        """
        self._prepare_output_directory()

        if time_idx is None or time_idx_wp is None:
            time_idx, time_idx_wp = self._calculate_time_indices()

        for settings in field_settings:
            self._create_field_visualization(
                time_idx, time_idx_wp, settings
            )

    def _calculate_time_indices(self) -> Tuple[int, int]:
        """Расчет временных индексов."""
        idx_end_wp = len(self.data_wp['Time']) - 1
        idx_sat_wp = np.argwhere(self.data_wp['Wells']['Producer_eta'] != 0)[0][0]
        idx_end = len(self.data['Time']) - 1
        idx_sat = np.argwhere(self.data['Wells']['Producer_eta'] != 0)[0][0]

        return idx_end, idx_end

    def _create_field_visualization(self, idx: int, idx_wp: int,
                                    settings: FieldSettings):
        """Создание визуализации поля."""
        fig = self._create_field_figure(idx, idx_wp, settings)
        self._save_field_figure(fig, settings.name, idx_wp)

    def _create_field_figure(self, idx: int, idx_wp: int,
                             settings: FieldSettings) -> go.Figure:
        """Создание фигуры с полем."""
        fig = go.Figure()

        field_wp = self.data_wp[settings.name][idx_wp]
        field = self.data[settings.name][idx]

        # Добавление контуров для Wp=5%
        fig.add_trace(go.Contour(
            x=self.x_mesh, y=self.y_mesh, z=field_wp, name='Wp=5%',
            contours=dict(
                coloring='lines', showlabels=True,
                start=settings.start, end=settings.end, size=settings.step,
            ),
            line=dict(width=3), colorscale=[[0, 'black'], [1, 'black']],
            showscale=False, showlegend=True
        ))

        # Добавление контуров для Wp=0%
        fig.add_trace(go.Contour(
            x=self.x_mesh, y=self.y_mesh, z=field, name='Wp=0%',
            contours=dict(
                coloring='lines', showlabels=True,
                start=settings.start, end=settings.end, size=settings.step,
            ),
            line=dict(width=3, dash='dash'), colorscale=[[0, 'red'], [1, 'red']],
            showscale=False, showlegend=True
        ))

        fig = self._apply_plot_params(fig, 'X', 'Y')
        return fig

    def _save_field_figure(self, fig: go.Figure, field_name: str, idx_wp: int):
        """Сохранение фигуры поля."""
        filename = f"{field_name}_{round(self.data_wp['Time'][idx_wp], 2)}.svg"
        fig.write_image(self.output_path / filename, width=700, height=600)

        # Конвертация в EPS
        from paraphin.utils import plotly_to_eps
        plotly_to_eps(fig_plotly=fig, filename=field_name, dpi=1200)


class PlotVisualizer(DataVisualizer):
    """Класс для создания графиков."""

    def create_plots(self, plot_settings: List[PlotSettings]):
        """
        Создание графиков.

        Args:
            plot_settings: Список настроек графиков
        """
        self._prepare_output_directory()

        for settings in plot_settings:
            self._create_plot_visualization(settings)

    def _create_plot_visualization(self, settings: PlotSettings):
        """Создание визуализации графика."""
        fig = self._create_plot_figure(settings)
        self._save_plot_figure(fig, settings.name_plot1, settings.name_plot2)

    def _create_plot_figure(self, settings: PlotSettings) -> go.Figure:
        """Создание фигуры графика."""
        fig = go.Figure()
        time = self.data_wp['Time']

        # Подготовка данных для первого графика
        plot_data1, plot_data_wp1 = self._prepare_plot_data(
            settings.name_plot1, self.data['Wells'], self.data_wp['Wells']
        )

        # Подготовка данных для второго графика
        plot_data2, plot_data_wp2 = self._prepare_plot_data(
            settings.name_plot2, self.data['Wells'], self.data_wp['Wells']
        )

        # Добавление трассировок
        fig.add_trace(go.Scatter(
            x=time, y=plot_data_wp1, yaxis='y2',
            mode='lines', name='Wp=5%',
            line=dict(color='black', width=3), showlegend=True
        ))

        fig.add_trace(go.Scatter(
            x=time, y=plot_data1, yaxis='y2',
            mode='lines', name='Wp=0%',
            line=dict(color='red', width=3, dash='dash'), showlegend=True
        ))

        fig.update_layout(
            yaxis2=dict(
                side="right", overlaying="y",
                title=settings.right_axis_title,
                domain=[0.0, 0.5],
                title_font=dict(size=18)
            )
        )

        fig.add_trace(go.Scatter(
            x=time, y=plot_data_wp2,
            mode='lines', line=dict(color='black', width=3), showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=time, y=plot_data2,
            mode='lines', line=dict(color='red', width=3, dash='dash'), showlegend=False
        ))

        fig = self._apply_plot_params(fig, "$$t,\\: сут$$", settings.left_axis_title)
        return fig

    def _prepare_plot_data(self, name: str, data: Dict, data_wp: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для построения графика."""
        if 'eta' in name or 'Q' in name:
            plot_data = np.abs(data[name])
            plot_data_wp = np.abs(data_wp[name])
        else:
            plot_data = np.abs(data[name]) * day_to_sec
            plot_data_wp = np.abs(data_wp[name]) * day_to_sec

        return plot_data, plot_data_wp

    def _save_plot_figure(self, fig: go.Figure, name1: str, name2: str):
        """Сохранение фигуры графика."""
        filename = f"{name1}_{name2}.svg"
        fig.write_image(self.output_path / filename, width=700, height=600)

        # Конвертация в EPS
        from paraphin.utils import plotly_to_eps
        plotly_to_eps(fig_plotly=fig, filename=f"{name1}_{name2}", dpi=1200)


class MapVisualizer(DataVisualizer):
    """Класс для создания карт."""

    def create_maps(self, map_settings: List[Tuple[str, float, float, float]],
                    time_idx_wp: int = None):
        """
        Создание карт.

        Args:
            map_settings: Список настроек карт
            time_idx_wp: Индекс времени
        """
        self._prepare_output_directory()

        if time_idx_wp is None:
            time_idx_wp = len(self.data_wp['Time']) - 1

        for settings in map_settings:
            self._create_map_visualization(time_idx_wp, *settings)

    def _create_map_visualization(self, idx_wp: int, field_name: str,
                                  start: float, end: float, step: float):
        """Создание визуализации карты."""
        fig = self._create_map_figure(idx_wp, field_name, start, end, step)
        self._save_map_figure(fig, field_name, idx_wp)

    def _create_map_figure(self, idx_wp: int, field_name: str,
                           start: float, end: float, step: float) -> go.Figure:
        """Создание фигуры карты."""
        fig = go.Figure()
        field = self.data_wp[field_name][idx_wp]

        fig.add_trace(go.Contour(
            x=self.x_mesh, y=self.y_mesh, z=field,
            contours=dict(coloring='fill', showlabels=True),
            colorscale='Jet',
            showscale=True, showlegend=False
        ))

        fig.update_layout(
            plot_bgcolor='white', xaxis_title='X', yaxis_title='Y',
            height=500, width=600, showlegend=True,
            margin=dict(t=0, b=0),
            xaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1),
            yaxis=dict(showgrid=True, gridcolor='black', linecolor='black', linewidth=1),
        )

        return fig

    def _save_map_figure(self, fig: go.Figure, field_name: str, idx_wp: int):
        """Сохранение фигуры карты."""
        filename = f"{field_name}_{round(self.data_wp['Time'][idx_wp], 2)}.svg"
        fig.write_image(self.output_path / filename, width=700, height=600)


class VisualizationManager:
    """Менеджер для координации всех визуализаций."""

    def __init__(self):
        """Инициализация менеджера визуализации."""
        self.field_visualizer = FieldVisualizer()
        self.plot_visualizer = PlotVisualizer()
        self.map_visualizer = MapVisualizer()

    def create_all_visualizations(self):
        """Создание всех типов визуализаций."""
        # Загрузка данных всеми визуализаторами
        self.field_visualizer.load_data()
        self.plot_visualizer.load_data()
        self.map_visualizer.load_data()

        # Настройки для полей
        field_settings = [
            FieldSettings('Pressure', 50, 150, 2),
            FieldSettings('Saturation', S_min, 1, 0.03),
            FieldSettings('Temperature', 25*1.001, init_T*0.99, 10)
        ]

        # Создание визуализаций полей
        self.field_visualizer.create_fields(field_settings)

        # Настройки для графиков
        plot_settings = [
            PlotSettings('Producer_oil', 'Producer_water',
                        '$$q_o,\\: м^3 \\setminus сут$$',
                        '$$q_w,\\: м^3 \\setminus сут$$'),
            PlotSettings('Producer_Q_oil', 'Producer_Q_water',
                        '$$Q_o,\\: м^3$$', '$$Q_w,\\: м^3$$')
        ]

        # Создание графиков
        self.plot_visualizer.create_plots(plot_settings)

        # Настройки для карт
        map_settings = [
            ['m', 0, 1, 0.03],
            ['k', 0, 1, 0.03],
            ['Wps dep', 0, 0.05, 0.03],
            ['Wps', 0, 0.05, 0.03],
            ['Wp', 0, 0.05, 0.03]
        ]

        # Создание карт
        self.map_visualizer.create_maps(map_settings)

        print('Done!')

    @staticmethod
    def _apply_plot_params(fig: go.Figure, x_axis_title: str,
                           y_axis_title: str) -> go.Figure:
        """Настройки внешнего вида графиков."""
        fig.update_layout(
            plot_bgcolor='white', xaxis_title=x_axis_title, yaxis_title=y_axis_title,
            height=500, width=600, showlegend=False,
            margin=dict(t=0, b=0),
            xaxis=dict(showgrid=True, gridcolor='black', linecolor='black',
                      linewidth=1, title_font=dict(size=18)),
            yaxis=dict(showgrid=True, gridcolor='black', linecolor='black',
                      linewidth=1, title_font=dict(size=18)),
        )

        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="black", width=1)
        )
        fig.add_hline(y=0, line=dict(color='black', width=1))

        return fig


# Функция для обратной совместимости
def create_graphs_and_maps():
    """Создание всех графиков и карт (старая функция)."""
    manager = VisualizationManager()
    manager.create_all_visualizations()