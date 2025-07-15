"""Пакет содержит утилиты для отображения результатов в виде графиков и интерактивных полей данных."""
from .read_data_files import convert_pkl_files, read_solution_data
from .visualisation import show_plot, visualize_solution
from .graphs import create_graphs_and_maps
from  .gif_animation import create_gif
