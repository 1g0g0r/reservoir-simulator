"""Пакет содержит утилиты для отображения результатов в виде графиков и интерактивных полей данных."""
import numpy as np

from paraphin.constants import Nx, Ny, X_max, X_min, Y_min, Y_max
from .gif_animation import create_gif
from .graphs import create_graphs_and_maps
from .read_data_files import convert_pkl_files, read_solution_data
from .save_in_eps_format import plotly_to_eps
from .visualisation import show_plot, visualize_solution

x_mesh = np.linspace(X_min, X_max, Nx)
y_mesh = np.linspace(Y_min, Y_max, Ny)

names_converter = {
    'fi': '$$\\varphi$$',
    'fi_o': '$$\\varphi_0$$'
}
