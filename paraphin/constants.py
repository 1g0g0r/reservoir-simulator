from pathlib import Path

import numpy as np
import taichi as ti

# Инициализация ядра taichi
data_type = ti.f64
ti.init(arch=ti.cpu, default_fp=data_type)  # , kernel_profiler=True
LOGGING = False
CONTOUR_PLOT = True

# Перевод единиц измерения
day_to_sec = 86400.0
bar_to_pa = 1e5
darcy_to_m2 = 9.869233e-13
cpoise_to_Pas = 1e-3
kal_to_J = 4.1868

# Параметры сетки
Nr = 31          # Число узлов сетки радиусов капилляров
Nx, Ny = 25, 25  # Число узлов сетки по x и y
X_min, X_max = 0., 200.  # Длина пласта, [м]
Y_min, Y_max = 0., 200.  # Ширина пласта, [м]
h = 10.0  # Толщина пласта, [м]
hx = (X_max - X_min) / Nx
hy = (Y_max - Y_min) / Ny
area = hx * hy
volume = area * h

# Параметры времени задачи
Time_end = day_to_sec * 365 * 6.0  # Время моделирования
dt = day_to_sec / 4e1  # Шаг дискретизации по времени
max_eta = 0.98         # Предельная обводненность

# Параметры скважин
rw = 0.1              # Радиус скважин, [м]
Pw = 150 * bar_to_pa  # Давление на нагнетательной скважине, [Па]
Po = 50 * bar_to_pa   # Давление на добывающей скважине, [Па]
Twater = 25           # Температура нагнетаемой воды, [С]

# Параметры ОФП
S_min = 0.18
S_max = 0.70
n_power = 2

# Данные инициализации
init_p   = (Pw + Po) / 2  # [Па]
init_S   = S_min
init_Wp  = 0.05
init_Wps = 0.0
init_k   = 0.2 * darcy_to_m2  # [м^2]
init_m   = 0.2
init_T   = 70  # [C]
init_qp  = 0.0
init_h_sloy = 0.0
porous_volume = (X_max - X_min) * (Y_max - Y_min) * h * init_m
geological_reserves = porous_volume * (1.0 - init_S)  # Геологические запасы пласта

# Параметры флюидов
ro_w  = 1000.0  # Плотность воды, [кг/м^3]
ro_o  = 860.0   # Плотность нефти, [кг/м^3]
ro_p  = 900.0   # Плотность парафина, [кг/м^3]
ro_f  = 2700.0  # Плотность пласта, [кг/м^3]
ro_ff = 2400.0  # Плотность окружающих пласт пород, [кг/м^3]
mu_w = 1.0 * cpoise_to_Pas  # Вязкость воды, [Па·с]
mu_o = 5.0 * cpoise_to_Pas  # Вязкость нефти, [Па·с]
c_w  = 4200 # Теплоемкость воды, [Дж/(кг·C)]
c_o  = 2000 # Теплоемкость нефти, [Дж/(кг·C)]
c_p  = 2200 # Теплоемкость парафина, [Дж/(кг·C)]
c_f  = 1000 # Теплоемкость пласта, [Дж/(кг·C)]
c_ff = 1000 # Теплоемкость окружающих пласт пород, [Дж/(кг·C)]
K_p  = 0.2  # Теплопроводность парафина, [Вт/(м·С)]
K_w  = 0.6  # Теплопроводность воды, [Вт/(м·С)]
K_o  = 0.12 # Теплопроводность нефти, [Вт/(м·С)]
K_f  = 1.8  # Теплопроводность пласта, [Вт/(м·С)]
K_ff = 2.0  # Теплопроводность окружающих пород пласта, [Вт/(м∙C)]

# Параметры моделирования кольматации\суффозии
g     = 9.81   # Ускорение свободного падения, [м/с^2]
Diff  = 1e-19  # Коэффициент диффузионного осаждения частиц, [м^2/с]
D     = 4e-6   # Размер частицы, [м]
betta = 0.01   # Доля блокируемых каналов, [-]
gamma = 0.4    # Отношение радиуса горла к радиусу канала, [-]
Lk    = 3e-5   # Длина капилляра, [м]
eta   = 1.0    # Коэффициент извилистости, [-]
Cf    = 3e-2   # Коэффициент сопротивления частицы в нефти, [-]
Delta = 0.005  # Кинетическая константа суффозии, [1/м]
R     = 8.31446261815324  # Газовая постоянная, [J/K/mol]

# Параметры парафина
MW = 350.0  # Молекулярная масса
Tm = 374.5 + 0.02617 * MW - 2.0172e4 / MW - 273.15 # Температура кристаллизации парафина, [C]
alpha = 0.1426 * MW * (Tm + 273.15) * kal_to_J     # Скрытая теплота плавления парафина, [J/gram-mol]

# Проверка числа Куранта
_re = 0.14 * np.sqrt(hx * hx + hy * hy)  # Радиус контура питания скважины, [м]
_u_aver = init_k / mu_o * Pw * 0.05      # Примерная средняя скорость потока
Courant_num = _u_aver * dt / _re         # Число Куранта
if Courant_num > 0.8:
    dt = 0.02 * _re / _u_aver
    dt = round(dt / day_to_sec, 5) * day_to_sec
    print(f'Не выполнено условие Куранта!! Новый шаг по времени {dt / day_to_sec} сут.')
# else:
#     dt /= (Courant_num / 0.8)
#     dt = 0.01 * round(dt / day_to_sec, 5) * day_to_sec
#     print(f'Новый шаг по времени увеличен до значения {dt / day_to_sec} сут.')

sol_time_step = dt * 50  # шаг по времени для сохранения результатов

if data_type == ti.f64:
    np_data_type = np.float64
else:
    np_data_type = np.float32

# Пути проекта
root_folder = Path(__file__).parent.parent
outputs_path = root_folder / 'outputs'
pictures_path = outputs_path / 'pictures'
results_path = outputs_path / f'results_wp={init_Wp}'
data_path = outputs_path / 'data'
logs_path = outputs_path / '.log'
js_path = root_folder / 'paraphin' / 'utils' / 'visualisation_utils' / 'plotly_script.js'
