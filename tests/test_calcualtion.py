"""Сравнение текущей версии солвера с потенциально достоверной.

Сравнение идет по физическому времени с интерполяцией на общую сетку, а не поэлементно по индексу
сохранения: шаг по времени подбирается адаптивно по числу Куранта, поэтому моменты сохранения у
эталона и у текущего расчета не совпадают.

Перед запуском нужно посчитать задачу (python start.py) с теми же константами, при которых получен
эталон, - тест только сравнивает уже сохраненные результаты.
"""
import numpy as np
from pathlib import Path
from pickle import load

from start import solve
from paraphin.constants import init_Wp
from paraphin.utils import read_solution_data

# TODO пиклить класс решения, чтобы хранить всю информацию о решении (данные о сетке и времени расчета). По возможности этот же пикл исопльзовать в тесте МРСТ
# Параллельно запускать расчеты wp=5 wp=0

TEST_DATA_PATH = Path(__file__).parent / 'test_data'
RTOL_WELLS = 2e-2   # допуск на кривые скважин: адаптивный шаг дает другую дискретизацию по времени
ATOL_FIELDS = 5e-3  # допуск на поля данных в сопоставимые моменты времени


def _load_pair():
	"""Эталон и текущий расчет для одного и того же значения init_Wp."""
	with open(TEST_DATA_PATH / f'Wp={init_Wp}_processed_data.pkl', 'rb') as file:
		_, test_data = load(file)

	# solve()  # раскомментировать, если нужно пересчитать задачу прямо из теста
	_, calculation_data = read_solution_data(f'Wp={init_Wp}_processed_data.pkl')

	return test_data, calculation_data


def _common_times(test_data, calculation_data):
	"""Сетка времени, покрытая обоими расчетами."""
	t_ref, t_new = np.asarray(test_data['Time']), np.asarray(calculation_data['Time'])
	times = t_ref[t_ref <= min(t_ref.max(), t_new.max())]

	assert times.size > 1, 'расчеты не пересекаются по времени, сравнивать нечего'

	return times, t_ref, t_new


def test_wells():
	"""Показатели работы скважин и КИН как функции времени."""
	test_data, calculation_data = _load_pair()
	times, t_ref, t_new = _common_times(test_data, calculation_data)

	groups = [name for name in ('Wells', 'Wells_accumulated', 'Other params')
			  if name in test_data and name in calculation_data]

	for group in groups:
		for name, ref_values in test_data[group].items():
			if name not in calculation_data[group]:
				continue

			ref = np.interp(times, t_ref, np.asarray(ref_values))
			new = np.interp(times, t_new, np.asarray(calculation_data[group][name]))
			error = np.abs(ref - new).max() / max(np.abs(ref).max(), 1e-30)

			assert error <= RTOL_WELLS, f'{group}/{name}: относительное отклонение {error:.4%} > {RTOL_WELLS:.4%}'


def test_fields():
	"""Поля данных в последний общий момент времени."""
	test_data, calculation_data = _load_pair()
	times, t_ref, t_new = _common_times(test_data, calculation_data)
	i_ref = int(np.argmin(np.abs(t_ref - times[-1])))
	i_new = int(np.argmin(np.abs(t_new - times[-1])))

	for name, ref_values in test_data.items():
		if isinstance(ref_values, dict) or name == 'Time' or name not in calculation_data:
			continue

		ref = np.asarray(ref_values)[i_ref]
		error = np.abs(ref - np.asarray(calculation_data[name])[i_new]).max() / max(np.abs(ref).max(), 1e-30)

		assert error <= ATOL_FIELDS, f'поле {name} на t={times[-1]:.4g} с: отклонение {error:.4%}'


if __name__ == '__main__':
	test_wells()
	test_fields()
	print('OK')
