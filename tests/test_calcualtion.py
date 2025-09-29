"""Сравнение текущей версии солвера с потенциально достоверной."""
import numpy as np
from pathlib import Path
from pickle import load

from start import solve
from paraphin.constants import init_Wp
from paraphin.utils import read_solution_data


# TODO запиклить класс решения чтобы хранить всю информацию (данные о сетке и времени расчета)
# Параллельно запускать расчеты wp=5 wp=0 (проблема: нужно сохранять данные в разные папки для избежания затирания данных)

def test():
	"""Сравнение текущей версии солвера с потенциально достоверной."""
	test_data_path = Path.cwd() / "test_data"
	with open(test_data_path / f'Wp={init_Wp}_processed_data.pkl', 'rb') as file:
		_n_test_data, test_data = load(file)

	# solve()
	_n_times, calculation_data = read_solution_data(f'Wp={init_Wp}_processed_data.pkl')
	arr = np.array([0.0])
	n = min(_n_test_data, _n_times)

	try:
		# Сравнение полей данных
		for name in test_data.keys() :
			if name not in calculation_data.keys() or name in ['Wells', 'Other params', 'Time']:
				continue

			if not np.all(np.isclose(test_data[name][:n], calculation_data[name][:n])):
				arr = test_data[name][:n] - calculation_data[name][:n]
				assert False

		# Сравнение показателей работы скважины
		for name in test_data['Wells'].keys():
			if name not in calculation_data['Wells'].keys():
				continue

			if not np.all(np.isclose(test_data['Wells'][name][:n], calculation_data['Wells'][name][:n])):
				arr = test_data['Wells'][name][:n] - calculation_data['Wells'][name][:n]
				assert False
	except:
		std = np.std(arr)
		# rmse = np.sqrt(np.mean(arr ** 2))
		mape = np.mean(np.abs(arr / test_data[name][:n])) * 100

		print()
		assert False, print(f'Поломалось поле данных {name}: std={std}, mape {mape}')
