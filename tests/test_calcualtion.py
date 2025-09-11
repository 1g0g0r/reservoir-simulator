import numpy as np
from pathlib import Path
from pickle import load

from start import solve
from paraphin.constants import init_Wp
from paraphin.utils import read_solution_data


# Здесь хранить данные о сетке и времени расчета (для попыток ускорения)
# Добавить метод обновления файлов в test_data (просто замена файла в папке)
# Параллельно запускать расчеты wp=5 wp=0 (проблема: нужно сохранять данные в разные папки для избежания затирания данных)

def test():
	"""Сравнение текущей версии солвера с потенциально достоверной."""
	test_data_path = Path.cwd() / "test_data"
	with open(test_data_path / f'Wp={init_Wp}_processed_data.pkl', 'rb') as file:
		_n_test_data, test_data = load(file)

	solve()
	_n_times, calculation_data = read_solution_data(f'Wp={init_Wp}_processed_data.pkl')
	arr = np.array([0.0])

	try:
		# Сравнение полей данных
		for name in test_data.keys() :
			if name not in calculation_data.keys() or name in ['Wells', 'Other params']:
				continue

			if not np.all(np.isclose(test_data[name], calculation_data[name][:_n_test_data])):
				arr = test_data[name]-calculation_data[name][:_n_test_data]
				raise False

		# Сравнение показателей работы скважины
		for name in test_data['Wells'].keys():
			if name not in calculation_data['Wells'].keys():
				continue

			if not np.all(np.isclose(test_data['Wells'][name], calculation_data['Wells'][name][:_n_test_data])):
				arr = test_data['Wells'][name]-calculation_data['Wells'][name][:_n_test_data]
				raise False
	except:
		std = np.std(arr)
		rmse = np.sqrt(np.mean(arr ** 2))
		raise print(f'Поломалось поле данных {name}. std={std}, rmse {rmse}')
