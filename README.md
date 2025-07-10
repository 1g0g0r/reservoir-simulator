# Paraphin 

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-100%25-brightgreen.svg)](https://github.com/1g0g0r/paraphin)

### Рекомендуемая версия Python: 3.10+

## 📝 Описание
-  🛢 Моделирование двухфазной фильтрации нефти
-  🌡️ Учет неизотермических процессов
- 🔄 Расчет процессов кольматации (оседание парафиновых частиц в капиллярах)
- 🌊 Моделирование суффозии (вынос парафиновых частиц из капилляров)
- 📈 Интерактивное представление результатов (интерактивные plotly графики)

## 📚 Математическая модель
### Основные уравнения:
  - Сохранения массы (воды, нефти, парафина)
  - Движения (закон Дарси)
  - Сохранения полной энергии (вода, нефть, парафин, скелет пористой среды)
  - Уравнения фазовых переходов парафина
  - Изменения радиусов и блокирования капилляров
### Численные методы:
  - Метод конечных объемов (регулярная прямоугольная сетка)
  - Метод IMPES (неявный по давлению, явный по насыщенности)

<hr style="border: 2px black;">

# <u> Результаты </u>
<div style="display: inline-flex; align-items: center; white-space: nowrap; font-size: 18px;">
  <div style="border-top: 5px dashed red; width: 50px;"></div>
  <pre style="font-weight: bold; margin: 0; padding: 0;"> — Массовая доля растворенного парафина 0% </pre>
</div>
<br>
<div style="display: inline-flex; align-items: center; white-space: nowrap; font-size: 18px;">
  <div style="border-top: 5px solid black; width: 50px;"></div>
  <pre style="font-weight: bold; margin: 0; padding: 0;"> — Массовая доля растворенного парафина 5% </pre>
</div>

## Поле насыщенности
![Водонасыщенность](resources/gif/Saturation.gif)

## Поле температуры
![Температура](resources\gif\Temperature.gif)

## Массовая доля выпавшего парафина
![Массовая доля выпавшего парафина](resources\gif\Wps dep.gif)

## Функция пор по размерам
![Функция пор по размерам](resources\gif\fi.gif)

## Множитель пористости
![Множитель пористости](resources\gif\m mult.gif)

## Множитель проницаемости
![Множитель проницаемости](resources\gif\k mult.gif)
