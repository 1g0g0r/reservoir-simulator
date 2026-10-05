"""Термодинамика на уравнении состояния Пенга-Робинсона: парафин (multi-solid), асфальтены (Nghiem), газ (flash).

Модули считаются numpy/scipy один раз при импорте `tables` и только при включенных флагах `wax_eos` или
`asph_nghiem`: в njit-ядра расчета попадают таблицы по (T, P) и интерполяция `tables.eos_interp`.
    eos              - PR EOS, тест устойчивости (Michelsen), flash пар-жидкость, давление насыщения;
    solids           - multi-solid (Lira-Galeana), идеальный твердый раствор (Won), асфальтены (Nghiem);
    characterization - свойства псевдокомпонентов (Riazi, Riazi-Daubert, Kesler-Lee), гамма-распределение;
    tables           - флюид модели из констант, калибровки, таблицы; `python -m paraphin.thermo.tables` -
                       сравнение с прежними моделями.
Пакет нарочно ничего не импортирует: `oil_composition` берет отсюда только `characterization`.
"""
