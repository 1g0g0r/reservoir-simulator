# Peaceman1983Interpretation

**Источник.** Peaceman D.W. Interpretation of Well-Block Pressures in Numerical Reservoir Simulation With Nonsquare Grid Blocks and Anisotropic Permeability // Society of Petroleum Engineers Journal. 1983. Vol. 23, No. 03. P. 531–543. DOI: 10.2118/10528-pa.

**Полный текст.** `C:\Users\Игорь\books\neft\Численное моделирование\Модели скважин и околоскважинные течения\Модель Писмана\[Peaceman] Interpretation of Well-Block Pressures in Numerical Reservoir Simulation With Nonsquare Grid Blocks and Anisotropic Permeability.pdf` (13 с.); подтемы: T12; релевантность: высокая.
**Постановка.** Давление в блоке со скважиной для неквадратных блоков и анизотропной проницаемости.
**Метод.** Аналитический вывод эквивалентного радиуса блока r0 = 0.28 [(ky/kx)^½ Δx² + (kx/ky)^½ Δy²]^½ / [(ky/kx)^¼ + (kx/ky)^¼] и численная проверка.
**Результаты.** Формула обобщает r0 ≈ 0.2Δx; связывает забойное давление, давление блока и дебит.
**Ограничения.** Изолированная вертикальная скважина в центре блока, однородная среда.
**Связь с проектом.** Формула коэффициента продуктивности `calc_well_prod` в `paraphin`; при кольматации проницаемость блока скважины текущая — модель скважины сама отражает повреждение ПЗП.
