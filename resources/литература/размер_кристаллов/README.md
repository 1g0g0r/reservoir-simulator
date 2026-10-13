# Размер кристаллов парафина: обоснование диаметра d_p по измерениям

Диаметр кристалла `D` (`paraphin/constants.py`, runtime `D_CRYST`) задает порог блокирования r_pass = d_p/(2γ) и
объем пробки. Прежде он подбирался по керновым опытам (15 мкм по Sutton & Roberts 1, 13.5–16.6 мкм в других
подборах) и выходил выше медианы пор модели (12 мкм). Здесь собраны прямые измерения; сводка и выбранное значение -
`experiments/data/crystal_size.json`. PDF - `fetch_open_access.py размер_кристаллов` (в git не попадают, после
скачивания - в books, см. `ГДЕ_ФАЙЛЫ.md`).

## Измерения в нефти (открытый доступ, скачаны)

Страницы - журнальные (номер в колонтитуле PDF).

| Файл | Источник | Метод и условия | Размер кристаллов | Где в статье |
|---|---|---|---|---|
| `dong2022_rheo_optic_wax_crystals.pdf` | Dong H., Ma R., Zhao J., Xi X., Wang Z. Application of Rheo-optic In Situ Measurement Technology to Study Waxy Crude Oil Rheology // ACS Omega. 2022. 7(21):17948–17962. doi:10.1021/acsomega.2c01251 | Нефть Хулунбуир (Т плавления парафина 60.3 °C), охлаждение от 40–70 °C до 20 °C; реооптическая ячейка MCR 702 и поляризационный микроскоп, разрешение 0.7 мкм, поле 440×330 мкм | Пик распределения по числу ~3 мкм у обеих установок; при охлаждении от 60 °C преобладают кристаллы < 4 мкм, от 70 °C - крупнее 4 мкм; при 40 °C меньше кристаллов < 2 мкм | с. 17950 (разрешение), с. 17951 (рис. 3, пик 3 мкм), с. 17953 (рис. 5) |
| `dong2026_emulsion_gelling_mesoscopic.pdf` | Dong H., Zhang S., Zhao J. Analysis of Gelling Characteristics and Mesoscopic Factors of Water-in-Oil Waxy Crude Oil Emulsions // ACS Omega. 2026. 11(29):43651–43668. doi:10.1021/acsomega.6c02578 | Эмульсии вода в нефти, охлаждение после термообработки при 70 и 80 °C, ~2000 кристаллов на кадр | Средний размер 4.11 мкм (70 °C, 2072 кристалла) и 5.2 мкм (80 °C, 1891). В табл. 5 то же 4.11 стоит в столбце «средняя площадь, мкм²» - если это площадь, эквивалентный диаметр 2.3 мкм | с. 43660 (рис. 18), с. 43663 (табл. 5) |
| `dong2023_wax_crystals_gelation.pdf` | Dong H., Ma R., Zhao J., Guo N., Wang Z. Dynamic Behavior Analysis of Wax Crystals during Crude Oil Gelation // ACS Omega. 2023. 8(34):31085–31099. doi:10.1021/acsomega.3c03021 | Рост кристаллов от WAT 48 °C до 25 °C при сдвиге; микрофотографии (шкала 50 мкм) и модель CFD-DEM | Диаметр частиц модели «по данным опыта» 0.32 мкм при 48 °C и 0.6 мкм при 37–25 °C, разброс 0.7–1.2 от среднего; объемная доля кристаллов 0.02–9.7 % | с. 31089 (табл. 1) |
| `zhao2020_insitu_microscopy_wax_crystals.pdf` | Zhao J., Zhao W., Dong H., Wei L., Liu Y. New Approach for the In Situ Microscopic Observation of Wax Crystals in Waxy Crude Oil during Quiescent and Dynamic Cooling // ACS Omega. 2020. 5(20):11491–11506. doi:10.1021/acsomega.0c00606 | Методика той же установки (поле 440×330 мкм); покой и сдвиг | Числа - только на рисунках (не оцифрованы): сдвиг дробит агрегаты, кристаллы отдельные мельче агрегатов | рис. 9–24 |
| `wang2022_ppd_crystallization_flow.pdf` | Wang C., Chen H., Shi H., Ma K., Ma Q., Gong J. Role of a Nanocomposite Pour Point Depressant on Wax Deposition in Different Flow Patterns from the Perspective of Crystallization Kinetics // ACS Omega. 2022. 7(13):11200–11207. doi:10.1021/acsomega.2c00068 | Нефть на петле, ламинарный и турбулентный режимы, с депрессатором и без | Без депрессатора заметна доля частиц > 32 мкм (агрегаты); депрессатор увеличивает долю < 32 мкм | с. 11204 |
| `dong2026_emulsion_cross_scale.pdf` | Dong H., Zhang J., Wang S., Zhao J. Macro-Meso-Micro Cross-Scale Correlation of Rheological Evolution in Waxy Crude Oil Emulsions under Dynamic Cooling Conditions // ACS Omega. 2026. 11(38):58250–58269. doi:10.1021/acsomega.6c08608 | Эмульсии при охлаждении со сдвигом | «Отдельные кристаллы» 20–85 мкм, агрегаты с каплями до 115 мкм - верхняя граница, это кристаллы с агрегатами в эмульсии | с. 58258 |

## Порог обнаружения и поровое пространство

| Файл | Источник | Что взято |
|---|---|---|
| `sandyga2020_formation_damage_wax.pdf` | Sandyga M.S., Struchkov I.A., Rogachev M.K. Formation damage induced by wax deposition: laboratory investigations and modeling // J. Petrol. Explor. Prod. Technol. 2020. 10:2541–2558. doi:10.1007/s13202-020-00924-2 | Поляризационная микроскопия видит частицы от 0.5 мкм (по Bacon et al. 2009), ближний ИК - кристаллы ~1 мкм на стадии зарождения (с. 3 файла); томография с разрешением 7–8 мкм (с. 5): закупорены поры 20–70 мкм - кристаллы и их агрегаты сравнимы с порами только после роста |
| `li2024_cooling_damage_processes.pdf` | Li X., Zhao L., Fei R., Zhou F., Yuan S. et al. Cooling Damage Characterization and Chemical-Enhanced Oil Recovery in Low-Permeable and High-Waxy Oil Reservoirs // Processes. 2024. 12(2):421. doi:10.3390/pr12020421 | Размеров кристаллов нет. Спектры ЯМР T2 керна (рис. 7, 9) при T2 = A·r (ур. 4), но A (релаксивность и форма пор) не приведен: из спектра получается только ширина распределения радиусов (σ ln r = σ ln T2), масштаб r_m - нет. Вариант керна Li с шириной по ЯМР - только если ступень 25 °C при d_p по измерениям хуже 0.05 (`experiments/li2024.py`) |

## В репозитории и в books

| Файл | Источник | Что взято |
|---|---|---|
| `resources/всякие статьи/hammami1999.pdf` (books: `АСПО в скважинах и трубопроводах/Hammami 1999 ...`) | Hammami A., Raines M.A. Paraffin Deposition From Crude Oils: Comparison of Laboratory Results With Field Data // SPE J. 1999. 4(1):9–18. doi:10.2118/54021-PA | Поляризационный микроскоп видит кристаллы ~1 мкм при WAT (зарождение), метод светопропускания - при росте до ~10 мкм (с. 6, 8–9 файла, рис. 8–9) |
| books: `dissertation/notes/Lu2024Nanofluidic.md` | Lu Z., Wan Y., Xu L. et al. Nanofluidic Study of Multiscale Phase Transitions and Wax Precipitation in Shale Oil Reservoirs // Energies. 2024. 17(10):2415. doi:10.3390/en17102415 | Зерна ~2 мкм на стенках нанофлюидных каналов, затем пластины |

## Найдены, но не скачаны (только описание)

| Источник | Что дает | Доступ |
|---|---|---|
| Bacon M.M., Romero-Zerón L.B., Chong K. Using Cross-Polarized Microscopy to Optimize Wax-Treatment Methods // SPE Annual Technical Conference and Exhibition. 2009 (по ссылке в Sandyga et al. 2020; номер статьи SPE уточнить) | Методика микроскопии, разрешение 0.5 мкм | платный (OnePetro) |
| Experimental Research on the Wax Crystal Particle Size Distribution of Waxy Crude Oil // Pet. Sci. Technol. 2014. 32(17). doi:10.1080/10916466.2013.769573 | Распределение размеров кристаллов нефти (авторы и числа - по аннотации) | платный |
| Quantitative analysis on distribution characteristics of wax crystal particles in waxy crude oil // Oil & Gas Storage and Transportation (Youqi Chuyun). 2015. doi:10.6047/j.issn.1000-8241.2015.10.005 | Распределение размеров, ширина, фрактальная размерность; с депрессатором кристаллы крупнее | на сайте журнала, открытой копии по DOI нет |
| Image Processing and Analysis for Fractal Dimension and Lacunarity Calculation of Waxy Oil Gels Using Polarized Light Micrographs // Cryst. Growth Des. 2024. doi:10.1021/acs.cgd.4c00669 | Диаметр Ферета кристаллов по микрофотографиям | платный |
| Kané M., Djabourov M., Volle J.-L., Lechaire J.-P., Frebourg G. Morphology of paraffin crystals in waxy crude oils cooled in quiescent conditions and under flow // Fuel. 2003. 82(2):127–135. doi:10.1016/S0016-2361(02)00222-3 | Пластинки в покое, мельче при сдвиге (числа сверить по PDF; выходные данные сверить по Crossref) | платный |
| Venkatesan R., Nagarajan N.R., Paso K., Yi Y.-B., Sastry A.M., Fogler H.S. The strength of paraffin gels formed under static and flow conditions // Chem. Eng. Sci. 2005. 60(13):3587–3598. doi:10.1016/j.ces.2005.02.045 | Размер кристаллов от скорости охлаждения и сдвига (сверить) | платный |

## Вывод для модели

Отдельные кристаллы в нефти при охлаждении - единицы микрометров: пик 3 мкм [Dong 2022], средние 2.3–5.2 мкм
[Dong 2026], ~1 мкм при зарождении и ~10 мкм при росте [Hammami 1999]. Десятки микрометров - агрегаты, в том числе с
каплями воды [Dong 2026b, Wang 2022]. Базовое значение d_p = 5 мкм, чувствительность - 3 и 10 мкм. При r_m = 12 мкм и
σ = 0.4 порог r_pass = 6.25 мкм отсекает 5.2 % каналов, 0.8 % объема и 0.07 % проводимости пучка (при 15 мкм - 87,
63 и 34 %).
