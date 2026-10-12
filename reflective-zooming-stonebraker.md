# Перекалибровка закупорки капилляров: d_p из независимых измерений

## Context
Сейчас диаметр кристалла d_p подбирается по керновым опытам:
- `constants.D = 15 мкм` (`core_flood.py`, Sutton & Roberts 1);
- `D_CRYST = 16.6 мкм` в общем наборе `unified.py`;
- `D_CRYST = 13.5 мкм` по ступени 25 °C Li (`params.json: li2024_cold`), из нее же полевая кинетика `run_thermal.field_kin`.

Следствие: r_pass = d_p/(2γ) = 18.75 мкм выше медианы пор 12 мкм. Под закупорку попадают 87% каналов, 63% объема, 34% проводимости.
- Функция пор fi проваливается в ноль посередине.
- k/k0 падает до ~0.57 уже за 0.1 PV. В опыте Sutton & Roberts 1 спад плавнее: 0.70 при 0.25 PV, 0.60 при 0.5 PV.

Измерения в нефти дают кристаллы 1–10 мкм (обзор ниже). Цель:
- зафиксировать d_p по измерениям;
- подобрать остальное (snowball, сеть пор, вынос, кинетика) по Sutton & Roberts 1, 2 и Li et al. 2024;
- обновить `constants.py`, `params.json`, `experiments/results`, числа и текст `article_crystallization`.

Сейчас ничего не считать. План рассчитан на запуск на мощной машине.

Решения пользователя:
- освободить snowball и сеть пор;
- пересчитать опыты и `constants`;
- сетка поля 75×75 (`constants`: Nx, Ny = 75, 75 уже стоит). Полевой этап — Этап 6.

---

## Этап 0. Обзор размеров кристаллов: скачать и записать

### Найденные источники

Открытый доступ (OA) — скачать. В книги (`books`) не класть до решения автора.

| # | Источник | Что дает | Доступ |
|---|---|---|---|
| 1 | Dong H., Ma R., Zhao J., Xi X., Wang Z., Wang Z. Application of Rheo-optic In Situ Measurement Technology to Study Waxy Crude Oil Rheology // ACS Omega. 2022. V. 7, No 21. P. 17948–17962. doi:10.1021/acsomega.2c01251 (PMC9161395) | Нефть, две оптические установки. Пик распределения кристаллов ~3 мкм. Доля мелких (<2–4 мкм) зависит от начальной температуры охлаждения | OA |
| 2 | Zhao J., Zhao W., Dong H., Wei L., Liu Y. New Approach for the In Situ Microscopic Observation of Wax Crystals in Waxy Crude Oil during Quiescent and Dynamic Cooling // ACS Omega. 2020. V. 5, No 20. P. 11491–11506. doi:10.1021/acsomega.0c00606 (PMC7254814) | Методика. Кристаллы при покое и сдвиге; поле зрения 440×330 мкм. Числа — из рисунков, оцифровать | OA |
| 3 | Dong H., Ma R., Zhao J., Guo N., Wang Z. Dynamic Behavior Analysis of Wax Crystals during Crude Oil Gelation // ACS Omega. 2023. V. 8, No 34. P. 31085–31099. doi:10.1021/acsomega.3c03021 (PMC10468901) | Рост кристаллов от WAT 48 °C до 25 °C. Размеры — из рисунков | OA |
| 4 | Dong H., Zhang S., Zhao J. Analysis of Gelling Characteristics and Mesoscopic Factors of Water-in-Oil Waxy Crude Oil Emulsions // ACS Omega. 2026. V. 11, No 29. P. 43651–43668. doi:10.1021/acsomega.6c02578 (PMC13425321) | Средний размер 4.11 и 5.2 мкм (~2000 кристаллов); гамма-распределение; порог микроскопа 1 мкм | OA |
| 5 | Dong H., Zhang J., Wang S., Zhao J. Macro-Meso-Micro Cross-Scale Correlation of Rheological Evolution in Waxy Crude Oil Emulsions under Dynamic Cooling Conditions // ACS Omega. 2026. V. 11, No 38. P. 58250–58269. doi:10.1021/acsomega.6c08608 (PMC13629359) | Эмульсии, охлаждение при сдвиге: отдельные кристаллы 20–85 мкм. Верхняя граница, агрегаты | OA |
| 6 | Wang C., Chen H., Shi H., Ma K., Ma Q., Gong J. Role of a Nanocomposite Pour Point Depressant on Wax Deposition in Different Flow Patterns from the Perspective of Crystallization Kinetics // ACS Omega. 2022. V. 7, No 13. P. 11200–11207. doi:10.1021/acsomega.2c00068 (PMC8992259) | Ламинарный поток: заметная доля кристаллов >32 мкм без депрессатора | OA |
| 7 | Hammami A., Raines M.A. Paraffin Deposition From Crude Oils: Comparison of Laboratory Results With Field Data // SPE J. 1999. V. 4, No 1. P. 9–18. doi:10.2118/54021-PA | ~1 мкм при WAT, ~10 мкм на стадии роста | есть: `resources/всякие статьи/hammami1999.pdf` |
| 8 | Lu Z., Wan Y., Xu L. et al. Nanofluidic Study of Multiscale Phase Transitions and Wax Precipitation in Shale Oil Reservoirs // Energies. 2024. V. 17, No 10. P. 2415. doi:10.3390/en17102415 | Зерна ~2 мкм на стенках каналов, затем пластины | есть в books (`dissertation/notes/Lu2024Nanofluidic.md`) |
| 9 | Sandyga M.S., Struchkov I.A., Rogachev M.K. Formation damage induced by wax deposition: laboratory investigations and modeling // J. Petrol. Explor. Prod. Technol. 2020. V. 10. P. 2541–2558. doi:10.1007/s13202-020-00924-2 | Закупориваются поры 20–70 мкм (томография). Порог микроскопии 0.5–1 мкм | OA, уже в проекте |
| 10 | Bacon M.M., Romero-Zerón L.B., Chong K. Using Cross-Polarized Microscopy to Optimize Wax-Treatment Methods // J. Phys.: Conf. Ser. 2009 / Proc. ICOMS (уточнить по ссылке в [9]) | Методика микроскопии, разрешение 0.5 мкм | уточнить, OA вероятно |
| 11 | Li X., Zhao L., Fei R., Zhou F., Yuan S. et al. Cooling Damage Characterization and Chemical-Enhanced Oil Recovery in Low-Permeable and High-Waxy Oil Reservoirs // Processes. 2024. V. 12, No 2. P. 421. doi:10.3390/pr12020421 | Размеров кристаллов нет. Есть ЯМР T2 — распределение радиусов пор керна Li: использовать для fi_0 керна (см. Этап 3) | OA (`mdpi-res.com/.../processes-12-00421.pdf`) |
| 12 | Experimental Research on the Wax Crystal Particle Size Distribution of Waxy Crude Oil // Pet. Sci. Technol. 2014. V. 32, No 17. doi:10.1080/10916466.2013.769573 | Распределение размеров кристаллов нефти. Авторы и числа — по аннотации | платная: только описание |
| 13 | Quantitative analysis on distribution characteristics of wax crystal particles in waxy crude oil // Oil & Gas Storage and Transportation (Youqi Chuyun). 2015. doi:10.6047/j.issn.1000-8241.2015.10.005 | Распределение размеров, ширина, фрактальная размерность; с депрессатором кристаллы крупнее | сайт журнала, проверить PDF |
| 14 | Image Processing and Analysis for Fractal Dimension and Lacunarity Calculation of Waxy Oil Gels Using Polarized Light Micrographs // Cryst. Growth Des. 2024. doi:10.1021/acs.cgd.4c00669 | Диаметр Ферета кристаллов из микрофотографий | платная: только описание |
| 15 | Kané M., Djabourov M., Volle J.-L., Lechaire J.-P., Frebourg G. Morphology of paraffin crystals in waxy crude oils cooled in quiescent conditions and under flow // Fuel. 2003. V. 82, No 2. P. 127–135. doi:10.1016/S0016-2361(02)00222-3 | Классика: пластинки в покое, мельче при сдвиге (числа сверить по PDF) | платная: только описание |
| 16 | Venkatesan R., Nagarajan N.R., Paso K., Yi Y.-B., Sastry A.M., Fogler H.S. The strength of paraffin gels formed under static and flow conditions // Chem. Eng. Sci. 2005. V. 60, No 13. P. 3587–3598. doi:10.1016/j.ces.2005.02.045 | Размер кристаллов от скорости охлаждения и сдвига (сверить) | платная: только описание |

Источники 15 и 16 указаны по памяти. Перед вставкой в статью сверить выходные данные по DOI через Crossref.

### Действия этапа 0
1. Дописать в `resources/литература/fetch_open_access.py` (`ARTICLES`) папку `размер_кристаллов`: 1–6, 9, 11, 13 с открытыми ссылками.
   - Europe PMC: `https://europepmc.org/articles/PMC.../pdf` или render.
   - MDPI: `mdpi-res.com`.
   - Скачивать через `curl_cffi` (`impersonate='chrome'`, см. память).
   - Платные — только запись в README.
2. Создать `resources/литература/размер_кристаллов/README.md`: таблица выше с конкретными числами из PDF (страница, рисунок).
3. Строки в `ГДЕ_ФАЙЛЫ.md` — «ожидает переноса в books».
4. Создать `experiments/data/crystal_size.json` по образцу других `data/*.json`: `source`, `method`, `conditions`, `size_um`: {`mean`, `peak`, `range`}, `basis` (число или объем; длина или эквивалентный диаметр).
5. Выбор d_p по сводке:
   - базовый вариант d_p = 5 мкм (средние 3–5 мкм в нефти [1, 4]; ~10 мкм — рост [7]);
   - чувствительность 3 и 10 мкм;
   - записать в `crystal_size.json: chosen` с обоснованием.

   r_pass = 6.25 мкм: под закупорку ~3.5% каналов. Ранний спад должны давать сужение, snowball или сеть.

## Этап 1. constants и модель по умолчанию (без кинетики)

- `paraphin/constants.py`: `D = 5e-6` с источником (`crystal_size.json`). `Lk` пометить «калибровка: core_flood.py, d_p фиксирован».
- `experiments/исходная_модель/core_flood.py`:
  - `D_GRID` → один фиксированный `D` из `constants`;
  - подбор по сетке `LK_GRID = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)` × `diff_mult` (0.1, 0.3, 1, 3, 10).
  - Snowball и сети в модели по умолчанию нет (`calc_velocities_h`): если согласие плохое — честно записать, механизмы не добавлять.
- `paraphin/geometry.py:24-28`: исправить устаревший комментарий «ниже r_pass лежит ~1%» на долю при новом D (оценка по `fi_0`).
- `docs/параметры_модели.md` разд. 7, `docs/WAX_PRECIPITATION_FINDINGS.md`: новый раздел «d_p по измерениям» — прежний подбор 15 мкм, новый, СКО до и после.
- CLAUDE.md: фраза «Калибрует D (d_p) и Lk» → «D — по измерениям, Lk — калибровка».

## Этап 2. Sutton & Roberts (`experiments/sutton_roberts.py`)

- Константа `D_CRYST_MEAS` из `data/crystal_size.json` (одна функция чтения в `common.py`, переиспользовать везде).
- `LSQ_FAMILIES`: убрать `D_CRYST` из `names`, подставлять `D_CRYST_MEAS` в `lsq_kin`.
  - `rate_bundle`: `(DIFF_MULT, SNOW_A)`. Snowball уже в `KERNEL`, сейчас гасится `SNOW_A = 0` в `lsq_kin`.
  - `rate_entrainment`: `(DIFF_MULT, ENT_TAU, ENT_RATE, SNOW_A)`.
  - `thermal_bundle`: `(DIFF_MULT, SNOW_A)`.
  - `thermal_network`: `(DIFF_MULT, NET_GAMMA)` + вариант `thermal_network_snow` с `snowball='True'` и `SNOW_A`.
  - Пределы: lg SNOW_A от −1 до 3 (log=True); NET_GAMMA 0.2–0.9 (уже есть).
- `VISC_X0` / `visc_kin`: убрать lg d_p, добавить SNOW_A.
- Сетки `ENT_GRID` / `FILT_GRID` (`others`) не трогать: d_p там не участвует.
- Отчетная метрика: k/k0 при 0.25 и 0.5 PV против 0.70/0.60 и 0.72/0.63 — отдельной строкой в `summary`: это ранний участок, о котором говорил пользователь.

## Этап 3. Li et al. 2024 (`experiments/li2024.py`)

- `COLD_X0 / LO / HI / STEP`: убрать lg D_CRYST. Оставить `(lg DIFF_MULT, lg K_CRYST)`, добавить lg SNOW_A. `cold_kin` — D_CRYST = измеренный. Флаг `snowball='True'` в `cold_job`.
- `NET_D_CRYST`: заменить на `(D_meas, 3e-6, 10e-6)`. Сеть — с подбором `NET_GAMMA` (новый `net_fit`, тот же `lsq`).
- fi_0 керна Li: сейчас Berea (r_m 12 мкм), а уменьшенные поры давали закупорку к 0.1 PV. С d_p = 5 мкм проверить r_m из ЯМР T2 (рис. в [11]): оцифровать распределение радиусов → `data/li2024.json: pore_radius_nmr`, передать `r_m`, `sigma_r`, `r_max` в `constants(flags)`.
  - Только как вариант чувствительности, если согласие ступени 25 °C хуже 0.05.
- Ступени 90–45 °C (удержание) не трогать: d_p в них не влияет.

## Этап 4. Общий набор (`experiments/unified.py`)

- `NAMES / X0 / LO / HI / STEP`: убрать `lg D_CRYST`, добавить `lg SNOW_A`. В `jobs_of` — D_CRYST = измеренный для sr, li, sd.
- `FLAGS = {'entrainment': 'True', 'snowball': 'True'}`. Убрать `SNOW_A=0.0`.
- Второй вариант набора — сеть пор: `FLAGS_NET = dict(FLAGS, pore_network='True')`, параметр `NET_GAMMA`.
  - Запуск `--net`.
  - В `results/unified.json` ключ `net`. Лучший по сумме невязок → `params.json: unified.best`.
- LOO-прогнозы (`loo`) и `hypothesis_checks` — с новым x. Индексы fixed пересчитать по новым NAMES.
- `sandyga2020.py`: проверить, что собственный подбор берет `D_CRYST` из измерения, а не из `constants.D` по умолчанию. Сейчас `kin_case` его не задает → получит новый `constants.D`, это и нужно; явно записать.

## Этап 5. Отчеты и статья

- `python experiments/run_all.py` (полный, с подбором) → `results/*.json`, `tables.json`, `params.json`. `python docs/build_docx.py сравнение_с_опытами`.
- `experiments/сравнение_с_опытами.md`: d_p — из измерений, таблица чувствительности 3/5/10 мкм (СКО SR1, SR2, Li 25 °C).
- `твт_статья_АСПО/article_crystallization.md`:
  - разд. ~строки 160–166: d_p — не подбор, а измерение [новые ссылки 1, 4, 7];
  - табл. 6: строка d_p «по измерениям, не подбирается», добавить строку SNOW_A (и NET_GAMMA, если выбрана сеть);
  - строка 260: подбор ступени 25 °C без d_p;
  - число параметров «восемь вместо пятнадцати» пересчитать;
  - библиография — новые источники (нумерация — `make_docx.py` сама).
- `python твт_статья_АСПО/make_article.py` → `results/numbers.json`; `python docs/build_docx.py статья_кристаллизация`. Проверить, что плейсхолдеры `{{...}}` все заполнены.

## Этап 6. Поле на 75×75

По ответу «пересчитай все на 75 75». Подтвердить перед запуском.
- `run_thermal.py`: базовая сетка варианта → `'Nx, Ny': '75, 75'` во всех вариантах без явной сетки; `uniquac_g25` оставить.
- `field_kin()` берет новый `li2024_cold`. Варианты статьи о кристаллизации: `base`, `equil`, `nogel`, `nopress`, `ret`, `base_g25`, `uniquac_g25`. Варианты тепловой статьи — по желанию.
- Текст статей: «сетка 50 × 50» → 75 × 75 (`article_crystallization`, `article_thermal_tvt` строки 142–144), сеточный абзац.

---

## Запуск на мощной машине

```bash
uv pip install --system -r requirements.txt pytest
export GOMP_SPINCOUNT=0                 # несколько расчетов на общих ядрах (CLAUDE.md)
export EXPERIMENTS_WORKERS=$(nproc)     # керны: по процессу на ядро, по потоку numba на процесс
python resources/литература/fetch_open_access.py                 # этап 0
python experiments/исходная_модель/core_flood.py --fit            # этап 1: Lk, diff_mult при D из измерений
python experiments/run_all.py sutton_roberts li2024 sandyga2020   # этапы 2-3
python experiments/unified.py && python experiments/unified.py --net   # этап 4 (часы)
python experiments/grid_check.py                                  # сходимость кернов при новых параметрах
python experiments/run_all.py --plot && python docs/build_docx.py сравнение_с_опытами
python твт_статья_АСПО/make_article.py && python docs/build_docx.py статья_кристаллизация
THERMAL_WORKERS=$(( $(nproc) / 4 )) python твт_статья_АСПО/run_thermal.py base equil nogel nopress ret base_g25 uniquac_g25  # этап 6
```

Оценки (на 4 ядрах нынешней машины):
- `run_all --quick` ~20 мин; полный `run_all` и `unified` — часы;
- поле 75×75 ~70 мин на вариант (20 мин × 3.4), UNIQUAC — дольше.

Кэш `experiments/results/cache` не в git: на новой машине все считается с нуля.

## Тесты и регрессия
- Смена `constants.D` меняет модель по умолчанию:
  - `tests/test_regression_flags_off.py` (побитовый эталон) и `test_reduces_to_legacy` (legacy при D = 15 мкм) разойдутся.
  - Решение: `tests/regression_baseline.py` считать в копии пакета с явной подстановкой `{'D': '15e-6', 'Lk': '3e-4'}` (`_patched_copy.make_copy`). Регрессия проверяет код, а не калибровку; эталоны остаются годными.
- `tests/test_patch_constants.py` — наборы подстановок (новые имена не вводятся; проверить).
- `python -m pytest tests -q --ignore=tests/test_calculation.py`. `test_composition_flags_on` (балансы с snowball, сетью).
- `python bench.py` до и после: ядра не правятся, замедления быть не должно.

## Проверка результата
- fi(r) в точке FI_PROBE после `python start.py` (75×75): нет провала до нуля в середине, сдвиг влево и спуск вниз; m/m0 и k/k0 соизмеримее, чем 0.935/0.36.
- SR1: k/k0(0.25 PV) ≈ 0.70 ± 0.05, СКО ≤ 0.05 (цель отчета). SR2 и Li 25 °C — СКО, сравнить с нынешними (0.055/0.066/0.141 общий набор).
- Если согласие заметно хуже нынешнего: в отчете и статье честно — «d_p по измерениям не объясняет ранний провал; нужен механизм X». Не возвращать d_p в подбор без решения автора.
