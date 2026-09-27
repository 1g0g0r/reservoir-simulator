# Давление и температура начала кристаллизации парафина (WAT)

Как давление и растворенный газ сдвигают WAT (`paraphin/equations/Thermo_wax.py`: поправка Пойнтинга и
разбавление раствора газом; `docs/модель_АСПО.docx`, разд. 3). PDF — `fetch_open_access.py`.

## Использованы в модели

| Файл (после скачивания) | Источник | Что взято в модель | Доступ |
|---|---|---|---|
| `pan1997_pressure_composition_wax.pdf` | Pan H., Firoozabadi A., Fotland P. Pressure and composition effect on wax precipitation: experimental data and model results // SPE Prod. Facil. 1997. 12(4):250–258. doi:10.2118/36740-PA (реферат — OSTI 468155) | Поправка Пойнтинга в multi-solid модели; WAT падает до 15 K от атмосферного давления до P_b из-за растворенного газа и растет выше P_b — V-образная WAT(P) (`n_gas`, `x_saturation`) | платный |
| `hpudsc_presalt_wat_pressure_gas.pdf` | Experimental determination of wax appearance temperature (WAT) in Brazilian presalt petroleum via HPμDSC: effects of pressure and gas composition (Europe PMC: PMC12878762) | Порядок наклона dWAT/dP для дегазированной нефти (~0.2 K/МПа) и эффект газа — сверка | открыт (PMC) |

## Данные калибровки (уже в репозитории)

| Файл | Что взято |
|---|---|
| `../парафиновое/sandyga2020_formation_damage_wax_core.pdf` | Регрессии WAT(P) для 10/30/60 % парафина в керосине: средний наклон 0.188 C/МПа на 0.1–20 МПа → `wax_dv_frac` (`experiments/calibrate.py`, разд. B); сдвиг WAT в поровом пространстве +3.8 C → `wax_pore_shift` (по умолчанию 0) |
| `../парафиновое/sandyga2022_dissertation_spmi.pdf` | Опыты охлаждения при 4 и 5.3 МПа эффективного давления (рис. 2.5.3): WAT растет с давлением — для будущей проверки |
| `../парафиновое/togasheva2026_uzen_cooling_WAT.pdf` | P_b = 9–10 МПа (`P_bubble`); WAT 41–44 C на проточной установке при 10 МПа |
