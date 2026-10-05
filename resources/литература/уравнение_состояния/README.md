# Уравнение состояния, flash и твердые фазы на нем: литература `paraphin/thermo`

Статьи, на которых построен вариант термодинамики на уравнении Пенга–Робинсона (флаги `wax_eos`, `asph_nghiem`;
`paraphin/thermo`, описание — `docs/модель_АСПО.md`, разд. 2.2, 3.4, 4.5) и сравнение с опытами
(`experiments/уравнение_состояния/сравнение_EOS.md`). Обзор с общей схемой — `твт_статья/full_review.pdf`,
разд. 1.4–1.6, 5.3.

PDF кладет `python resources/литература/fetch_open_access.py` (открытые копии: издатель, Unpaywall или OpenAlex);
после скачивания их перекладывают в библиотеку books (`../ГДЕ_ФАЙЛЫ.md`).

## Использованы в модели

| Файл (после скачивания) | Источник | Что взято | Доступ |
|---|---|---|---|
| `peng_robinson1976_eos.pdf` | Peng D.-Y., Robinson D.B. A new two-constant equation of state // Ind. Eng. Chem. Fundam. 1976. 15(1):59–64. doi:10.1021/i160057a011 | Уравнение состояния, фугитивности (`thermo/eos.py`) | платный |
| — | Robinson D.B., Peng D.-Y. The characterization of the heptanes and heavier fractions for the GPA Peng–Robinson programs // GPA RR-28. 1978 | m(omega) при omega > 0.491 | нет в сети |
| `michelsen1982_stability.pdf` | Michelsen M.L. The isothermal flash problem. Part I. Stability // Fluid Phase Equilib. 1982. 9:1–19. doi:10.1016/0378-3812(82)85001-2 | Тест устойчивости по касательной плоскости (`stability_test`) | платный |
| `michelsen1982_phase_split.pdf` | Michelsen M.L. The isothermal flash problem. Part II. Phase-split calculation // Fluid Phase Equilib. 1982. 9:21–40. doi:10.1016/0378-3812(82)85002-4 | Последовательные подстановки во flash (`flash`) | платный |
| `rachford_rice1952_flash.pdf` | Rachford H.H., Rice J.D. // J. Pet. Technol. 1952. 4(10). doi:10.2118/952327-G | Уравнение Рачфорда–Райса (`rachford_rice`) | открыт на OnePetro, но сайт отдает 403 скриптам — скачать вручную |
| `liragaleana1996_multisolid_wax.pdf` (папка `состав_нефти`) | Lira-Galeana C., Firoozabadi A., Prausnitz J.M. // AIChE J. 1996. 42:239–248 | Multi-solid с жидкостью по уравнению состояния (`MultiSolidWax`) | платный |
| `pan1997_pressure_composition_wax.pdf` (папка `давление_WAT`) | Pan H., Firoozabadi A., Fotland P. // SPE Prod. Facil. 1997. 12:250–258 | Поправка Пойнтинга на скачок объема в твердой фазе | платный |
| `riazi_alsahhaf1996_heavy_fractions.pdf` | Riazi M.R., Al-Sahhaf T.A. Physical properties of heavy petroleum fractions and crude oils // Fluid Phase Equilib. 1996. 117:217–224. doi:10.1016/0378-3812(95)02956-7 | Корреляции гомологического ряда н-алканов (T_b, SG от M; в книге Riazi, 2005, разд. 2.3.3) | платный |
| `riazi1987_petroleum_fractions.pdf` (папка `состав_нефти`) | Riazi M.R., Daubert T.E. // Ind. Eng. Chem. Res. 1987. 26:755–759 | T_b растворителя по M и SG | платный |
| — | Riazi M.R., Daubert T.E. Simplify property predictions // Hydrocarbon Process. 1980. 59(3):115–116 | T_c, P_c по T_b и SG | без DOI |
| — | Kesler M.G., Lee B.I. Improve prediction of enthalpy of fractions // Hydrocarbon Process. 1976. 55(3):153–158 | Ацентрический фактор, обе ветви | без DOI |
| `nghiem1993_asphaltene_solid_model.pdf` | Nghiem L.X. et al. Efficient modelling of asphaltene precipitation // SPE 26642. 1993. doi:10.2118/26642-MS | Модель твердой фазы асфальтенов (`NghiemAsphaltene`) | платный |
| `qin2000_asphaltene_reservoir_simulation.pdf` | Qin X. et al. Modeling asphaltene precipitation in reservoir simulation // Ind. Eng. Chem. Res. 2000. 39:2644–2654. doi:10.1021/ie990781g | Та же модель в композиционном симуляторе | платный |
| books: `Асфальтены/Tabzar 2018 - ...pdf` | Tabzar A. et al. // Oil Gas Sci. Technol. 2018. 73:51. doi:10.2516/ogst/2018039 | kij асфальтены–легкие = 0.4, V_s = 0.8 л/моль (табл. 1, разд. 2.1.1) | открыт |

## Данные и модели для сравнения

| Файл | Источник | Что взято |
|---|---|---|
| books: `Термодинамика и фазовое равновесие/Dasilva 2017 - ...pdf` | da Silva V.M. et al. Paraffin solubility and calorimetric data calculation using Peng-Robinson EoS and modified UNIQUAC models // J. Pet. Sci. Eng. 2017. 156:945–957. doi:10.1016/j.petrol.2017.06.064 | Составы 5 синтетических смесей, WDT опыта и моделей, кривые растворимости — `experiments/data/dasilva2017_sle.json` |
| books: `Термодинамика и фазовое равновесие/broadhurst1962_nalkanes_melting_nbs.pdf` | Broadhurst M.G. // J. Res. NBS. 1962. 66A:241–249 | Теплоты плавления и твердо-твердого перехода н-алканов (табл. 3) — оценка погрешности корреляции Won |
| `coutinho2006_predictive_uniquac_wax.pdf` | Coutinho J.A.P., Mirante F., Pauly J. // Fluid Phase Equilib. 2006. 247:8–17. doi:10.1016/j.fluid.2006.06.002 | Предсказательный UNIQUAC для твердого раствора — направление развития (`docs/модель_АСПО.md`, разд. 12, п. 10) |
| `burke1990_asphaltene_precipitation_data.pdf` | Burke N.E., Hobbs R.E., Kashou S.F. // J. Pet. Technol. 1990. 42:1440–1446. doi:10.2118/18273-PA | Опыты: выпадение асфальтенов от давления с максимумом у P_b |
| books: `Термодинамика и фазовое равновесие/Coutinho 2002 - ...pdf` (скачан с сайта автора) | Coutinho J.A.P., Edmonds B., Moorwood T., Szczepanski R., Zhang X. Reliable wax predictions for flow assurance // SPE 78324. 2002 | Приложение: T_m, T_tr и теплоты плавления и перехода н-алканов (A1)-(A6; подписи (A4) и (A6) перепутаны - сверено с Broadhurst), параметры предсказательного UNIQUAC - `characterization.coutinho_nalkane`, `solids.UniquacSolidWax` |
| books: `Асфальтены/Tabzar 2018 - ...pdf` | Tabzar et al. (2018), Table 3, 4 (данные Jamaluddin et al.) | Состав живой нефти (16 компонентов) и выпавшие асфальтены при 212 F - `experiments/data/tabzar2018_asphaltene.json`, калибровка Нгхайема по кривой |

## Найдены, но не скачаны (нет открытой копии)

Peng & Robinson (1976), Michelsen (1982, обе части), Nghiem et al. (1993), Qin et al. (2000), Riazi & Al-Sahhaf
(1996), Burke et al. (1990), Ji et al. (2004, doi:10.1016/j.fluid.2003.05.011), Coutinho & Daridon (2001,
doi:10.1021/ef010072r), Coutinho et al. (2006), Pedersen, Skovborg & Rønningsen (1991, скачок теплоемкости;
doi:10.1021/ef00030a022), Lee, Gonzalez & Eakin (1966, вязкость газа; doi:10.2118/1340-PA). Rachford & Rice (1952)
открыт, но OnePetro не отдает файл скриптам.
