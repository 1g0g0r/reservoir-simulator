# Состав нефти и равновесие твердое–жидкое: литература детального состава

Статьи, на которых построены характеризация нефти (SARA + SCN) и равновесие групп парафинов и восков
(`paraphin/oil_composition.py`, `paraphin/equations/Thermo_wax.py`, описание — `docs/модель_АСПО.docx`,
разд. 2–3). Обзор, из которого взята общая схема, — `твт_статья/full_review.pdf`, разд. 1.2–1.7.

PDF сюда кладет `python resources/литература/fetch_open_access.py --email ...`: при сборке модели внешняя
сеть среды была закрыта. Скрипт берет только открытые копии (издатель, NIST, arXiv, OSTI, Europe PMC,
Unpaywall); платные остаются в таблице без файла.

## Использованы в модели

| Файл (после скачивания) | Источник | Что взято в модель | Доступ |
|---|---|---|---|
| `broadhurst1962_nalkanes_melting_nbs.pdf` | Broadhurst M.G. An analysis of the solid phase behavior of the normal paraffins // J. Res. NBS. 1962. 66A(3):241–249. doi:10.6028/jres.066A.024 | Температуры и теплоты плавления н-алканов — сверка корреляции Won (`tests/test_thermo_wax.py::test_won_correlation`) | открыт (NIST) |
| `won1986_wax_solid_solution.pdf` | Won K.W. Thermodynamics for solid solution–liquid–vapor equilibria: wax phase formation from heavy hydrocarbon mixtures // Fluid Phase Equilib. 1986. 30:265–279. doi:10.1016/0378-3812(86)80061-9 | Tm = 374.5 + 0.02617M − 20172/M, dH = 0.1426·M·Tm — свойства групп (`won_tm`, `won_dh`) и калориметрическая теплота в уравнении энергии | платный |
| `liragaleana1996_multisolid_wax.pdf` | Lira-Galeana C., Firoozabadi A., Prausnitz J.M. Thermodynamics of wax precipitation in petroleum mixtures // AIChE J. 1996. 42(1):239–248. doi:10.1002/aic.690420120 | Multi-solid: каждая группа выпадает своей чистой твердой фазой (`sle_split`) | платный |
| `pedersen1991_north_sea_wax_modeling.pdf` | Pedersen K.S., Skovborg P., Rønningsen H.P. Wax precipitation from North Sea crude oils. 4. Thermodynamic modeling // Energy Fuels. 1991. 5(6):924–932. doi:10.1021/ef00030a021 | Экспоненциальное распределение плюс-фракции z_n ~ exp(−s·n) (`scn_distribution`) | платный |
| `pedersen1995_cloud_point.pdf` | Pedersen K.S. Prediction of cloud point temperatures and amount of wax precipitation // SPE Prod. Facil. 1995. 10(1):46–49. doi:10.2118/27629-PA | Точка помутнения как первая насыщенная группа (`wat_cell`) | платный |
| `hansen1988_wax_thermodynamic_model.pdf` | Hansen J.H., Fredenslund Aa., Pedersen K.S., Rønningsen H.P. A thermodynamic model for predicting wax formation in crude oils // AIChE J. 1988. 34(12):1937–1942. doi:10.1002/aic.690341202 | Неидеальность реальной нефти — обоснование эффективных параметров растворимости групп | платный |
| `coutinho1996_wilson_nalkanes.pdf` | Coutinho J.A.P., Stenby E.H. Predictive local composition models for solid/liquid equilibrium in n-alkane systems // Ind. Eng. Chem. Res. 1996. 35:918–925. doi:10.1021/ie950447u | Альтернатива (твердый раствор) — в разделе ограничений docx | платный |
| `katz1978_scn_plus_fraction.pdf` | Katz D.L., Firoozabadi A. Predicting phase behavior of condensate/crude-oil systems using methane interaction coefficients // JPT. 1978. 30:1649–1655. doi:10.2118/6721-PA | Разбиение плюс-фракции на SCN-группы | платный |
| `whitson1983_plus_fractions.pdf` | Whitson C.H. Characterizing hydrocarbon plus fractions // SPEJ. 1983. 23(4):683–694. doi:10.2118/12233-PA | Укрупнение SCN в псевдокомпоненты (`lump_groups`) | платный |
| `riazi1987_petroleum_fractions.pdf` | Riazi M.R., Daubert T.E. Characterization parameters for petroleum fractions // Ind. Eng. Chem. Res. 1987. 26:755–759. doi:10.1021/ie00064a023 | Свойства фракций по M и плотности — в описании характеризации | платный |

## Данные калибровки (уже в репозитории)

| Файл | Что взято |
|---|---|
| `../парафиновое/li2024_zhetybai_cooling_damage.pdf` | Кривая выпадения и WAT по ДСК нефти Жетыбая, SARA (смолы 5.52 %, асфальтены 0.91 %) — `scn_slope`, `wax_alpha_eff`, `wax_Tm_shift` (`docs/calibrate.py`, разд. A) |
| `../парафиновое/togasheva2026_uzen_cooling_WAT.pdf` | Узень XIII: парафин 22–29 %, смолы до 21 %, асфальтены 0.94–3 %, P_b = 9–10 МПа |
| `../../full_review.pdf` | Обзор: SARA, SCN, multi-solid и твердый раствор, Флори–Хаггинс, PC-SAFT — разд. 1 |
