# Литература по парафиновой кольматации пласта: что лежит здесь и с чем можно сравнивать модель

Отбор — работы не старше 15 лет (2011–2026), кроме уже использованных ранее классических. Для каждой — что в ней
есть для проверки модели (`paraphin`, `experiments/исходная_модель/core_flood.py`, `validation_*.py`) и что уже сделано.
Поиск 27.09.2026; скачано только то, что открыто законно (сайты издателей, репозитории вузов, Europe PMC).
Пиратские копии (dokumen.pub, scribd и т. п.) не брались.

## Уже используются в сравнениях

| Файл | Источник | Что взято |
|---|---|---|
| `li2024_zhetybai_cooling_damage.pdf` | Li X. et al. Cooling Damage Characterization ... // Processes. 2024. 12:421. doi:10.3390/pr12020421, CC BY | кривая выпадения (подгонка Tm, alpha), вязкость (рис. 4а), k/k0 керна при 90/65/45/25 C (рис. 5) — `validation_li2024.py` |
| `sandyga2020_formation_damage_wax_core.pdf` | Sandyga M.S., Struchkov I.A., Rogachev M.K. // J. Petrol. Explor. Prod. Technol. 2020. 10:2541. CC BY | WAT 10–60 %, градиент давления при охлаждении, томография пор — `validation_sandyga2020.py` |
| `he2020_changchunling_cold_damage.pdf` | He Y. et al. // Sci. Rep. 2020. 10:14223. CC BY | пористость и проницаемость кернов при 50–5 C — `validation_he2020.py` |
| `togasheva2026_uzen_cooling_WAT.pdf` | Togasheva A. et al. // Eng. 2026. 7(4):184. CC BY | Узень XIII: парафин 22–29 %, WAT 41–44 C, остывание зон до 20–30 C — параметры полевого расчета |

## Новые находки, скачаны

| Файл | Источник | Что в нем для сравнения | Полезность |
|---|---|---|---|
| `sandyga2022_dissertation_spmi.pdf` | Сандыга М.С. Предотвращение образования органических отложений в системе «пласт–скважина» на поздней стадии разработки нефтяного месторождения: дис. ... канд. техн. наук. СПб.: Горный ун-т, 2022 | Подробнее, чем статья 2020: проницаемость керна по азоту и по кривой рис. 2.5.4 — ~6.5 мД выше 35 C (снимает противоречие k0 в нашем сравнении, где по перепаду из статьи выходило 0.03 мД); два опыта охлаждения 1 C/ч при эффективном давлении 4 и 5.3 МПа, начальная T 60 C (рис. 2.5.3 — WAT растет с давлением); **опыт при постоянных 34 C: градиент давления от прокачанного объема до 12 PV (рис. 2.5.5)** — прямо сопоставим с изотермическим расчетом керна; реология раствора (рис. 2.4.2); распределение пор до/после (рис. 2.6.4). Парафин Т-1 (ГОСТ 23689-89) в керосине ТС-1. Гл. 3 — влияние ПАВ на WAT и реологию. Текст — растр без Unicode, данные только с графиков | высокая |
| `khaibullina2019_dissertation_spmi.pdf` | Хайбуллина К.Ш. Обоснование комплексной технологии удаления и предупреждения органических отложений в скважинах на поздней стадии разработки нефтяного месторождения: дис. ... канд. техн. наук. СПб.: Горный ун-т, 2018 | Та же школа (Рогачев); отложения в скважине, свойства нефтей и WAT — для термодинамики, не для керна | низкая |
| `aleksandrov2022_dissertation_ESP_ASPO.pdf` | Александров А.Н. Обоснование комплексной технологии предупреждения образования АСПО при добыче высокопарафинистой нефти погружными ЭЦН из многопластовых залежей: дис. ... канд. техн. наук. СПб.: Горный ун-т, 2022 | Скважина и насос; данные по WAT и выпадению высокопарафинистых нефтей | низкая |
| `abileva2025_phd_satbayev.pdf` | Абилева С.Ж. Совершенствование химических и тепловых методов увеличения добычи нефти из неоднородных пластов: дис. ... PhD. Алматы: КазНИТУ им. Сатпаева, 2025 | Полимерное заводнение, гели, фильтрация вязкопластичных нефтей (разд. 3.1–3.2), пароциклика; парафина мало — пригодится для гелеобразования (в модели его нет) | низкая |
| `sobina_dissertation_ugtu.pdf` | Собин А.М. Регулирование разработки нефтяных месторождений на основе выявленных закономерностей фильтрации флюидов в призабойной зоне скважины: дис. ... канд. техн. наук. Ухта: УГТУ, 2015 | Фильтрация в призабойной зоне, повреждение пласта; парафиновой кольматации как темы нет | низкая |
| `review2022_crude_oil_wax_petroleum_science.pdf` | Kiyingi W., Guo J.-X., Xiong R.-Y., Su L., Yang X.-H., Zhang S.-L. Crude oil wax: A review on formation, experimentation, prediction, and remediation techniques // Petroleum Science. 2022. 19:2343–2357, CC BY-NC-ND | Обзор: методы WAT, термодинамические и кинетические модели, свойства парафинистых нефтей разных регионов — для проверки растворимости | средняя |
| `huang_fogler2015_wax_deposition_book_preview.pdf` | Huang Z., Zheng S., Fogler H.S. Wax Deposition: Experimental Characterizations, Theoretical Modeling, and Field Practices. CRC Press, 2015 — ознакомительный фрагмент издателя | Монография по отложению парафина (в основном трубопроводы): термодинамика, массоперенос, кинетика; полный текст платный | средняя |
| `ochieng2024_phd_wax_deposition_jkuat.pdf` | Ochieng F.O. Modeling and Analysis of Wax Deposition from Multiphase Flow in Field-Scale Crude Oil Pipeline Transport Systems: PhD thesis. JKUAT (Кения), 2024 | Модели отложения в трубопроводе; для пласта — только постановки массопереноса | низкая |
| `pmc12949295_high_pour_point_co2_hot_water.xml` | Yu P. et al. Comparison of mining performance of carbon dioxide displacement and hot water displacement for high pour point oil based on index variation characteristics // Science Progress. 2026. doi:10.1177/00368504261428354, CC BY-NC (полный текст JATS XML из Europe PMC; PDF закрыт капчей) | Длинный керн и slim tube: закупорка парафином при вытеснении горячей водой и CO2 высокозастывающей нефти — качественное сравнение | средняя |
| `pmc10724555_review_porous_media_models.xml` | Simonov O.A., Erina Yu.Yu., Ponomarev A.A. Review of modern models of porous media for numerical simulation of fluid flows // Heliyon. 2023. e22292. doi:10.1016/j.heliyon.2023.e22292, CC BY-NC-ND (JATS XML) | Модели пористой среды (пучок капилляров, сетевые, цифровой керн) — к вопросу о форме fi_0 и связи k(m) | средняя |

## Найдены, но не скачаны (платный доступ или только за входом)

| Источник | Что в нем | Почему важен |
|---|---|---|
| Nikiforov A.I., Sadovnikov R.V., Nikiforov G.A. Modeling of Paraffin Deposition During Oil Production by Cold Water Injection // Lobachevskii J. Math. 2022. 43:1178–1183. doi:10.1134/S199508022208025X | Расчеты той же группы (А.И. Никифоров — соавтор статьи ТВТ) на прежней версии модели | Прямое сравнение «до/после» правок модели: взять у авторов |
| Comparative Analysis of Methods for Determining the Wax Crystallization Onset Temperature of High-Paraffin Crude Oil from the Uzen Field, 2026 (ResearchGate 401609702; журнал не установлен) | Узень XIII, парафин 22.5–27.5 %: начало кристаллизации 38.0–41.7 C (светопропускание, 12 МПа), 41–44 C (Wax Flow Loop), массовая кристаллизация 33.5–35 C | WAT модели для w = 0.25 — 38.8 C, в пределах светопропускания при 12 МПа |
| A New Approach to Simulation of Wax Precipitation During Cold Water Injection in Carbonate Reservoir of Kharyaga Oilfield // SPE Russian Petroleum Technology Conference, 2021 (OnePetro 21RPTC, D021S009R002) | Расчет выпадения парафина при закачке холодной воды на карбонатах Харьяги | Расчеты других авторов для российского месторождения |
| Nie X., Yang J. Numerical simulation of paraffin wax deposition in waxy crude oil reservoir with cold water flooding // Geosystem Engineering. 2014. 17(4). doi:10.1080/12269328.2014.959623 | Численная модель кольматации при холодном заводнении (цитируется у Li et al.) | Расчеты других авторов |
| Cold water-flooding in a heterogeneous high-pour-point oil reservoir using computerized tomography scanning: characteristics of flow channel and trapped oil distribution // J. Petrol. Sci. Eng. 2021 (S0920410521002540) | Томография керна при холодном заводнении высокозастывающей нефти | Двухфазная постановка, как в поле |
| Experimental investigation of the live oil-water relative permeability and displacement efficiency on Kingfisher waxy oil reservoir // J. Petrol. Sci. Eng. 2019 (S0920410519303638) | ОФП пластовой парафинистой нефти | Двухфазная проверка |
| Waterflooding strategy for a reservoir containing a high pour point oil // Petroleum Science and Technology. 2016. 34(13) | Стратегия заводнения высокозастывающей нефти | Поле |
| Civan F. Reservoir Formation Damage: Fundamentals, Modeling, Assessment, and Mitigation. 3rd ed. Gulf Professional, 2015 (ч. IV — органические отложения) | Модели осаждения в порах (в т.ч. модель, с которой сравниваются Wang & Civan 2005) | Классика моделей кольматации |
| Fei R. et al. Intrinsic Causes and Characteristics of Cold Injury in the Flow of Highly Waxy Crude Oil within Porous Media in Low Permeability Sandstone Reservoirs (SSRN 5335415, 2025) | ЯМР онлайн-вытеснение, 0.28 г парафина в керне при 32 C | Та же группа, что Li et al. |

## Собранные ранее (термодинамика и кинетика выпадения)

`hammami1997.pdf`, `kord2014.pdf`, `fallahnejad2015.pdf`, `solaimany-nazar2011.pdf`, `subramanian2015.pdf`,
`issledovanie-kinetiki-vydeleniya-parafinov-iz-nefti.pdf`, `супер казах.pdf`, `О переносе частиц.doc`, `2340901.pdf`,
`Solid Phase Equation of State ... .pdf`. Ring et al. (1994), Wang & Civan (2005), Sutton & Roberts (1974),
Maloney & Oesthus (2004) — опорные данные калибровки, см. `docs/WAX_PRECIPITATION_FINDINGS.md`.

## Что стоит сделать дальше

1. Оцифровать из диссертации Сандыги рис. 2.5.5 (градиент от PV при 34 C) и 2.5.4 (k от T, k0 ~ 6.5 мД) и добавить в
   `validation_sandyga2020.py` — изотермическое сравнение и k0 из опыта вместо пучка капилляров.
2. Запросить у А.И. Никифорова статью 2022 г. — сравнение с прежней версией модели.
