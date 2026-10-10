# Этап 1. Разбор `resources/full_review.pdf`

*Документ прочитан целиком (71 с., извлечение текста PyMuPDF; исходник LaTeX и BibTeX — в `resources/claude_reserch/very_big_review.zip`). Ниже — пересказ своими словами, без дословного копирования; номера в квадратных скобках — номера списка литературы `full_review`.*

## 1. Паспорт

| Параметр | Значение |
|---|---|
| Название | «Математическое моделирование осаждения парафинов, асфальтенов и других тяжелых компонентов нефти при неизотермической фильтрации в пористой среде. Систематический обзор литературы» |
| Год | 2026 |
| Объем | 71 с.: введение, 8 разделов, заключение, список литературы (с. 65–71) |
| Источников | 105; у 22 нет авторов (только год и название), у 2 нет DOI |
| Проверка списка | 96 DOI ведут на заявленные работы (у 10 год отличается от OpenAlex на год онлайн-публикации), **6 DOI ведут на посторонние статьи или не существуют**, 1 запись — энциклопедическая статья PetroWiki без авторов, 2 записи без DOI (таблица в разд. 8) |
| Годы цитируемых работ (97 проверенных) | 1960-е — 1; 1970-е — 1; 1980-е — 11; 1990-е — 11; 2000-е — 16; 2010-е — 31; 2020-е — 26; медиана 2014; доля 2019–2026 — 30 %, 2023–2026 — 11 работ |

## 2. Структура и логика изложения

Логика обзора — иерархия масштабов: молекулярная термодинамика → кинетика переноса к стенке → континуальная модель пласта → численная реализация и проверка → тренды. Явно сформулирована «петля обратной связи»: термодинамика → кинетика захвата и выноса → изменение пористости и проницаемости → поле скоростей и температур → снова условия выпадения.

| Раздел (страницы) | Содержание | Опорные работы `full_review` |
|---|---|---|
| Введение (4–6) | Единая природа выпадения (потеря устойчивости раствора) при разной геометрии (пласт, скважина, труба); различие парафинов (молекулярный раствор, кристаллизация) и асфальтенов (коллоидная система); цель и структура | — |
| 1. Термодинамика (7–13) | Классы углеводородов (PNA/PIONA), SARA и SCN-характеризация, условие равновесия через летучести, термодинамический цикл плавления; твердый раствор (Вон; Хансен; Педерсен), multi-solid (Лира-Галеана и др.), локально-композиционная модель Кутиньо–Стенби, Флори–Хаггинс и коллоидные модели асфальтенов, PC-SAFT; WAT как точка бифуркации | [103], [39], [70], [49], [27], [104], [105], [47], [90], [91], [36], [18], [42], [102], [76] |
| 2. Кинетика (14–20) | Механизмы переноса к стенке (диффузия Фика, сдвиговая дисперсия, броуновская диффузия, оседание, сдирание), модели роста слоя (мгновенная растворимость против кинетики кристаллизации), сопряженная тепловая задача, старение (встречная диффузия) | [17], [101], [15], [40], [81], [82] |
| 3. Связанные неизотермические модели (21–27) | Балансы компонентов со стоком в отложения, многофазный Дарси, уравнение энергии с теплотой кристаллизации (энтальпийно-пористый метод), кинетика захвата/увлечения, типовые допущения, граничные условия для скважинных задач | [14], [55], [13], [49], [48], [4], [5], [37], [19], [61], [77] |
| 4. Повреждение пласта (28–33) | Объемный и «горловой» эффекты, Козени–Карман и степенные поправки, функция повреждения, скин Хокинса, обратная связь | [37], [86], [32], [45], [19], [83], [77], [61] |
| 5. Асфальтены и смолы (34–39) | Молекулярная архитектура (Йен–Маллинс), гетероатомы, колоколообразная зависимость от давления, PC-SAFT/CPA/NRTL-SAC, модели осаждения (Civan, стохастическая глубинная фильтрация, ADEPT, MAD-ADEPT), связь с композиционной фильтрацией | [58], [57], [73], [11], [84], [56], [46], [69], [67], [68], [28], [2], [91], [80], [72], [43], [52], [1], [85], [34], [55], [87], [8] |
| 6. Численные методы и симуляторы (40–48) | МКО, IMPES/FIM, полунеявные схемы, CPR, MsFV, OBL (DARTS); псевдокомпонентное представление; таблица симуляторов (ADEPT, MAD-ADEPT, UTCOMP, GEM, ECLIPSE, MRST+PC-SAFT, CPA-модель Nasrabadi, Kohse–Nghiem, DARTS, MsFV, Svendsen, Nie–Yang) | [33], [25], [87], [51], [29], [26], [23], [24], [54], [95], [94] |
| 7. Валидация (49–54) | WAT (ДСК, ЯМР), SARA (μSARA-HPLC), cold finger и flow loop; валидация на трубопроводе (Ochieng; эмпирическая модель Ильюшина и др.), на скважине (Marrat, переинтерпретация Kargarpour); источники неопределенности | [62], [74], [50], [79], [20], [63], [100], [44], [41], [96] |
| 8. Тренды (55–60) | Детальный состав (ФТ-ИЦР-МС), ML для WAT, скорости отложения, AOP; PINN и гибридные модели; открытые проблемы | [63], [64], [75], [92], [93], [78], [22], [31], [3], [7], [6], [12], [53], [38], [71], [35] |
| Заключение (61–64) | Трехуровневая систематизация (термодинамический, кинетический, континуальный уровни) и 6 открытых проблем | — |

## 3. Ключевые выводы (пересказ)

1. **Термодинамика.** Для парафинов рабочие модели — твердый раствор и multi-solid; для узких по составу восков разумен твердый раствор, для широких — расслоение на несколько твердых фаз (multi-solid). Для асфальтенов нужен либо полимерный (Флори–Хаггинс) / коллоидный подход, либо уравнения состояния с ассоциацией (PC-SAFT, CPA). Точность WAT определяется прежде всего характеризацией тяжелого хвоста н-алканов.
2. **Кинетика.** В трубах доминирует молекулярная диффузия растворенного парафина к холодной стенке; учет кинетики кристаллизации в объеме исправляет и завышение (модель мгновенной растворимости), и занижение толщины отложения [40]. Старение отложения — встречная диффузия компонентов с критическим углеродным числом [81].
3. **Континуальная модель.** Минимальный состав: балансы компонентов со стоком в осадок, многофазный Дарси с вязкостью, зависящей от температуры, уравнение энергии с теплотой кристаллизации, равновесие и кинетика захвата/увлечения. Типовые допущения: локальное равновесие (фазовое и термическое), квазистационарное давление на шаге, пучок капилляров или упаковка шаров для эффективных коэффициентов.
4. **Повреждение.** Проницаемость падает сильнее, чем пористость: осадок садится в горлах; классический Козени–Карман занижает эффект, степенные поправки дают показатели до 8–19 [45].
5. **Численные методы и ПО.** Узкое место — не дискретизация переноса, а согласование термодинамики твердой фазы с кинетикой захвата. Промышленные симуляторы (UTCOMP, GEM, ECLIPSE) на одинаковых PVT-данных расходятся в AOP и массе осадка [23].
6. **Валидация.** Точность ограничивают входные данные (состав, кинетические константы), а не форма модели; лабораторные методики (cold finger и flow loop) плохо масштабируются.
7. **Тренды.** Переход к детальному составу и ML/PINN; физически информированные модели лучше экстраполируют [3].

## 4. Классификации и таблицы сравнения

- **Трехуровневая схема** (с. 62): термодинамический («сколько твердого»), кинетический («где и как быстро»), континуальный («как осаждение меняет течение»); для каждого уровня — модели для парафинов, для асфальтенов и главный источник неопределенности. Эта схема принята и в настоящем обзоре (разд. 3), но дополнена четвертым уровнем — **порового пространства** (пучок, сеть, глубинная фильтрация), который в `full_review` растворен в разделе 4.
- **Таблица симуляторов** (с. 44–47): 12 инструментов по типу термодинамической модели, учтенным процессам, области применения и ограничениям. Ни один из перечисленных пластовых симуляторов не учитывает неизотермическую кристаллизацию парафина с кинетикой осаждения в пласте: асфальтеновые модули UTCOMP/GEM/ECLIPSE изотермичны по сути, модели Svendsen и ADEPT — трубные, Nie–Yang — пластовая, но равновесная и без пористой микроструктуры.

## 5. Открытые проблемы (по `full_review`, с. 63–64, пересказ)

1. Нет переносимых между месторождениями физически обоснованных корреляций для констант захвата и выноса.
2. Детальная кинетика в каждой ячейке промыслового симулятора дорога; нужны суррогаты и апскейлинг.
3. Масштабирование лабораторных установок (cold finger, flow loop) к промыслу.
4. Нет больших открытых стандартизированных баз SARA, WAT, AOP.
5. Почти нет количественной оценки неопределенности прогноза.
6. Трудно разделить вклады термодинамики, кинетики и многофазной гидродинамики при интерпретации промысловых данных (пример Marrat).

## 6. Критический анализ

**Сильные стороны.** Последовательная архитектура «от молекулы к пласту», явная формулировка обратной связи, широкий охват асфальтеновой термодинамики, таблица симуляторов, связка с ML-трендами.

**Слабые места (важные для диссертации).**

1. **Ошибки библиографии.** Шесть DOI указывают на посторонние работы (фурфурол, адсорбция фенола, ионные жидкости, газ угольных пластов, ванадат висмута) или не существуют (№ 6, 31, 78, 92, 93, 98). У 22 записей нет авторов, у двух — DOI (№ 12, 22; у № 22 неверен и журнал). Все семь работ найдены по названию, верные DOI — в разд. 8 и `data/excludes.csv`. Цитировать `full_review` без перепроверки нельзя.
2. **Пласт представлен слабее, чем труба и скважина.** Из 105 ссылок прямо о парафине в пористой среде — 3 (Ring et al. 1994 [77], Nie & Yang 2014 [94], Никифоров и др. 2022 [61]); разделы 2 и 7 почти целиком трубные. Керновые опыты с парафином, на которых только и можно проверить пластовую модель (Sutton & Roberts 1974; Yang et al. 2013; Sandyga et al. 2020; He et al. 2020; Wang et al. 2021; Li et al. 2024), не упомянуты.
3. **Модели порового пространства не разобраны.** Нет пучка капилляров с функцией распределения пор, сетевых моделей (Fatt, Kirkpatrick), различия «сужение стенки / закупорка горла», модели Wang–Civan с параллельными путями, функции максимального удержания Бедриковецкого — то есть именно того, что определяет связь k(φ) для органических отложений.
4. **Неизотермичность пласта — без ее специфики.** Нет решений Ловерье и Маркса–Лангенхайма, метода Винсома–Вестервельда для теплообмена с кровлей и подошвой, обсуждения отставания теплового фронта от фронта вытеснения (Maloney & Oesthus 2004), LTNE упомянуто одним абзацем.
5. **Реология в пористой среде отсутствует.** Гель и предел текучести обсуждаются для трубопроводов (рестарт), но не закон Дарси для вязкопластичных жидкостей (Chevalier et al. 2013; Liu et al. 2019; Wu et al. 1992) и не опыты Al-Fariss & Pinder (1987).
6. **Численные методы — общими словами.** Нет анализа устойчивости IMPES (Coats 2000, 2003), решателей давления (многосеточные, AMG/CPR), моделей скважин (Писман), ограничения скорости осаждения подводом, жесткости кинетики.
7. **Русскоязычная школа** — 2 работы (Бородин и др. 2022 о гидратах [14], Никифоров и др. 2022 [61]); нет Никифорова–Никаньшина (1998), Бедриковецкого и соавт. (1997) о заводнении высокопарафинистых пластов, работ Горного университета (Рогачев, Стручков, Сандыга), Казахстана (Узень: Togasheva et al. 2026; Bayamirova et al. 2024, 2026).
8. **Свежесть.** Медиана — 2014 г.; работы 2023–2026 гг. — в основном ML и обзоры; нет пластовых моделей 2024–2026 гг. (Bekibayev et al. 2026; Zhao et al. 2026; Togasheva et al. 2026; Eskin 2025; Chen et al. 2025, 2026; Tang et al. 2024).
9. **Отдельные утверждения без опоры.** Например, «в стеклянных микромоделях закупорка по механизму снежного кома» дана по одной работе [72] без количественных данных; оценки точности эмпирических моделей (6 % по толщине) перенесены с одной скважины [100] на класс моделей.

## 7. Что устарело и что дополнено свежей литературой (2019–2026)

| Тема `full_review` | Что дополнено в настоящем обзоре |
|---|---|
| Термодинамика парафина (опора — 1986–2001) | давление и газ в WAT (Helsper & Liberatore 2024; Khalighi et al. 2022), WAT от скорости охлаждения (Ruwoldt et al. 2018; Struchkov & Rogachev 2017), PR + UNIQUAC (da Silva et al. 2017), методы определения WAT высокопарафинистой нефти (Bayamirova et al. 2026) |
| Асфальтены (опора — 1987–2016) | AOP при закачке газа (Abutaqiya et al. 2020), трехфазные алгоритмы равновесия (Li et al. 2019), обзоры AOP (Fakher et al. 2024), опыты в керне без осадителя (Struchkov et al. 2019) и с микро-КТ (Lin et al. 2021; Zhang et al. 2022) |
| Кинетика в пористой среде | вынос и эрозия (Bizmark et al. 2020; Kahza & Sanaei 2024; Jäger et al. 2017), кинетика агрегации (Li X. et al. 2017; Su et al. 2021), адсорбция полярных компонентов (Piñerez Torrijos et al. 2020; Puntervold et al. 2021; Mamonov et al. 2022) |
| Модели пласта | Bekibayev et al. 2026, Togasheva et al. 2026, Zhao et al. 2026, Eskin 2025, Chen et al. 2025/2026, Akter et al. 2025, Никифоров и др. 2024, Tananykhin et al. 2022 |
| Неизотермическая фильтрация | Maloney & Oesthus 2004, Wang M. et al. 2021 (КТ холодного заводнения), Bayamirova et al. 2024, Leonov 2025 |
| Реология в пористой среде | Chevalier et al. 2013, Liu et al. 2019, Tang et al. 2024 (неньютоновская фильтрация высокозастывающей нефти) |
| Численные методы и ПО | OPM (2021), DuMux 3 (2021), MRST (2019, 2021), JutulDarcy (2025), DARTS (2024), GEOS (2024), SPE11 (2024) |
| ML | U-FNO (2022), DeepONet (2021), ограничения PINN для переноса (Fuks & Tchelepi 2020), обзор ML в моделировании пластов (Samnioti & Gaganis 2023) |

## 8. Структурированный список литературы `full_review` с проверкой

Таблица построена скриптом (`data/full_review_refs.md`): номер в `full_review`, запись в сокращении, DOI и результат проверки в OpenAlex/Crossref. «Год в записи … по OpenAlex …» — обычно разница между онлайн- и печатной публикацией, это не ошибка.

| № | Запись в full_review (сокращенно) | DOI | Проверка (OpenAlex/Crossref, 09.10.2026) |
|---|---|---|---|
| 1 | Anjushri S. Kurup, Jianxin Wang, Hariprasad J. Subramani, Jill Buckley, Jefferson L. Creek, Walter G. Chapman. (2012). Revisiting Asphaltene | 10.1021/ef300714p | верно: Anjushri S. Kurup, Jianxin Wang и др., 2012 |
| 2 | Md Rashedul Islam, Yifan Hao, Chau-Chyun Chen. (2020). Aggregation thermodynamics of asphaltenes: Prediction of asphaltene precipitation in  | 10.1016/j.fluid.2020.112655 | верно: Md Rashedul Islam, Yifan Hao и др., 2020 |
| 3 | Ahmadi, M. (2025). From composition to deposition: A machine learning framework for wax deposition prediction in petroleum fluids using comp | 10.1063/5.0276315 | верно: Mohammad Ali Ahmadi, 2025 |
| 4 | Alhosani, Ahmed, Ravichandran, Sriram, Daraboina, Nagu. (2020). Review of Asphaltene Deposition Modeling in Oil and Gas Production. Energy & | 10.1021/acs.energyfuels.0c02981 | верно: Ahmed Alhosani, Sriram Ravichandran и др., 2020 |
| 5 | Alimohammadi, Sepideh, Zendehboudi, Sohrab, James, Lesley. (2019). A comprehensive review of asphaltene deposition in petroleum reservoirs:  | 10.1016/j.fuel.2019.03.016 | верно: Sepideh Alimohammadi, Sohrab Zendehboudi и др., 2019 |
| 6 | (2024). Robust asphaltene onset pressure prediction using ensemble learning. Geoenergy Science and Engineering. | 10.1016/j.geoen.2024.212857 | **ОШИБКА**: DOI из full_review (№ 6) ведет на статью о газе угольных пластов; верная работа - Khalighi et al. 2024, 10.1016/j.rineng.2024.103483 (включена) |
| 7 | (2023). Integrated Machine Learning Model for Predicting Asphaltene Damage Risk and the Asphaltene Onset Pressure. Energy & Fuels. | 10.1021/acs.energyfuels.2c03319 | верно: Iván Moncayo-Riascos, Christian Guerrero-Benavides и др., 2022; год в записи 2023, по OpenAlex 2022 |
| 8 | Hamed Darabi, Mahdy Shirdel, M. Hosein Kalaei, Kamy Sepehrnoori. (2014). Aspects of Modeling Asphaltene Deposition in a Compositional Couple | 10.2118/169121-ms | верно: Hamed Darabi, Mahdy Shirdel и др., 2014 |
| 9 | Asphaltene deposition, plugging. (2025). Asphaltene deposition and plugging. | 10.2118/pw0043 | DOI верен (PetroWiki); энциклопедическая статья без авторов — в реестр не включена |
| 10 | Ahmed Alhosani, Nagu Daraboina. (2020). Modeling of asphaltene deposition during oil/gas flow in wellbore. Fuel, 280, 118617. | 10.1016/j.fuel.2020.118617 | верно: Ahmed Alhosani, Nagu Daraboina, 2020 |
| 11 | John G. Reynolds. (1994). Chapter 10 Effects of Asphaltene Precipitation on the Size of Vanadium-, Nickel-, And Sulfur-Containing Compounds  | 10.1016/s0376-7361(09)70257-2 | верно: John G. Reynolds, 1994 |
| 12 | (2024). Asphaltene Stability Prediction Using Hybrid Artificial Neural Network Modeling Approach. Springer Nature. |  | нет DOI и авторов; работа найдена: Sulaimon et al. 2024, SPE-221598-MS, 10.2118/221598-ms (включена) |
| 13 | Banki, Reza, Hoteit, Hussein, Firoozabadi, Abbas. (2008). Mathematical formulation and numerical modeling of wax deposition in pipelines fro | 10.1016/j.ijheatmasstransfer.2007.11.012 | верно: Reza Banki, Hussein Hoteit и др., 2008 |
| 14 | Borodin, S. L., Musakaev, N. G., Belskikh, Denis S. (2022). Mathematical Modeling of a Non-Isothermal Flow in a Porous Medium Considering Ga | 10.3390/math10244674 | верно: С. Л. Бородин, N. G. Musakaev и др., 2022 |
| 15 | Brown, T. S., Niesen, Vicki G., Erickson, Dale. (1993). Measurement and Prediction of the Kinetics of Paraffin Deposition. SPE Annual Techni | 10.2118/26548-ms | верно: Trent S. Brown, Vicki G. Niesen и др., 1993 |
| 16 | Buenrostro-González, Eduardo, Lira-Galeana, C., Gil‐Villegas, Alejandro, Wu, Jianzhong. (2004). Asphaltene precipitation in crude oils: Theo | 10.1002/aic.10243 | верно: Eduardo Buenrostro-González, C. Lira-Galeana и др., 2004 |
| 17 | Burger, E. D., Perkins, T.K., Striegler, J.H. (1981). Studies of Wax Deposition in the Trans Alaska Pipeline. Journal of Petroleum Technolog | 10.2118/8788-pa | верно: Edward D. Burger, Thomas K. Perkins и др., 1981 |
| 18 | Chapman, Walter G., Gubbins, Keith E., Jackson, George, Radosz, Maciej. (1989). SAFT: Equation-of-state solution model for associating fluid | 10.1016/0378-3812(89)80308-5 | верно: Walter G. Chapman, Keith E. Gubbins и др., 1989 |
| 19 | Civan, Faruk. (2015). Modified Formulations of Particle Deposition and Removal Kinetics in Saturated Porous Media. Transport in Porous Media | 10.1007/s11242-015-0600-z | верно: Faruk Civan, 2015 |
| 20 | (2018). Effect of the Flow Field on the Wax Deposition and Performance of Wax Inhibitors: Cold Finger and Flow Loop Testing. Energy & Fuels. | 10.1021/acs.energyfuels.7b00253 | верно: Yuandao Chi, Nagu Daraboina и др., 2017; год в записи 2018, по OpenAlex 2017 |
| 21 | (2009). Solids Deposition during “Cold Flow’ ’ of Wax-Solvent Mixtures in a Flow-loop Apparatus with Heat Transfer. Energy & Fuels. | 10.1021/ef900224r | верно: Hamid O. Bidmus, Anil Kumar Mehrotra, 2009 |
| 22 | (2018). Combining of intelligent models through committee machine for estimation of wax deposition. Petroleum. |  | нет DOI и авторов, журнал указан неверно («Petroleum»); работа найдена: Gholami et al. 2018, J. Chin. Chem. Soc., 10.1002/jccs.201700329 (включена) |
| 23 | Abdulaziz Al-Qasim, Mudhish A AlDawsari. (2017). Comparison Study of Asphaltene Precipitation Models Using UTCOMP, CMG/GEM and ECLIPSE Simul | 10.2118/185370-ms | верно: Abdulaziz S. Al-Qasim, Mudhish Aldawsari, 2017 |
| 24 | K. H. Coats, K. H. Coats, L. K. Thomas, R. G. Pierson. (1995). Compositional and Black Oil Reservoir Simulation. SPE Reservoir Simulation Sy | 10.2118/29111-ms | верно: Keith H. Coats, K. H. Coats и др., 1995 |
| 25 | H. Hajibeygi, H. A. Tchelepi. (2013). Compositional Multiscale Finite-Volume Formulation. SPE Reservoir Simulation Symposium. | 10.2118/163664-ms | верно: Hadi Hajibeygi, Hamdi A. Tchelepi, 2013 |
| 26 | K. Gonzalez, M. A. Barrufet, H. Nasrabadi. (2014). Development of a Compositional Reservoir Simulator Including Asphaltene Precipitation fro | 10.2118/169401-ms | верно: Karin Gonzalez, María A. Barrufet и др., 2014 |
| 27 | Coutinho, João A. P., Stenby, Erling H. (1996). Predictive Local Composition Models for Solid/Liquid Equilibrium in n-Alkane Systems: Wilson | 10.1021/ie950447u | верно: João A. P. Coutinho, Erling Halfdan Stenby, 1996 |
| 28 | Wenlong Jia, Ryosuke Okuno. (2018). Modeling of asphaltene and water associations in petroleum reservoir fluids using cubic‐plus‐association | 10.1002/aic.16191 | верно: Wenlong Jia, Ryosuke Okuno, 2018 |
| 29 | Hadi Nasrabadi, Joachim Moortgat, Abbas Firoozabadi. (2013). A New Three-Phase Multicomponent Compositional Model for Asphaltene Precipitati | 10.2118/163587-ms | верно: Hadi Nasrabadi, Joachim Moortgat и др., 2013 |
| 30 | P. Bedrikovetsky, A. Santos, A. Siqueira, A. L. Souza, F. Shecaira. (2003). A Stochastic Model for Deep Bed Filtration and Well Impairment.  | 10.2118/82230-ms | верно: Pavel Bedrikovetsky, A. Santos и др., 2003 |
| 31 | (2024). Prediction Model of Wax Deposition Rate in Waxy Crude Oil Pipelines by Elman Neural Network Based on Improved Reptile Search Algorit | 10.1016/j.petrol.2024.212763 | **ОШИБКА**: DOI из full_review (№ 31) не существует; верная работа - Chen et al. 2024, Energy Engineering, 10.32604/ee.2023.045270 (включена) |
| 32 | Feng, Xiao, Zeng, Jianhui, Ma, Yong, Jia, Kaiyu, Qiao, Juncheng, Zhang, Yongchao, Feng, Sen. (2017). Asphaltene Deposition Preference and Pe | 10.1021/acs.energyfuels.7b01389 | верно: Xiao Feng, Jianhui Zeng и др., 2017 |
| 33 | Sebastián Echavarría-Montaña, Steven Velásquez, Nicolás Bueno, Juan David Valencia, Hillmert Alexander Solano, Juan Manuel Mejía. (2021). Se | 10.3390/fluids6100341 | верно: Sebastián Echavarría-Montaña, Steven Velásquez Chanci и др., 2021 |
| 34 | Xiangjun Qin, Peng Wang, Kamy Sepehrnoori, Gary A. Pope. (2000). Modeling Asphaltene Precipitation in Reservoir Simulation. Industrial & Eng | 10.1021/ie990781g | верно: Xiangjun Qin, Peng Wang и др., 2000 |
| 35 | (2025). A Comprehensive Review of Flow Assurance in the Energy Transition: Flow Loop Platforms and AI-Driven Solutions for Hydrogen, CO2, an | 10.1007/s13369-025-10758-x | верно: Ali Mahmoud, Ala Shafeq AL-Dogail и др., 2025 |
| 36 | Groß, Joachim, Sadowski, Gabriele. (2001). Perturbed-Chain SAFT: An Equation of State Based on a Perturbation Theory for Chain Molecules. In | 10.1021/ie0003887 | верно: Joachim Groß, Gabriele Sadowski, 2001 |
| 37 | Gruesbeck, C., Collins, R. E. (1982). Entrainment and Deposition of Fine Particles in Porous Media. Society of Petroleum Engineers Journal,  | 10.2118/8430-pa | верно: C. Gruesbeck, R. E. Collins, 1982 |
| 38 | Han, Jiang-Xia, Xue, Liang, Wei, Yu-Shu, others. (2023). Physics-informed neural network-based petroleum reservoir simulation with sparse da | 10.1016/j.petsci.2023.10.019 | верно: Jiangxia Han, Liang Xue и др., 2023 |
| 39 | Hansen, Jens Henrik, Fredenslund, Aa., Pedersen, Karen Schou, Rønningsen, Hans Petter. (1988). A thermodynamic model for predicting wax form | 10.1002/aic.690341202 | верно: Jens Henrik Hansen, Aa. Fredenslund и др., 1988 |
| 40 | Huang, Zhenyu, Lee, H. S., Senra, Michael, Fogler, H. Scott. (2010). A fundamental model of wax deposition in subsea oil pipelines. AIChE Jo | 10.1002/aic.12517 | верно: Zhenyu Huang, Hyun Su Lee и др., 2010 |
| 41 | Kargarpour, M., Dandekar, A. Y. (2016). Analysis of asphaltene deposition in Marrat oil well string: a new approach. Journal of Petroleum Ex | 10.1007/s13202-015-0221-7 | верно: Mohammad Ali Kargarpour, Abhijit Yeshwant Dandekar, 2015; год в записи 2016, по OpenAlex 2015 |
| 42 | Katz, David L., Firoozabadi, Abbas. (1978). Predicting Phase Behavior of Condensate/Crude-Oil Systems Using Methane Interaction Coefficients | 10.2118/6721-pa | верно: D.L. Katz, Abbas Firoozabadi, 1978 |
| 43 | B. ZareNezhad, H. Parsa. (2015). A Rigorous Kinetic Model for Description of Asphaltene Deposition in Porous Media of Petroleum Reservoirs.  | 10.1080/10916466.2015.1107846 | верно: Bahman ZareNezhad, Hakime Parsa, 2015 |
| 44 | Kor, Pooria, Kharrat, Riyaz, Ayoubi, Ali. (2017). Comparison and evaluation of several models in prediction of asphaltene deposition profile | 10.1007/s13202-016-0269-z | верно: Peyman Kor, Riyaz Kharrat и др., 2016; год в записи 2017, по OpenAlex 2016 |
| 45 | Krauss, Eva D., Mays, D. C. (2013). Modification of the Kozeny-Carman Equation to Quantify Formation Damage by Fines in Clean Unconsolidated | 10.2118/165148-pa | верно: Eva D. Krauss, D. C. Mays, 2014; год в записи 2013, по OpenAlex 2014 |
| 46 | Shaghayegh Darjani, Archana Jagadisan, Joel Koplik, Sanjoy Banerjee. (2022). Lattice-Gas Model for Asphaltene Interactions Observed at Inter | 10.1021/acs.energyfuels.2c01413 | верно: Shaghayegh Darjani, Archana Jagadisan и др., 2022 |
| 47 | Leontaritis, Kosta J., Mansoori, G. Ali. (1987). Asphaltene Flocculation During Oil Production and Processing: A Thermodynamic Colloidal Mod | 10.2118/16258-ms | верно: Kosta J. Leontaritis, G. Ali Mansoori, 1987 |
| 48 | Leontaritis, Kosta J., Mansoori, G. Ali. (1988). Asphaltene deposition: a survey of field experiences and research approaches. Journal of Pe | 10.1016/0920-4105(88)90013-7 | верно: Kosta J. Leontaritis, G. Ali Mansoori, 1988 |
| 49 | Lira-Galeana, C., Firoozabadi, Abbas, Prausnitz, John M. (1996). Thermodynamics of wax precipitation in petroleum mixtures. AIChE Journal, 4 | 10.1002/aic.690420120 | верно: C. Lira-Galeana, Abbas Firoozabadi и др., 1996 |
| 50 | (2025). Crude Oil Analysis by Low-Field NMR Relaxometry. Magnetic Resonance in Chemistry. | 10.1002/mrc.70096 | верно: Salim Ok, Marsel G. Fazlyyyakhmatov, 2026; год в записи 2025, по OpenAlex 2026 |
| 51 | K. C. Hong. (1982). Lumped-Component Characterization of Crude Oils for Compositional Simulation. SPE Enhanced Oil Recovery Symposium. | 10.2118/10691-ms | верно: K. C. Hong, 1982 |
| 52 | Saman Naseri, Saeid Jamshidi, Vahid Taghikhani. (2020). A new multiphase and dynamic asphaltene deposition tool (MAD-ADEPT) to predict the d | 10.1016/j.petrol.2020.107553 | верно: Saman Naseri, Saeid Jamshidi и др., 2020 |
| 53 | (2023). Comparative study of machine learning algorithms in predicting asphaltene precipitation with a novel validation technique. Earth Sci | 10.1007/s12145-023-01075-8 | верно: Jafar Khalighi, Аlexey Cheremisin, 2023 |
| 54 | Bruce F. Kohse, Long X. Nghiem. (2004). Modelling Asphaltene Precipitation and Deposition in a Compositional Reservoir Simulator. SPE/DOE Sy | 10.2118/89437-ms | верно: Bruce Kohse, Long X. Nghiem, 2004 |
| 55 | Monteagudo, Jorge E., Rajagopal, Krishnaswamy, Lage, Paulo L. C. (2002). Simulating oil flow in porous media under asphaltene deposition. Ch | 10.1016/s0009-2509(01)00407-9 | верно: Jorge E. Monteagudo, Krishnaswamy Rajagopal и др., 2002 |
| 56 | Qinghao Wu, Douglas J. Seifert, Andrew E. Pomerantz, Oliver C. Mullins, Richard N. Zare. (2014). Constant Asphaltene Molecular and Nanoaggre | 10.1021/ef500281s | верно: Qinghao Wu, Douglas J. Seifert и др., 2014 |
| 57 | Andrew E. Pomerantz, Qinghao Wu, Oliver C. Mullins, Richard N. Zare. (2015). Laser-Based Mass Spectrometric Assessment of Asphaltene Molecul | 10.1021/ef5020764 | верно: Andrew E. Pomerantz, Qinghao Wu и др., 2015 |
| 58 | Oliver C. Mullins, Hassan Sabbah, Joëlle Eyssautier, Andrew E. Pomerantz, Loïc Barré, A. Ballard Andrews, Yosadara Ruiz-Morales, Farshid Mos | 10.1021/ef300185p | верно: Oliver C. Mullins, Hassan Sabbah и др., 2012 |
| 59 | Mullins, Oliver C., Seifert, Douglas J., Zuo, Julian Y., Zeybek, Murat. (2012). Clusters of Asphaltene Nanoaggregates Observed in Oilfield R | 10.1021/ef301338q | верно: Oliver C. Mullins, Douglas J. Seifert и др., 2012 |
| 60 | Nichita, Dan Vladimir, Goual, Lamia, Firoozabadi, Abbas. (2001). Wax Precipitation in Gas Condensate Mixtures. SPE Production & Facilities,  | 10.2118/74686-pa | верно: Dan Vladimir Nichita, Lamia Goual и др., 2001 |
| 61 | Nikiforov, A. I., Sadovnikov, R. V., Nikiforov, G. A. (2022). Modeling of Paraffin Deposition During Oil Production by Cold Water Injection. | 10.1134/s199508022208025x | верно: A. I. Nikiforov, R. V. Sadovnikov и др., 2022 |
| 62 | (2023). Novel Nuclear Magnetic Resonance Techniques To Assess the Wax Precipitation Evolution in Crude Oil Systems. Energy & Fuels. | 10.1021/acs.energyfuels.2c03309 | верно: George Claudiu Savulescu, Sébastien Simon и др., 2022; год в записи 2023, по OpenAlex 2022 |
| 63 | Ochieng, R., others. (2022). Mathematical Modeling of Wax Deposition in Field-Scale Crude Oil Pipeline Systems. Journal of Applied Mathemati | 10.1155/2022/2845221 | верно: Francis Oketch Ochieng, Mathew Kinyanjui и др., 2022 |
| 64 | Ochieng, R., others. (2023). Numerical Study of Wax Deposition from Multiphase Flow in Oil Pipelines with Heat and Mass Transfer. Mathematic | 10.1155/2023/1173505 | верно: Francis Oketch Ochieng, Mathew Kinyanjui и др., 2023 |
| 65 | Pan, Huanquan, Firoozabadi, Abbas, Fotland, Per. (1997). Pressure and Composition Effect on Wax Precipitation: Experimental Data and Model R | 10.2118/36740-pa | верно: Huanquan Pan, Abbas Firoozabadi и др., 1997 |
| 66 | B. Soltani, S. Esteghamat, R. Kharrat. (2017). Asphaltene Precipitation Modeling by Using of PC-SAFT Equation of State. Proceedings. | 10.3997/2214-4609.201701511 | верно: Bahram Soltani, S. Esteghamat и др., 2017 |
| 67 | M. Masoudi, S. Parvin, R. Miri, S. Kord, H. Hellevang. (2021). Implementation of PC-SAFT Equation of State Into MRST Compositional for Model | 10.3997/2214-4609.202011432 | верно: Mohammad Masoudi, S. Parvin и др., 2021 |
| 68 | Ali Kariman Moghaddam, Saeid Jamshidi. (2022). Performance evaluation and improvement of PC-SAFT equation of state for the asphaltene precip | 10.1016/j.fluid.2021.113340 | верно: Ali Kariman Moghaddam, Saeid Jamshidi, 2021; год в записи 2022, по OpenAlex 2021 |
| 69 | Mohammad Tavakkoli, Andrew Chen, Francisco M. Vargas. (2016). Rethinking the modeling approach for asphaltene precipitation using the PC-SAF | 10.1016/j.fluid.2015.11.003 | верно: Mohammad Tavakkoli, Andrew Chen и др., 2015; год в записи 2016, по OpenAlex 2015 |
| 70 | Pedersen, Karen Schou. (1995). Prediction of Cloud Point Temperatures and Amount of Wax Precipitation. SPE Production & Facilities, 10, 46-4 | 10.2118/27629-pa | верно: Karen Schou Pedersen, 1995 |
| 71 | (2025). Review of physics-informed machine learning (PIML) methods applications in subsurface engineering. Geoenergy Science and Engineering | 10.1016/j.geoen.2025.213713 | верно: Utkarsh Sinha, Birol Dindoruk, 2025 |
| 72 | Yutaka Onaka, Kozo Sato. (2021). Dynamics of pore-throat plugging and snow-ball effect by asphaltene deposition in porous media micromodels. | 10.1016/j.petrol.2021.109176 | верно: Yutaka Onaka, Kôzô Satô, 2021 |
| 73 | H. N. Dunning, J. W. Moore, Herman Bieber, R. B. Williams. (1960). Porphyrin, Nickel, Vanadium, and Nitrogen in Petroleum. Journal of Chemic | 10.1021/je60008a036 | верно: H. Neal Dunning, J. W. Moore и др., 1960 |
| 74 | (2018). Prediction of wax content in crude oil and petroleum fraction by proton NMR. Petroleum Science and Technology. | 10.1080/10916466.2018.1536713 | верно: Himanshu Saxena, Arakshita Majhi и др., 2018 |
| 75 | (2019). Modeling Wax Disappearance Temperature Using Advanced Intelligent Frameworks. Energy & Fuels. | 10.1021/acs.energyfuels.9b03296 | верно: Chahrazed Benamara, Menad Nait Amar и др., 2019 |
| 76 | Riazi, Mohammad R., Daubert, Thomas E. (1987). Characterization parameters for petroleum fractions. Industrial & Engineering Chemistry Resea | 10.1021/ie00064a023 | верно: Mohammad R. Riazi, Thomas E. Daubert, 1987 |
| 77 | Ring, Jasper N., Wattenbarger, R. A., Keating, James F., Peddibhotla, Sriram. (1994). Simulation of Paraffin Deposition in Reservoirs. SPE P | 10.2118/24069-pa | верно: Jasper N. Ring, Robert A. Wattenbarger и др., 1994 |
| 78 | (2021). Predicting wax deposition using robust machine learning techniques. South African Journal of Chemical Engineering. | 10.1016/j.sajce.2021.02.003 | **ОШИБКА**: DOI из full_review (№ 78) ведет на статью об адсорбции фенола; верная работа - Nait Amar, Jahanbani Ghahfarokhi 2021, 10.1016/j.petlm.2021.07.005 (включена) |
| 79 | ACS Omega authors. (2025). ACS Omega. | 10.1021/acsomega.5c00439 | верно: Ibrahim Atwah, Maram AlSaif и др., 2025 |
| 80 | M. Jamialahmadi, K. Ahmadi. (2003). A New Scaling Equation for Modeling of Asphaltene Precipitation. Nigeria Annual International Conference | 10.2118/85673-ms | верно: M. Jamialahmadi, K. Ahmadi, 2003 |
| 81 | Singh, Probjot, Venkatesan, R., Fogler, H. Scott, Nagarajan, Nagi. (2000). Formation and aging of incipient thin film wax‐oil gels. AIChE Jo | 10.1002/aic.690460517 | верно: Probjot Singh, Ramachandran Venkatesan и др., 2000 |
| 82 | Singh, Probjot, Venkatesan, R., Fogler, H. Scott, Nagarajan, Nagi. (2001). Morphological evolution of thick wax deposits during aging. AIChE | 10.1002/aic.690470103 | верно: Probjot Singh, Ramachandran Venkatesan и др., 2001 |
| 83 | Soulgani, Bahram Soltani, Tohidi, Bahman, Jamialahmadi, M., Rashtchian, Davood. (2011). Modeling Formation Damage due to Asphaltene Depositi | 10.1021/ef101195a | верно: Bahram Soltani Soulgani, Bahman Tohidi и др., 2011 |
| 84 | Ana C. R. Sodero, Hugo Santos Silva, Patricia Guevara Level, Brice Bouyssiere, Jean-Pierre Korb, Hervé Carrier, Ahmad Alfarra, Didier Bégué, | 10.1021/acs.energyfuels.6b00757 | верно: Ana Carolina Rennó Sodero, Hugo Santos Silva и др., 2016 |
| 85 | Narmadha Rajan Babu, Pei-Hsuan Lin, Mohammed I. L. Abutaqiya, Caleb J. Sisco, Jianxin Wang, Francisco M. Vargas. (2019). Systematic Investig | 10.1021/acs.energyfuels.8b03239 | верно: Narmadha Rajan Babu, Pei‐Hsuan Lin и др., 2018; год в записи 2019, по OpenAlex 2018 |
| 86 | Taheri-Shakib, Jaber, Shekarifard, Ali, Naderi, Hassan. (2018). Experimental investigation of the asphaltene deposition in porous media: Acc | 10.1016/j.petrol.2018.01.017 | верно: Jaber Taheri-Shakib, Ali Shekarifard и др., 2018 |
| 87 | Mark Khait, Denis Voskov. (2019). Integrated Framework for Modelling of Thermal-Compositional Multiphase Flow in Porous Media. SPE Reservoir | 10.2118/193932-ms | верно: Mark Khait, Denis Voskov, 2019 |
| 88 | E. Rogel. (2004). Thermodynamic Modeling of Asphaltene Aggregation. Langmuir, 20, 1003-1012. | 10.1021/la035339o | верно: Estrella Rogel, 2004 |
| 89 | Yurko Duda, C. Lira-Galeana. (2006). Thermodynamics of asphaltene structure and aggregation. Fluid Phase Equilibria, 241, 257-267. | 10.1016/j.fluid.2005.12.043 | верно: Yurko Duda, C. Lira-Galeana, 2006 |
| 90 | Victorov, Alexey I., Firoozabadi, Abbas. (1996). Thermodynamic micellizatin model of asphaltene precipitation from petroleum fluids. AIChE J | 10.1002/aic.690420626 | верно: Alexey I. Victorov, Abbas Firoozabadi, 1996 |
| 91 | Wang, Meng, Hao, Yifan, Islam, Md Rashedul, Chen, Chau‐Chyun. (2016). Aggregation thermodynamics for asphaltene precipitation. AIChE Journal | 10.1002/aic.15173 | верно: Meng Wang, Yifan Hao и др., 2016 |
| 92 | (2025). Toward accurate modeling of wax appearance temperature (WAT) using intelligent ensemble techniques: Application to heavy crude oils. | 10.1016/j.molliq.2025.128500 | **ОШИБКА**: DOI из full_review (№ 92) - исправление статьи о ванадате висмута; верная работа - Yadav et al. 2025, Chem. Eng. Sci., 10.1016/j.ces.2025.122296 (включена) |
| 93 | (2024). New intelligent models for predicting wax appearance temperature using experimental data - Flow assurance implications. Fuel. | 10.1016/j.fuel.2024.133423 | **ОШИБКА**: DOI из full_review (№ 93) ведет на статью о фурфуроле; верная работа - Mahmoudi Kouhi et al. 2024, 10.1016/j.fuel.2024.133146 (включена) |
| 94 | Xiangrong Nie, Shenglai Yang. (2014). Numerical simulation of paraffin wax deposition in waxy crude oil reservoir with cold water flooding.  | 10.1080/12269328.2014.959623 | верно: Xiangrong Nie, Shenglai Yang, 2014 |
| 95 | John A. Svendsen. (1993). Mathematical modeling of wax deposition in oil pipeline systems. AIChE Journal, 39, 1377-1388. | 10.1002/aic.690390815 | верно: John Arild Svendsen, 1993 |
| 96 | (2024). Wax Deposition during the Transportation of Waxy Crude Oil: Mechanisms, Influencing Factors, Modeling, and Outlook. Energy & Fuels. | 10.1021/acs.energyfuels.3c04687 | верно: Haoran Zhu, Yun Lei и др., 2024 |
| 97 | (2022). Crude oil wax: A review on formation, experimentation, prediction, and remediation techniques. Petroleum Science, 19, 2214-2237. | 10.1016/j.petsci.2022.08.008 | верно: Wyclif Kiyingi, Jixiang Guo и др., 2022 |
| 98 | (2020). Rheology of waxy crude oils in relation to restart of gelled pipelines. Chemical Engineering Science. | 10.1016/j.ces.2019.115270 | **ОШИБКА**: DOI из full_review (№ 98) ведет на статью об ионных жидкостях; верная работа - Fakroun, Benkreira 2019, 10.1016/j.ces.2019.115212 (включена) |
| 99 | (2008). Time-Dependent Rheology of a Model Waxy Crude Oil with Relevance to Gelled Pipeline Restart. Energy & Fuels. | 10.1021/ef800628g | верно: Jules John Magda, Husam El‐Gendy и др., 2008 |
| 100 | (2023). Development of a New Model for the Formation of Wax Deposits through the Passage of Crude Oil within the Well. Sustainability, 15(12 | 10.3390/su15129616 | верно: Pavel Ilushin, Kirill A. Vyatkin и др., 2023 |
| 101 | Weingarten, J. S., Euchner, Julie. (1988). Methods for Predicting Wax Precipitation and Deposition. SPE Production Engineering, 3, 121-126. | 10.2118/15654-pa | верно: J. S. Weingarten, Julie Euchner, 1988 |
| 102 | Whitson, Curtis H. (1983). Characterizing Hydrocarbon Plus Fractions. Society of Petroleum Engineers Journal, 23, 683-694. | 10.2118/12233-pa | верно: Curtis Hays Whitson, 1983 |
| 103 | Won, K.W. (1986). Thermodynamics for solid solution-liquid-vapor equilibria: wax phase formation from heavy hydrocarbon mixtures. Fluid Phas | 10.1016/0378-3812(86)80061-9 | верно: K.W. Won, 1986 |
| 104 | Wu, Jianzhong, Prausnitz, John M., Firoozabadi, Abbas. (1998). Molecular‐thermodynamic framework for asphaltene‐oil equilibria. AIChE Journa | 10.1002/aic.690440516 | верно: Jianzhong Wu, John M. Prausnitz и др., 1998 |
| 105 | Wu, Jianzhong, Prausnitz, John M., Firoozabadi, Abbas. (2000). Molecular thermodynamics of asphaltene precipitation in reservoir fluids. AIC | 10.1002/aic.690460120 | верно: Jianzhong Wu, John M. Prausnitz и др., 2000 |
