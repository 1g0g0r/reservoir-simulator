# Систематический литературный обзор: Математическое моделирование осаждения парафинов, асфальтенов и других тяжёлых компонентов нефти при неизотермической фильтрации в пористой среде

*Подготовлено: 2026-06-27 · Проработано источников: ~60 · Полнотекстовое чтение: ключевые открытые статьи · Уровень достоверности: высокий по международной рецензируемой литературе; средний по части библиографических деталей русскоязычных монографий и по точным числам цитирований (Google Scholar не опрашивался — оценки значимости даны качественно и помечены).*

> **Методологическая оговорка.** Поиск выполнен инструментами WebSearch/WebFetch (MCP-инструменты Firecrawl/Exa в данной среде не сконфигурированы). Точные числа цитирований не извлекались автоматически; значимость источников оценена качественно по их роли в литературе (частота ссылок в обзорах, статус «канонической» модели). Имена первых авторов для двух полнотекстово недоступных статей (Sci. Rep. Changchunling; Geosystem Engineering 2014) рекомендуется сверить перед цитированием — они помечены [⚠ сверить].

---

## Содержание
1. Термодинамика осаждения парафинов (SLE, WAT, multi-solid / solid-solution, EOS, характеризация)
2. Кинетика и механизмы осаждения (диффузия, сдвиг, броуновское движение, гравитация; рост и старение)
3. Связанные неизотермические модели в пористой среде (неразрывность + Дарси + энергия + фазовый переход)
4. Повреждение пласта (снижение φ и k, обратная связь с течением и температурой)
5. Моделирование асфальтенов (осаждение/кольматация/вынос; термо-баро-композиционная связь)
6. Численные методы и симуляторы
7. Экспериментальная валидация и промысловые кейсы
8. Современные направления (2018–2025): композиционное моделирование, ML, гибриды
- Сводная оценка ключевых источников
- BibTeX
- Пробелы и противоречия
- Выводы для диссертации

---

## 1. Термодинамика осаждения парафинов

### Физика процесса
Парафины — это насыщенные н-алканы (преим. C18–C60+), кристаллизующиеся при охлаждении нефти ниже **температуры начала кристаллизации (Wax Appearance Temperature, WAT / cloud point)**. Термодинамическая задача — описание равновесия «жидкость—твёрдое» (Solid–Liquid Equilibrium, SLE): какие компоненты и в каком количестве переходят в твёрдую фазу при заданных (T, p, состав). Базовое условие равновесия — равенство фугитивностей компонента i в жидкой и твёрдой фазах: f_i^L(T,p,x) = f_i^S(T,p,z). Растворимость падает с понижением T; при разгазировании (потере лёгких н-алканов-растворителей) WAT **повышается**.

### Две парадигмы описания твёрдой фазы
**(а) Модели твёрдого раствора (Solid-Solution, SS)** — все компоненты взаимно растворимы в одной твёрдой фазе; коэффициенты активности по теории регулярных или полимерных растворов (Флори—Хаггинса). Схема EOS+G^E: кубическое УРС (PR/SRK) для пар/жидкость + G^E-модель для SLE.
- **Won (1986)** — первая инженерная SS-модель (модифицированный регулярный раствор) для выпадения парафина из тяжёлых УВ-смесей ([контекст, RG](https://www.researchgate.net/publication/238639852_Thermodynamic_Modeling_of_Wax_Precipitation_in_Crude_Oils1)).
- **Hansen et al. (1988)** и **Pedersen et al. (1991)** — модели на базе теории полимерных растворов (Флори); Pedersen ввёл эмпирическую поправку, ограничивающую долю компонента, способного перейти в твёрдую фазу ([контекст, Springer](https://link.springer.com/article/10.1007/s12182-015-0071-4)).

**(б) Многотвёрдофазная модель (Multi-Solid, MS)** — каждый выпадающий компонент образует **отдельную чистую несмешивающуюся** твёрдую фазу; число твёрдых фаз определяется критерием стабильности.
- **Lira-Galeana, Firoozabadi, Prausnitz (1996)** — каноническая MS-модель; пар/жидкость по PR-EOS, характеризация фракций по Уитсону ([обзор, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0009250901003384)). Считается эталоном точности для тяжёлых фракций.

**(в) Предиктивные локально-композиционные модели**
- **Coutinho (1998), Predictive UNIQUAC** — предсказывает расщепление твёрдой фазы на несколько сосуществующих **твёрдых растворов**, устраняя переоценку выпадения лёгких алканов, характерную для Wilson/модиф. UNIQUAC ([ACS](https://pubs.acs.org/doi/full/10.1021/ie980340h); [новый pred. UNIQUAC, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0378381206002743)).

**(г) Уравнения состояния семейства SAFT**
- **PC-SAFT** — современный подход для совместного описания VLE/SLE, WAT и массы парафина; эффективен при характеризации тяжёлого «хвоста» ([PC-SAFT для парафина, Springer](https://link.springer.com/article/10.1007/s12182-015-0071-4); [MS + PC-SAFT, Fuel 2021](https://www.sciencedirect.com/science/article/abs/pii/S0016236121020810)).

### Характеризация состава
Ключевой и наиболее чувствительный этап — разбиение C7+/C30+ «хвоста» на псевдокомпоненты (метод **Whitson**, гамма-распределение), задание корреляций T_пл, ΔH_пл н-алканов. От характеризации сильнее всего зависит предсказанная WAT ([WDT по MS-модели, Springer](https://link.springer.com/article/10.1007/s13202-018-0480-1)).

### Противоречие
SS-модели склонны переоценивать выпадение лёгких компонентов; MS-модели точнее для тяжёлых фракций, но дают ступенчатую (негладкую) кривую выпадения; predictive UNIQUAC компромиссен. Универсального консенсуса нет — выбор модели зависит от типа флюида ([Fuel 2021](https://www.sciencedirect.com/science/article/abs/pii/S0016236121020810)).

---

## 2. Кинетика и механизмы осаждения

### Физика: четыре механизма переноса к холодной поверхности/стенке поры
1. **Молекулярная диффузия** парафина по градиенту концентрации, индуцированному радиальным градиентом температуры (закон Фика; J = -ρ D_wo dC/dr·dC/dT). Признан **доминирующим** для большинства условий.
2. **Сдвиговая дисперсия** — перенос кристаллов сдвиговым потоком.
3. **Броуновская диффузия** взвешенных кристаллов.
4. **Гравитационное оседание** более плотных кристаллов.

**Основополагающая работа — Burger, Perkins, Striegler (1981)** (Trans-Alaska Pipeline): отложение объяснено суммой молекулярной диффузии, сдвиговой дисперсии и броуновской диффузии ([JPT, OnePetro](https://www.onepetro.org/journal-paper/SPE-8788-PA)).
**Brown, Niesen, Erickson (1993)** экспериментально показали несущественность сдвиговой дисперсии и отсутствие отложения при нулевом тепловом потоке — это закрепило молекулярную диффузию как главный механизм ([контекст-обзор](https://www.sciencedirect.com/science/article/abs/pii/S0016236123022901)). **Это явное противоречие в литературе** (Burger vs. Brown) по роли сдвиговой дисперсии — отмечается до сих пор.

### Модели роста и старения отложения
- **Singh, Venkatesan, Fogler, Nagarajan (2000, 2001)** — модель Мичигана: радиальная диффузия молекул парафина в гель вызывает одновременный **рост** и **старение (aging)** отложения за счёт встречной диффузии (counter-diffusion) внутрь геля и роста доли твёрдой фазы; вводят понятие критической углеродной длины (critical carbon number) ([AIChE J. 2000, Deep Blue](https://deepblue.lib.umich.edu/handle/2027.42/34241); [AIChE J. 2001, Wiley](https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.690470103)).
- **Huang, Lee, Senra, Fogler (2011)** — фундаментальная модель совместного тепло- и массопереноса, прогноз толщины и доли парафина (ламинар/турбулент) ([PDF, Deep Blue](https://deepblue.lib.umich.edu/bitstream/handle/2027.42/87116/12517_ftp.pdf)).
- **Ramirez-Jaramillo, Lira-Galeana, Manero (2004)** — многокомпонентная гидродинамическая модель, связка SLE + неньютоновское течение; молекулярная диффузия через пограничный слой ([RG](https://www.researchgate.net/publication/232874607_Modeling_Wax_Deposition_in_Pipelines)).
- **Correra et al. (2007)** — уточнение коэффициента диффузии парафина в температурном градиенте ([ZAMM, Wiley](https://onlinelibrary.wiley.com/doi/10.1002/zamm.200510293)).

---

## 3. Связанные неизотермические модели в пористой среде

### Каноническая система уравнений
Совместно решаются:

**(1) Сохранение массы компонента i (многофазное течение):**
∂/∂t(φ Σ_p ρ_p S_p x_{i,p}) + ∇·(Σ_p ρ_p x_{i,p} **u**_p) = q_i − R_i^{dep}
где R_i^{dep} — источниковый член осаждения/выпадения (связь с блоками 1, 2, 4).

**(2) Закон Дарси (для каждой фазы p = oil, water, gas):**
**u**_p = −(k·k_{rp}/μ_p)(∇p_p − ρ_p g ∇D)

**(3) Уравнение энергии (тепломассоперенос):**
∂/∂t[(1−φ)ρ_r c_r T + φ Σ_p ρ_p S_p û_p] + ∇·(Σ_p ρ_p h_p **u**_p) = ∇·(λ_eff ∇T) + Q_{JT} + Q_{cryst} + q_H
- λ_eff — эффективная теплопроводность скелет+флюид;
- **Q_{JT}** — эффект Джоуля—Томсона (дросселирование при ∇p);
- **Q_{cryst}** — тепловыделение при кристаллизации парафина: Q_{cryst} = L_{cryst}·(∂m_s/∂t), где L_{cryst} — скрытая теплота, m_s — масса выпавшей твёрдой фазы (источник связи термики и фазового перехода);
- адиабатический эффект и теплота разгазирования учитываются аналогичными членами ([многофазные течения, oilgasjournal](http://oilgasjournal.ru/issue_21/chetvertushkin-lyupa-trapeznikova.pdf)).

**(4) Замыкание — термодинамика выпадения** из §1: m_s = m_s(T, p, состав), WAT(T,p); ниже WAT включается фазовый переход.

### Допущения (типовые)
Локальное термическое равновесие скелет↔флюид; несжимаемая порода или слабая пороупругость; постоянные/слабозависящие теплофизические свойства; мгновенное термодинамическое равновесие выпадения (или кинетика 1-го порядка). Часть работ снимает допущение о равновесии (кинетика осаждения, §4–5).

### Ключевые источники (методология)
- **Banki, Hoteit, Firoozabadi (2008)** — ⭐ перенос на пористую среду: поток диффузии в неизотермике через **термодинамику необратимых процессов**; **enthalpy–porosity** подход к фазовому переходу; закупорка — через кинетическое уравнение пористости. Методологически наиболее близок к задаче «выпадение + неизотермика + изменение проницаемости» ([Int. J. Heat Mass Transfer, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0017931007006722)).
- **Российская школа:** Чекалюк (1965, термодинамика пласта) ([GeoKniga](https://www.geokniga.org/books/4848)); Алишаев, Розенберг, Теслюк (1985, неизотермическая фильтрация); автомодельные задачи неизотермической двухфазной фильтрации ([dissercat](https://www.dissercat.com/content/avtomodelnye-zadachi-neizotermicheskoi-dvukhfaznoi-filtratsii-i-teplovogo-pogranichnogo-sloy)); фильтрация с фазовыми переходами ([dissercat](https://www.dissercat.com/content/issledovanie-protsessov-neizotermicheskoi-filtratsii-zhidkosti-i-gaza-s-fazovymi-perekhodami)); вязкопластичные нефти при термозаводнении ([Вестник АГТУ](https://vestnik.astu.org/ru/nauka/article/71928/view)).
- **Аналогия фазового перехода:** обзор неизотермической фильтрации с разложением газогидрата ([Mathematics, MDPI 10(24):4674](https://www.mdpi.com/2227-7390/10/24/4674)) — образцовая структура связанной системы (гидрат ↔ парафин формально близки).

---

## 4. Повреждение пласта: снижение проницаемости и пористости

### Физика
Выпавшая твёрдая фаза откладывается на поверхности пор (постепенно) и закупоривает поровые каналы (резко), снижая φ и k и изменяя относительные фазовые проницаемости (ОФП), смачиваемость и вязкость. Обратная связь: ↓k → ↑∇p → ↑Q_{JT} и изменение поля T → новое выпадение.

### Базовые модели связи φ↔k
- **Уравнение Козени—Кармана** и его модификации (Civan): k = k_0 (φ/φ_0)^n, n≈3, с поправкой на структуру осадка.
- **Civan, Reservoir Formation Damage (3-е изд., 2016)** — энциклопедический свод: кинетика осаждения, модифиц. Козени—Карман, три механизма (поверхностное осаждение, закупорка горловин, реэнтрейнмент) ([книга, vdoc.pub](https://vdoc.pub/documents/reservoir-formation-damage-fundamentals-modeling-assessment-and-mitigation-7d5nok4gg810)).
- **Civan (2001)** — кинетическая модель масштабного эффекта φ–k ([AIChE J., Wiley](https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.690470206)).
- **Gruesbeck, Collins (1982)** — модель «параллельных путей»: среда делится на мелкие каналы (пробковая закупорка) и крупные (поверхностное непробковое осаждение); фундамент для всех последующих deposition-моделей ([контекст](https://www.sciencedirect.com/science/article/abs/pii/S001623611831278X)).

### Применительно к парафину в пласте
- **He et al. [⚠ сверить], Scientific Reports 10:14595 (2020)** — Changchunling: уравнение фазового превращения + двухфазная нефть-вода фильтрация; эксперимент показывает резкое падение радиуса горловин и ОФП с понижением T ([Nature](https://www.nature.com/articles/s41598-020-71065-z)).
- **[⚠ сверить] (2014), Geosystem Engineering 17(4)** — холодное заводнение высокопарафинистой залежи; фронт «холодового повреждения» смещается вглубь пласта со временем ([T&F](https://www.tandfonline.com/doi/abs/10.1080/12269328.2014.959623)).

---

## 5. Моделирование асфальтенов

### Физика и термодинамика выпадения (4 конкурирующих парадигмы)
Асфальтены — наиболее полярные, высокомолекулярные компоненты; выпадают преимущественно при **снижении давления** к давлению насыщения (максимум выпадения ≈ давление насыщения) и при закачке газа/CO₂. Четыре школы термодинамики:
1. **Растворная (Флори—Хаггинса), обратимая — Hirschberg et al. (1984):** асфальтены растворены; выпадение реверсивно при изменении растворяющей способности нефти ([контекст-обзор, MDPI Processes 11(3):765](https://www.mdpi.com/2227-9117/11/3/765)).
2. **Коллоидная — Leontaritis & Mansoori (1987):** асфальтены — твёрдые коллоидные частицы, стабилизированные адсорбированными смолами; выпадение чаще **необратимо** ([SPE 16258, RG](https://www.researchgate.net/publication/237196338_Asphaltene_Flocculation_During_Oil_Recovery_and_Processing_A_Thermodynamic-Colloidal_Model)). **Противоречие с (1)** по обратимости — ключевой нерешённый спор.
3. **Термодинамика мицеллообразования — Victorov & Firoozabadi (1996); Pan & Firoozabadi (1998/2000):** асфальтен-мицеллы со смолами на оболочке; выпадение как жидкость-жидкостное равновесие (тяжёлая фаза = асфальтены+смолы) + PR-EOS для мономеров ([AIChE J., Wiley](https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.690420626); [SPE 38857, OnePetro](https://onepetro.org/SPEATCE/proceedings-abstract/97SPE/97SPE/SPE-38857-MS/189120)).
4. **Твёрдофазная (solid model) — Nghiem et al. (1993):** осадок = два «твёрдых»: Solid 1 (обратимо равновесен с асфальтеном в нефти), Solid 2 (образуется из Solid 1 по реакции флокуляции — обратимой/необратимой); тяжёлая фракция C31+ делится на непреципитирующую и преципитирующую части ([контекст, IECR](https://pubs.acs.org/doi/10.1021/ie990781g)). Доминирует в коммерческих симуляторах.

### Кинетика осаждения/кольматации/выноса в пористой среде
- **Wang & Civan (трёхчленная модель)** — суммарное осаждение = поверхностное осаждение + закупорка горловин + (вычитается) вынос (entrainment):
  **∂E/∂t = α·C·φ − β·E·(v − v_cr) + γ(1 + σE)·u·C**
  - α — поверхностное осаждение (∝ концентрации взвеси C);
  - β — вынос (реэнтрейнмент), активен при v > v_cr (критическая поровая скорость);
  - γ — закупорка горловин (∝ скорости u), σ — «снежный» коэффициент лавинного роста.
  Затем **φ = φ₀ − E**, **k = k₀(φ/φ₀)³** ([полнотекст: Tabzar et al., OGST 73 (2018)](https://ogst.ifpenergiesnouvelles.fr/articles/ogst/full_html/2018/01/ogst180048/ogst180048.html)).
- **Tabzar et al. (2018)** — открытая статья с **полной системой** (4 связанных баланса масс в цилиндрических координатах: нефть, газ, вода, асфальтен; уравнение (3) — кинетика; (4)–(5) — φ,k; (1) — фугитивность твёрдого по Nghiem; решение Ньютона—Рафсона в MATLAB). Образцовый шаблон сопряжённой модели ([OGST 2018](https://ogst.ifpenergiesnouvelles.fr/articles/ogst/full_html/2018/01/ogst180048/ogst180048.html)).
- **Коллоидный подход к моделированию осаждения** ([OGST 2008, PDF](https://ogst.ifpenergiesnouvelles.fr/articles/ogst/pdf/2008/01/ogst07068.pdf)); **новые модели pore-blocking** ([JPSE 2019](https://www.sciencedirect.com/science/article/pii/S0920410519309337); [Fuel 2019](https://www.sciencedirect.com/science/article/abs/pii/S001623611831278X)).

### Термо-баро-композиционная связь
Выпадение асфальтенов управляется (p, T, состав): депрессуризация, закачка CO₂/растворителя, изменение T. В компонентном симуляторе deposition-член связывается с flash-расчётом (3-фазное равновесие) и обновлением свойств флюида и породы ([3-фазное равновесие, JPSE 2016](https://www.sciencedirect.com/science/article/pii/S092041051630359X)).

---

## 6. Численные методы и симуляторы

### Дискретизация и схемы
- **Метод конечных объёмов (FVM)** — преобладает в пластовой симуляции (локальная консервативность); скалярные переменные (p, S_p, концентрации) дискретизируются по FVM; двухточечная (TPFA) или многоточечная (MPFA) аппроксимация потока для неортогональных сеток и полного тензора проницаемости ([полунеявный FVM, Fluids/MDPI 6(10):341](https://www.mdpi.com/2311-5521/6/10/341)).
- **Конечные элементы / смешанно-гибридные FE** — для сложной геометрии/тензоров ([mixed-hybrid FE, J. Comput. Phys. 2017](https://www.sciencedirect.com/science/article/abs/pii/S0021999117304874)).
- **Схемы связывания:** полностью неявная (FIM, Ньютон—Рафсон, безусловно устойчива), IMPES/IMPEC, IMPESC (давление неявно, составы явно), последовательно-неявная.

### Симуляторы выпадения/осаждения
- **Полностью неявный компонентный симулятор осаждения асфальтена при истощении** — одновременное решение уравнений фазового равновесия, объёмного ограничения, переноса компонентов, multiphase flash и deposition ([Fluid Phase Equilibria 2015](https://www.sciencedirect.com/science/article/abs/pii/S0378381215001831)).
- **Параллельный FIM EOS-симулятор осаждения асфальтена** ([SPE-120203-STU](https://onepetro.org/SPEATCE/proceedings-abstract/08ATCE/08ATCE/SPE-120203-STU/145597)).
- **Nghiem & Coombe / Qin et al. (2000)** — реализация solid-model в компонентной симуляции ([IECR, ACS](https://pubs.acs.org/doi/10.1021/ie990781g); [SPE-89437](https://onepetro.org/SPEIOR/proceedings-abstract/04IOR/04IOR/SPE-89437-MS/71368)).
- **Коммерческие пакеты:** **CMG-GEM** (компонентный, модуль асфальтенов; deposition/flocculation/precipitation — [CMG-GEM кейс, RG](https://www.researchgate.net/publication/267869547_Compositional_simulation_of_deposition_flocculation_and_precipitation_of_Asphaltene_in_oil_reservoir_using_CMG-GEM)); **CMG-STARS** (тепловой); **PVTsim Nova / Multiflash** (термодинамика парафина/асфальтена, тюнинг к эксперименту, экспорт в OLGA/PIPESIM — [Calsep](https://calsep.com/pvtsim-nova/find-a-pvtsim-package/flow-assurance/)); **OLGA** (динамическое многофазное течение, риск отложений в трубопроводах — [SLB](https://www.slb.com/products-and-services/delivering-digital-at-scale/software/olga/olga-dynamic-multiphase-flow-simulator)); **PIPESIM**, **Multiflash Wax** (использован в [Кищенко и др., 2018](https://cyberleninka.ru/article/n/modelirovanie-protsessa-obrazovaniya-organicheskih-otlozheniy-parafinovogo-tipa-pri-ekspluatatsii-skvazhin-elektrotsentrobezhnymi)).

> Замечание: специализированных **пластовых** симуляторов именно **парафиноотложения** (в отличие от асфальтенов) в открытой литературе мало; чаще адаптируют асфальтеновый аппарат или строят исследовательские коды (см. §3 Banki et al.; §5 Tabzar et al.).

---

## 7. Экспериментальная валидация и промысловые кейсы

### Методы измерения WAT и кинетики
- **DSC (дифференциальная сканирующая калориметрия)** — по экзотермическому пику фиксируют WAT и массу выпавших кристаллов vs. T.
- **Кросс-поляризационная микроскопия (CPM)** — детекция первых кристаллов по вращению плоскости поляризации (часто наиболее чувствительна).
- **Вискозиметрия/реометрия** — по излому кривой μ(T); **переоценивает** WAT относительно DSC.
- Сравнения методов и влияние скорости охлаждения ([DSC vs термомикроскопия vs реометрия, JPSE 2016](https://www.sciencedirect.com/science/article/pii/S0920410516304557); [реометрия/DSC/CPM, RG](https://www.researchgate.net/publication/284640242_Determination_of_Wax_CrystallizationGelation_Temperature_by_Rheometry_DSC_and_CPM)). **Противоречие методов** по абсолютному значению WAT — известный источник неопределённости при валидации.

### Установки для кинетики отложения
- **Flow loop** (петлевые стенды), **cold finger** (холодный палец), **core flooding** (фильтрация через керн) — для динамического осаждения в пористой среде ([WAT/стенды, PSL](https://psl-systemtechnik.com/en/wax-appearance-temperature-wat/)).

### Промысловые/керновые кейсы (асфальтены)
- **Core flooding:** снижение проницаемости 72–98 % при необратимом удержании асфальтенов/смол ([Permeability reduction…, RG](https://www.researchgate.net/publication/239142334_Permeability_reduction_by_asphaltenes_and_resins_deposition_in_porous_media)).
- **CO₂-заводнение, Changqing (сверхнизкая проницаемость)** — влияние режима/темпа закачки; максимум осаждения у минимального давления смешиваемости (MMP) ([Geofluids 2021, Wiley](https://onlinelibrary.wiley.com/doi/10.1155/2021/6626114)); длинный керн CO₂/flue gas ([Sci. Rep. 2024](https://www.nature.com/articles/s41598-024-54395-0)); влияние давления/расхода в песчанике ([Fuel 2021](https://www.sciencedirect.com/science/article/abs/pii/S0016236121022924)).
- **Промысловые кейсы парафина:** холодное заводнение высокопарафинистых залежей (§4); до ~75 % осложнений по АСПО (ЛУКОЙЛ-Пермь) ([dissercat](https://www.dissercat.com/content/nauchno-metodicheskie-osnovy-modelirovaniya-protsessov-upravleniya-ekspluatatsionnymi-kharak-0)).

---

## 8. Современные направления (2018–2025)

### Композиционное и многофазное моделирование
Тренд — полностью неявные **компонентные** симуляторы с 3-фазным flash и сопряжённым deposition-членом; учёт CO₂-EOR и термо-баро эффектов ([3-фазное равновесие, JPSE 2016](https://www.sciencedirect.com/science/article/pii/S092041051630359X); [эффективная реализация выпадения, 2024](https://www.sciencedirect.com/science/article/pii/S2949891024008777); [CO₂ в глубоких пластах, SPE 2023](https://onepetro.org/SPEATCE/proceedings-abstract/23ATCE/3-23ATCE/535420)).

### Машинное обучение
- **Прогноз WAT:** ANN, ансамбли, SVM, ELM, ANFIS; «интеллектуальные» модели по экспериментальным данным ([WAT по AI, AJSE/Springer](https://link.springer.com/article/10.1007/s13369-019-04290-y); [новые интеллект. модели WAT, Fuel 2024](https://www.sciencedirect.com/science/article/abs/pii/S0016236124022956); [ансамбли для тяжёлой нефти, Chem. Eng. Sci. 2025](https://www.sciencedirect.com/science/article/pii/S0009250925011170)).
- **Прогноз скорости отложения парафина:** RBF/случайный лес/XGBoost/PINN на композиционно-термодинамических данных ([ML для отложения, Earthdoc](https://www.earthdoc.org/content/papers/10.3997/2214-4609.202032021); [ML для глубоководных трубопроводов, RG](https://www.researchgate.net/publication/375012949_Development_of_Predictive_Model_for_Wax_Formation_in_Deep-water_Pipeline_Using_Machine_learning)).
- **Асфальтены:** ANFIS, RBF+GWO, ELM, MLP — кинетика агрегации и тесты осаждения ([AI агрегация асфальтенов, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10613205/)).

### Гибридные / физико-информированные подходы
- **PINN (physics-informed neural networks)** для многофазного течения в пористой среде: захват фронтов насыщенности (Buckley—Leverett), неоднородные/трещиноватые среды, mesh-free FIM-формулировки, доменная декомпозиция при разреженных данных ([PINN двухфазное течение, ScienceDirect 2024](https://www.sciencedirect.com/science/article/abs/pii/S0309170824001180); [PINN+домены, GEEN 2023](https://www.sciencedirect.com/science/article/pii/S1995822623002947); [PIAWNN, AWR 2025](https://www.sciencedirect.com/science/article/abs/pii/S030917082500288X)). Прямого применения PINN к **неизотермическому осаждению парафина/асфальтена в пласте** в открытой литературе пока почти нет — **перспективная ниша**.

---

## Сводная оценка ключевых источников

| # | Источник (авторы, год) | Тип модели / вклад | Валидация | Значимость* |
|---|---|---|---|---|
| 1 | Won (1986) | SS, регулярный раствор; первая инженерная SLE-модель парафина | лаб. SLE | Основополагающая |
| 2 | Pedersen et al. (1991) | SS, полимерный раствор + эмпирич. поправка | North Sea crudes | Высокая |
| 3 | Lira-Galeana, Firoozabadi, Prausnitz (1996) | **Multi-solid**, PR-EOS | многокомпонент. смеси | Каноническая |
| 4 | Coutinho (1998) | Predictive UNIQUAC, неск. тв. растворов | н-алкановые смеси | Высокая |
| 5 | Burger, Perkins, Striegler (1981) | Кинетика: диффузия+сдвиг+броун. | Trans-Alaska | Основополагающая |
| 6 | Singh, Venkatesan, Fogler, Nagarajan (2000/2001) | Рост+старение геля, counter-diffusion | flow loop | Каноническая |
| 7 | Huang, Lee, Senra, Fogler (2011) | Тепло+массоперенос, толщина отложения | flow loop | Высокая |
| 8 | Banki, Hoteit, Firoozabadi (2008) | Необратимая термодинамика + enthalpy-porosity (пористый слой) | числ./лаб. | Высокая (мост к пласту) |
| 9 | Civan (2016, 3-е изд.) | Свод formation damage; модиф. Козени—Карман | книга/множество кейсов | Каноническая |
| 10 | Gruesbeck, Collins (1982) | Параллельные пути: осаждение/закупорка | керн | Основополагающая |
| 11 | Hirschberg et al. (1984) | Асфальтены: Флори—Хаггинс, обратимо | PVT | Основополагающая |
| 12 | Leontaritis & Mansoori (1987) | Асфальтены: коллоидная, (не)обратимо | промысл. | Основополагающая |
| 13 | Nghiem et al. (1993) | Асфальтены: solid model (2 солида) | компонент. симуляция | Каноническая (в softwarе) |
| 14 | Pan & Firoozabadi (1998/2000) | Асфальтены: мицеллярная, LLE | PVT/высокие p,T | Высокая |
| 15 | Wang & Civan / Tabzar et al. (2018) | Сопряжённая: 3-членная кинетика + φ,k + flash | лаб. растворимость | Высокая (шаблон) |
| 16 | Sci. Rep. 10:14595 (2020) [⚠] | Парафин в пласте: фазовый переход + 2-фазн. фильтрация | эксперимент+числ. | Средняя-высокая (релевантна) |
| 17 | Чекалюк (1965); Алишаев и др. (1985) | Термодинамика пласта; неизотерм. фильтрация | аналит./числ. | Основополагающие (рус.) |

*Значимость оценена качественно (роль в литературе), без точных чисел цитирований — см. оговорку в начале.

---

## Итоговый список наиболее значимых источников (BibTeX)

```bibtex
@article{Won1986,
  author = {Won, K. W.},
  title = {Thermodynamics for solid solution-liquid-vapor equilibria: wax phase formation from heavy hydrocarbon mixtures},
  journal = {Fluid Phase Equilibria}, volume = {30}, pages = {265--279}, year = {1986}}

@article{Hansen1988,
  author = {Hansen, J. H. and Fredenslund, Aa. and Pedersen, K. S. and R{\o}nningsen, H. P.},
  title = {A thermodynamic model for predicting wax formation in crude oils},
  journal = {AIChE Journal}, volume = {34}, number = {12}, pages = {1937--1942}, year = {1988}}

@article{Pedersen1991,
  author = {Pedersen, K. S. and Skovborg, P. and R{\o}nningsen, H. P.},
  title = {Wax precipitation from North Sea crude oils. 4. Thermodynamic modeling},
  journal = {Energy \& Fuels}, volume = {5}, number = {6}, pages = {924--932}, year = {1991}}

@article{LiraGaleana1996,
  author = {Lira-Galeana, C. and Firoozabadi, A. and Prausnitz, J. M.},
  title = {Thermodynamics of wax precipitation in petroleum mixtures},
  journal = {AIChE Journal}, volume = {42}, number = {1}, pages = {239--248}, year = {1996}}

@article{Coutinho1998,
  author = {Coutinho, J. A. P.},
  title = {Predictive UNIQUAC: a new model for the description of multiphase solid-liquid equilibria in complex hydrocarbon mixtures},
  journal = {Industrial \& Engineering Chemistry Research}, volume = {37}, number = {12}, pages = {4870--4875}, year = {1998}}

@article{Burger1981,
  author = {Burger, E. D. and Perkins, T. K. and Striegler, J. H.},
  title = {Studies of wax deposition in the Trans Alaska pipeline},
  journal = {Journal of Petroleum Technology}, volume = {33}, number = {6}, pages = {1075--1086}, year = {1981}}

@inproceedings{Brown1993,
  author = {Brown, T. S. and Niesen, V. G. and Erickson, D. D.},
  title = {Measurement and prediction of the kinetics of paraffin deposition},
  booktitle = {SPE Annual Technical Conference and Exhibition}, note = {SPE-26548-MS}, year = {1993}}

@article{Singh2000,
  author = {Singh, P. and Venkatesan, R. and Fogler, H. S. and Nagarajan, N.},
  title = {Formation and aging of incipient thin film wax-oil gels},
  journal = {AIChE Journal}, volume = {46}, number = {5}, pages = {1059--1074}, year = {2000}}

@article{Singh2001,
  author = {Singh, P. and Venkatesan, R. and Fogler, H. S. and Nagarajan, N. R.},
  title = {Morphological evolution of thick wax deposits during aging},
  journal = {AIChE Journal}, volume = {47}, number = {1}, pages = {6--18}, year = {2001}}

@article{RamirezJaramillo2004,
  author = {Ram{\'i}rez-Jaramillo, E. and Lira-Galeana, C. and Manero, O.},
  title = {Modeling wax deposition in pipelines},
  journal = {Petroleum Science and Technology}, volume = {22}, number = {7-8}, pages = {821--861}, year = {2004}}

@article{Huang2011,
  author = {Huang, Z. and Lee, H. S. and Senra, M. and Fogler, H. S.},
  title = {A fundamental model of wax deposition in subsea oil pipelines},
  journal = {AIChE Journal}, volume = {57}, number = {11}, pages = {2955--2964}, year = {2011}}

@article{Banki2008,
  author = {Banki, R. and Hoteit, H. and Firoozabadi, A.},
  title = {Mathematical formulation and numerical modeling of wax deposition in pipelines from enthalpy-porosity approach and irreversible thermodynamics},
  journal = {International Journal of Heat and Mass Transfer}, volume = {51}, number = {13-14}, pages = {3387--3398}, year = {2008}}

@book{Civan2016,
  author = {Civan, F.},
  title = {Reservoir Formation Damage: Fundamentals, Modeling, Assessment, and Mitigation},
  edition = {3rd}, publisher = {Gulf Professional Publishing}, year = {2016}}

@article{Civan2001,
  author = {Civan, F.},
  title = {Scale effect on porosity and permeability: kinetics, model, and correlation},
  journal = {AIChE Journal}, volume = {47}, number = {2}, pages = {271--287}, year = {2001}}

@article{GruesbeckCollins1982,
  author = {Gruesbeck, C. and Collins, R. E.},
  title = {Entrainment and deposition of fine particles in porous media},
  journal = {Society of Petroleum Engineers Journal}, volume = {22}, number = {6}, pages = {847--856}, year = {1982}}

@article{Hirschberg1984,
  author = {Hirschberg, A. and deJong, L. N. J. and Schipper, B. A. and Meijer, J. G.},
  title = {Influence of temperature and pressure on asphaltene flocculation},
  journal = {Society of Petroleum Engineers Journal}, volume = {24}, number = {3}, pages = {283--293}, year = {1984}}

@inproceedings{LeontaritisMansoori1987,
  author = {Leontaritis, K. J. and Mansoori, G. A.},
  title = {Asphaltene flocculation during oil production and processing: a thermodynamic-colloidal model},
  booktitle = {SPE International Symposium on Oilfield Chemistry}, note = {SPE-16258-MS}, year = {1987}}

@inproceedings{Nghiem1993,
  author = {Nghiem, L. X. and Hassam, M. S. and Nutakki, R. and George, A. E. D.},
  title = {Efficient modelling of asphaltene precipitation},
  booktitle = {SPE Annual Technical Conference and Exhibition}, note = {SPE-26642-MS}, year = {1993}}

@inproceedings{PanFiroozabadi1997,
  author = {Pan, H. and Firoozabadi, A.},
  title = {Thermodynamic micellization model for asphaltene precipitation from reservoir crudes at high pressures and temperatures},
  booktitle = {SPE Annual Technical Conference and Exhibition}, note = {SPE-38857-MS}, year = {1997}}

@article{Tabzar2018,
  author = {Tabzar, A. and Fathinasab, M. and Salehi, A. and Bahrami, B. and Mohammadi, A. H.},
  title = {Multiphase flow modeling of asphaltene precipitation and deposition},
  journal = {Oil \& Gas Science and Technology}, volume = {73}, pages = {51}, year = {2018}}

@article{Qin2000,
  author = {Qin, X. and Wang, P. and Sepehrnoori, K. and Pope, G. A.},
  title = {Modeling asphaltene precipitation in reservoir simulation},
  journal = {Industrial \& Engineering Chemistry Research}, volume = {39}, number = {8}, pages = {2644--2654}, year = {2000}}

@article{ChangchunlingSciRep2020,
  author = {{[verify authors]}},
  title = {Cold damage from wax deposition in a shallow, low-temperature, and high-wax reservoir in Changchunling Oilfield},
  journal = {Scientific Reports}, volume = {10}, pages = {14595}, year = {2020}}

@book{Chekalyuk1965,
  author = {Чекалюк, Э. Б.},
  title = {Термодинамика нефтяного пласта},
  publisher = {Недра}, address = {Москва}, year = {1965}}

@book{Alishaev1985,
  author = {Алишаев, М. Г. and Розенберг, М. Д. and Теслюк, Е. В.},
  title = {Неизотермическая фильтрация при разработке нефтяных месторождений},
  publisher = {Недра}, address = {Москва}, year = {1985}}

@book{Tronov1970,
  author = {Тронов, В. П.},
  title = {Механизм образования смоло-парафиновых отложений и борьба с ними},
  publisher = {Недра}, address = {Москва}, year = {1970}}

@article{Yu2024review,
  author = {Yu, et al.},
  title = {A comprehensive review of wax deposition in crude oil systems: mechanisms, influencing factors, prediction and inhibition techniques},
  journal = {Fuel}, volume = {357}, pages = {129676}, year = {2024}}
```

> Поля, помеченные приблизительно (vol/pages/note SPE-номера, инициалы рус. авторов), сверьте по OnePetro/eLibrary/издателю перед включением в диссертацию.

---

## Пробелы и противоречия

**Противоречия в литературе (не усреднять — фиксировать):**
1. **Роль сдвиговой дисперсии** в отложении парафина: Burger et al. (1981) включают её, Brown et al. (1993) считают несущественной. Спор не закрыт.
2. **Обратимость выпадения асфальтенов:** растворная модель Hirschberg (обратимо) vs. коллоидная Leontaritis–Mansoori (часто необратимо). Современные solid/мицеллярные модели вводят частичную обратимость как параметр — консенсуса нет.
3. **Абсолютное значение WAT** зависит от метода (DSC vs CPM vs вискозиметрия дают разброс), что искажает валидацию кинетических моделей.
4. **SS vs MS термодинамика:** SS переоценивает лёгкие, MS даёт негладкую кривую; «правильный» выбор зависит от флюида.

**Пробелы (открытые задачи — релевантны диссертации):**
1. **Дефицит чисто пластовых неизотермических моделей именно парафина** (в отличие от трубопроводных и от асфальтеновых пластовых). Аппарат Banki–Hoteit–Firoozabadi (необратимая термодинамика + enthalpy-porosity) на пласт перенесён слабо.
2. **Полная связка всех блоков** (термодинамика выпадения с реальным EOS/PC-SAFT + неизотермическая многофазная фильтрация с Q_cryst + 3-членная кинетика осаждения/выноса + обратная связь φ→k и на ОФП) в одном пластовом симуляторе встречается редко.
3. **Совместный учёт неньютоновской реологии/начального градиента** в зоне геля и термодинамики выпадения.
4. **Совместное (co-precipitation) моделирование парафин+асфальтен+смолы** при неизотермике — почти не разработано (Civan рассматривал совместное осаждение как редкий случай).
5. **PINN/гибриды для неизотермического осаждения в пласте** — практически отсутствуют; ML пока в основном для прогноза WAT/скорости, а не для решения связанной краевой задачи.
6. **Верификация** на промысловых данных холодного заводнения высокопарафинистых залежей ограничена немногими кейсами (Changchunling; китайские/российские месторождения).

---

## Выводы для диссертации (архитектура модели `paraphin`)
Рекомендуемая структура сопряжённой модели:
1. **Термодинамический модуль** → m_s(T,p,состав), WAT: для парафина — MS (Lira-Galeana) или predictive UNIQUAC (Coutinho)/PC-SAFT; для асфальтена — solid model (Nghiem) либо мицеллярная (Pan–Firoozabadi).
2. **Фильтрационно-тепловой каркас** (§3): неразрывность компонентов + Дарси + энергия с Q_cryst и Q_JT; фазовый переход через enthalpy-porosity (Banki et al.).
3. **Кинетика осаждения** (§5): 3-членная модель Wang–Civan/Tabzar (поверхностное осаждение + закупорка + вынос), σ-лавинный член.
4. **Обратная связь** φ = φ₀ − E; k = k₀(φ/φ₀)ⁿ (Козени—Карман/Civan); модификация ОФП.
5. **Численная схема:** FVM + полностью неявная (Ньютон—Рафсон), локальное измельчение в ПЗП (осаждение — околоскважинный эффект).
6. **Валидация:** лаб. WAT (DSC/CPM) + core flooding + промысловый кейс холодного заводнения.

---

## Методология обзора
Выполнено ~16 поисковых запросов (англ./рус.) по 8 под-вопросам; широкий охват → углубление по канонdическим и новейшим работам; полнотекстовое чтение открытых статей (в т.ч. Tabzar et al., OGST 2018 — извлечена полная система уравнений). Перекрёстная сверка между источниками; противоречия зафиксированы явно. Числа цитирований и часть библиографических деталей рекомендуется сверить по первоисточникам.
