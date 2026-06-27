# Математическое моделирование отложения парафинов и других тяжёлых компонентов нефти при неизотермической фильтрации в пласте

**Литературный обзор**
*Подготовлено: 2026-06-27 · Источников проработано: ~45 · Уровень достоверности: высокий (по международной литературе), средний (по отдельным библиографическим деталям русскоязычных монографий — рекомендуется сверить том/страницы по каталогам РГБ/eLibrary).*

---

## Аннотация (Executive Summary)

Проблема выпадения и отложения парафинов, смол и асфальтенов (асфальтосмолопарафиновые отложения, АСПО) при движении нефти в пористой среде представляет собой связанную термо-гидро-химическую задачу. По данным эксплуатации (например, ЛУКОЙЛ-Пермь) АСПО составляют до ~75 % всех осложнений и затрагивают тысячи скважин ([dissercat](https://www.dissercat.com/content/nauchno-metodicheskie-osnovy-modelirovaniya-protsessov-upravleniya-ekspluatatsionnymi-kharak-0)). Математическое описание этого процесса строится на трёх взаимосвязанных блоках: (1) **термодинамика фазового равновесия** «жидкость—твёрдая фаза», определяющая температуру насыщения нефти парафином (Wax Appearance Temperature, WAT) и массу выпавшей твёрдой фазы; (2) **кинетика/перенос** (молекулярная диффузия, сдвиговая дисперсия, гелеобразование), описывающие рост отложения; (3) **неизотермическая многофазная фильтрация** с уравнениями сохранения массы компонентов и энергии, на которой «навешивается» модель повреждения коллектора (снижение пористости и проницаемости). Подавляющее большинство зрелых моделей разработано для **трубопроводов и стволов скважин**; моделирование непосредственно в **пласте/призабойной зоне** существенно менее развито и составляет актуальную нишу для диссертационного исследования. Ниже систематизированы ключевые школы, модели и наиболее значимые источники.

---

## 1. Постановка проблемы: пласт vs. скважина vs. трубопровод

Принципиально различают три области, в которых происходит выпадение твёрдой фазы, и в которых физика, а значит и математические модели, отличаются:

- **Пласт и призабойная зона (ПЗП).** Фильтрация при повышенных скоростях, градиентах давления и температуры; снижение температуры до WAT приводит к образованию кристаллов, соизмеримых с размером пор, создающих дополнительное фильтрационное сопротивление и снижающих фазовую проницаемость. Главные движущие факторы — снижение давления (разгазирование, повышающее WAT) и охлаждение пласта при закачке холодного агента ([neftegaz.ru](https://magazine.neftegaz.ru/articles/dobycha/747854-mekhanizmy-obrazovaniya-asfaltosmoloparafinovykh-otlozheniy-metodiki-issledovaniya-/)).
- **Ствол скважины.** Профили T и p по стволу, многофазное течение, межфазный массоперенос ([Кищенко и др., 2018](https://cyberleninka.ru/article/n/modelirovanie-protsessa-obrazovaniya-organicheskih-otlozheniy-parafinovogo-tipa-pri-ekspluatatsii-skvazhin-elektrotsentrobezhnymi)).
- **Промысловый/подводный трубопровод.** Наиболее изученный случай; именно здесь развиты диффузионно-тепловые модели роста отложения.

Ключевой вывод обзора: модели для пласта необходимо **сопрягать** термодинамику выпадения с неизотермической фильтрацией и моделью повреждения коллектора — в отличие от трубопроводных моделей, где доминирует одномерный тепломассоперенос у стенки.

---

## 2. Физико-химические основы и механизмы отложения

**Состав и природа.** АСПО — смесь высокомолекулярных н-алканов (парафины, преим. C18–C60+), смол и асфальтенов. Двумя необходимыми условиями выпадения парафиновых АСПО являются присутствие парафиновых веществ и снижение температуры потока до **температуры насыщения нефти парафином (WAT)**, при которой появляются первые кристаллы ([neftegaz.ru](https://magazine.neftegaz.ru/articles/dobycha/747854-mekhanizmy-obrazovaniya-asfaltosmoloparafinovykh-otlozheniy-metodiki-issledovaniya-/)). При разгазировании WAT **повышается** из-за уменьшения доли лёгких фракций — растворителей парафина.

**Механизмы переноса к стенке/поверхности отложения** (классическая классификация):
- **молекулярная диффузия** под действием градиента концентрации, индуцированного температурным градиентом (доминирующий механизм для большинства условий);
- **сдвиговая дисперсия**;
- **броуновская диффузия**;
- **гравитационное осаждение**;
- **термодиффузия (эффект Соре)**.

Основополагающая работа — [Burger, Perkins, Striegler (1981), «Studies of wax deposition in the Trans Alaska pipeline», JPT 33(6):1075–1086](https://www.onepetro.org/journal-paper/SPE-8788-PA), где отложение объяснено совокупностью молекулярной диффузии, сдвиговой дисперсии и броуновской диффузии. Позже Brown, Niesen, Erickson (1993–1994) экспериментально показали несущественность сдвиговой дисперсии и отсутствие отложения при нулевом тепловом потоке — что закрепило **молекулярную диффузию** как главный механизм.

**Реология и гелеобразование.** Ниже температуры гелеобразования высокопарафинистая нефть ведёт себя как неньютоновская среда с пределом текучести (модели Бингама, Гершеля—Балкли); присутствие воды усиливает гелеобразование, изменяя температуру застывания и предел текучести ([гелеобразование, RG](https://www.researchgate.net/publication/235429714_Gelation_Behavior_of_Model_Wax-Oil_and_Crude_Oil_Systems_and_Yield_Stress_Model_Development); [нелинейная реология, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0920410522003308)). Для пласта это означает появление **начального (предельного) градиента давления** и порогового температурного эффекта — нефть фильтруется, лишь когда градиент давления превышает порог ([неньютоновская фильтрация тяжёлой нефти, Springer](https://link.springer.com/article/10.1007/s13202-012-0043-9)).

---

## 3. Термодинамическое моделирование фазового равновесия (выпадение твёрдой фазы)

Этот блок отвечает на вопрос «сколько и какой твёрдой фазы выпадает при заданных (T, p, состав)» и задаёт WAT. Две основные парадигмы:

### 3.1. Модели твёрдого раствора (Solid-Solution, SS)
Предполагают, что все компоненты в твёрдой фазе взаимно растворимы. Описание: кубическое уравнение состояния (EOS) для пар/жидкость + модель коэффициентов активности (G^E) для равновесия «жидкость—твёрдое»:
- **Won (1986)** — «Thermodynamics for solid solution-liquid-vapor equilibria: wax phase formation from heavy hydrocarbon mixtures», *Fluid Phase Equilibria* 30:265–279 — модель регулярного раствора ([контекст, RG](https://www.researchgate.net/publication/238639852_Thermodynamic_Modeling_of_Wax_Precipitation_in_Crude_Oils1)).
- **Hansen, Fredenslund, Pedersen, Rønningsen (1988)**, *AIChE Journal* — полимерная модель раствора.
- **Pedersen, Hansen, Larsen, Nielsen, Rønningsen (1991)** — «Wax precipitation from North Sea crude oils. 4. Thermodynamic modeling», *Energy & Fuels* 5(6):924–932 ([контекст, Springer](https://link.springer.com/article/10.1007/s12182-015-0071-4)).

### 3.2. Многотвёрдофазная модель (Multi-Solid, MS)
Каждый выпадающий компонент образует **отдельную несмешивающуюся** чистую твёрдую фазу:
- **Lira-Galeana, Firoozabadi, Prausnitz (1996)** — «Thermodynamics of wax precipitation in petroleum mixtures», *AIChE Journal* 42(1):239–248 — каноническая MS-модель; пар/жидкость по PR-EOS, характеризация фракций по Уитсону ([обзор моделей, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0009250901003384)).

### 3.3. Предиктивный UNIQUAC и современные подходы
- **Coutinho (1998)** — «Predictive UNIQUAC: A new model for the description of multiphase solid−liquid equilibria in complex hydrocarbon mixtures», *Ind. Eng. Chem. Res.* 37:4870–4875 — предсказывает расщепление твёрдой фазы на несколько сосуществующих твёрдых растворов; устраняет переоценку выпадения лёгких алканов ([ACS](https://pubs.acs.org/doi/full/10.1021/ie980340h); [новый predictive UNIQUAC, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0378381206002743)).
- **PC-SAFT-подходы** — современное направление для прогноза WAT и массы парафина, в т.ч. в рамках MS-каркаса ([PC-SAFT для парафина, Springer](https://link.springer.com/article/10.1007/s12182-015-0071-4); [MS + PC-SAFT, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0016236121020810)).

> Практический итог блока 3: термодинамический модуль выдаёт зависимость массовой доли твёрдой фазы w_s(T, p, состав) и WAT, которые служат источниковым членом для уравнений переноса/отложения. В русскоязычной инженерной практике часто используется коммерческий модуль **Multiflash Wax** ([Кищенко и др., 2018](https://cyberleninka.ru/article/n/modelirovanie-protsessa-obrazovaniya-organicheskih-otlozheniy-parafinovogo-tipa-pri-ekspluatatsii-skvazhin-elektrotsentrobezhnymi)).

---

## 4. Кинетические и транспортные модели роста отложения

Отвечают на вопрос «с какой скоростью растёт отложение». Основные школы:

- **Молекулярно-диффузионная модель (школа Fogler, Мичиган).** [Singh, Venkatesan, Fogler, Nagarajan (2000), «Formation and aging of incipient thin film wax-oil gels», *AIChE J.* 46(5):1059–1074](https://deepblue.lib.umich.edu/handle/2027.42/34241) и [Singh et al. (2001), «Morphological evolution of thick wax deposits during aging», *AIChE J.* 47(1):6–18](https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.690470103). Радиальная диффузия молекул парафина из ядра потока в гель вызывает одновременный рост и «старение» (aging) отложения за счёт встречной диффузии (counter-diffusion) и увеличения доли твёрдой фазы.
- **Тепломассоперенос как замкнутая модель.** [Huang, Lee, Senra, Fogler (2011), «A fundamental model of wax deposition in subsea oil pipelines», *AIChE J.* 57(11):2955–2964](https://deepblue.lib.umich.edu/bitstream/handle/2027.42/87116/12517_ftp.pdf) — прогноз толщины и доли парафина в отложении на основе совместного тепло- и массопереноса (ламинарный/турбулентный режим), учёт конкуренции «осаждение vs. выпадение в объёме».
- **Гидродинамическая многокомпонентная модель.** Ramirez-Jaramillo, Lira-Galeana, Manero (2004), «Modeling wax deposition in pipelines» — связка фазового равновесия с неньютоновским течением; молекулярная диффузия через пограничный слой как доминирующий механизм ([контекст, RG](https://www.researchgate.net/publication/232874607_Modeling_Wax_Deposition_in_Pipelines)).
- **Необратимая термодинамика / энтальпийно-пористый (enthalpy–porosity) подход.** [Banki, Hoteit, Firoozabadi (2008), «Mathematical formulation and numerical modeling of wax deposition in pipelines from enthalpy–porosity approach and irreversible thermodynamics», *Int. J. Heat Mass Transfer* 51:3387–3398](https://www.sciencedirect.com/science/article/abs/pii/S0017931007006722) — поток диффузии в неизотермических условиях моделируется через термодинамику необратимых процессов; закупорка имитируется изменением проницаемости по кинетическому уравнению для пористости. Эта работа **методологически наиболее близка** к задаче «выпадение + неизотермика + изменение проницаемости» и легко переносится с трубы на пористую среду.
- **Коэффициент диффузии парафина в температурном градиенте.** [Correra et al. (2007), «Wax diffusivity under given thermal gradient: a mathematical model», *ZAMM* 87](https://onlinelibrary.wiley.com/doi/10.1002/zamm.200510293).

---

## 5. Неизотермическая фильтрация в пласте: базовые модели тепломассопереноса

Несущий каркас для пластовой задачи — система уравнений сохранения массы фаз/компонентов (закон Дарси) и **энергии** (конвективно-кондуктивный перенос тепла), с учётом эффектов Джоуля—Томсона, адиабатического расширения и теплоты разгазирования/кристаллизации ([многофазное неизотермическое движение, oilgasjournal](http://oilgasjournal.ru/issue_21/chetvertushkin-lyupa-trapeznikova.pdf)).

**Российская научная школа (фундамент):**
- **Чекалюк Э.Б. (1965). Термодинамика нефтяного пласта. — М.: Недра.** Классическая монография по термодинамике пластовых процессов ([GeoKniga](https://www.geokniga.org/books/4848)).
- **Намиот А.Ю.** — теплопередача при подъёме нефти, фазовые равновесия (в т.ч. с водой при высоких T).
- **Алишаев М.Г., Розенберг М.Д., Теслюк Е.В. (1985). Неизотермическая фильтрация при разработке нефтяных месторождений. — М.: Недра (~270 с.).** Базовый труд по математическому описанию неизотермической фильтрации, термозаводнению ([контекст в литературе](https://natural-sciences.ru/article/view?id=36164)).
- **Плохотников С.П., Фатыхов Р.Х. Математическое моделирование фильтрации в слоистых пластах** — модели двухфазной фильтрации в слоистых пластах при изотермическом и неизотермическом режимах ([PDF, GeoKniga](https://www.geokniga.org/bookfiles/geokniga-matematicheskoe-modelirovanie-filtracii-v-sloistyh-plastah.pdf)).

**Современные постановки и численные методы:**
- Автомодельные задачи неизотермической двухфазной фильтрации и теплового пограничного слоя ([dissercat](https://www.dissercat.com/content/avtomodelnye-zadachi-neizotermicheskoi-dvukhfaznoi-filtratsii-i-teplovogo-pogranichnogo-sloy)).
- Неизотермическая фильтрация жидкости и газа с **фазовыми переходами** ([dissercat](https://www.dissercat.com/content/issledovanie-protsessov-neizotermicheskoi-filtratsii-zhidkosti-i-gaza-s-fazovymi-perekhodami)).
- Разрешимость задачи неизотермической фильтрации в пороупругом слое ([Изв. АлтГУ](https://izvestiya.asu.ru/article/view/(2026)1-13)).
- Численное моделирование многофазных течений в пласте ([ПМТФ, 2019](https://sibran.ru/upload/iblock/748/74828c573eba9e9bc630ef9164b45a63.pdf)).
- Методологически родственный обзор связанной задачи: [Mathematical Modeling of a Non-Isothermal Flow in a Porous Medium Considering Gas Hydrate Decomposition: A Review, *Mathematics* (MDPI) 10(24):4674](https://www.mdpi.com/2227-7390/10/24/4674) — полезен как образец постановки неизотермической фильтрации с фазовым переходом (гидрат ↔ парафин аналогичны по структуре уравнений).
- Расчёт фильтрации вязкопластичных нефтей в задачах термозаводнения ([Вестник АГТУ](https://vestnik.astu.org/ru/nauka/article/71928/view)).

---

## 6. Сопряжённое моделирование: выпадение → отложение → повреждение коллектора

Любой пластовый симулятор отложения должен включать (а) термодинамический модуль выпадения и (б) модель связи количества отложенной фазы с изменением проницаемости породы. Канон здесь — **формационное повреждение по Civan**:

- **Civan, F. Reservoir Formation Damage: Fundamentals, Modeling, Assessment, and Mitigation (Gulf Professional Publishing; 3-е изд. 2016).** Модифицированное уравнение **Козени—Кармана** для среды, изменённой осаждением; кинетика осаждения; три механизма — поверхностное осаждение, закупорка поровых каналов (pore-throat plugging) и реэнтрейнмент (повторный вынос) ([книга, vdoc.pub](https://vdoc.pub/documents/reservoir-formation-damage-fundamentals-modeling-assessment-and-mitigation-7d5nok4gg810); [масштабный эффект пористость/проницаемость, Civan 2001, AIChE J.](https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.690470206)).

**Асфальтеновое осаждение в пористой среде** (методологически переносимо на парафины):
- [Wang, Civan и др. — Modeling Formation Damage due to Asphaltene Deposition in the Porous Media, *Energy & Fuels*](https://pubs.acs.org/doi/abs/10.1021/ef101195a) — осаждение, флокуляция, адсорбция, закупорка, реэнтрейнмент; результирующее снижение пористости/проницаемости, рост вязкости, изменение смачиваемости.
- [Modelling Asphaltene Precipitation and Deposition in a Compositional Reservoir Simulator, SPE-89437 (Nghiem и др.)](https://onepetro.org/SPEIOR/proceedings-abstract/04IOR/04IOR/SPE-89437-MS/71368) — реализация в компонентном (compositional) симуляторе.
- [Simulating oil flow in porous media under asphaltene deposition, *Chem. Eng. Sci.*](https://www.sciencedirect.com/science/article/abs/pii/S0009250901004079); [новая модель ухудшения проницаемости, *Fuel* 2019](https://www.sciencedirect.com/science/article/abs/pii/S001623611831278X); [влияние на смачиваемость и проницаемость, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8358938/).

> Перенос на парафины: w_s(T,p) из блока 3 → объёмная доля отложения σ → φ(σ), k(φ) по Козени—Карману/Civan → обновление коэффициентов в уравнениях фильтрации блока 5. Это и есть «скелет» пластового симулятора АСПО.

---

## 7. Численное моделирование парафиноотложения непосредственно в пласте и ПЗП

Наиболее релевантная вашей теме группа работ — выпадение парафина именно **в коллекторе**, а не в трубе:

- [Simulation of Paraffin Deposition in Petroleum Reservoirs (academia.edu)](https://www.academia.edu/91089623/Simulation_of_Paraffin_Deposition_in_Petroleum_Reservoirs_3_Subscripts) — численный симулятор парафиноотложения в пласте; показывает, что выпадение/отложение — преимущественно **околоскважинное** явление (где градиент давления максимален).
- [Wang et al. (2014), «Numerical simulation of paraffin wax deposition in waxy crude oil reservoir with cold water flooding», *Geosystem Engineering* 17(4)](https://www.tandfonline.com/doi/abs/10.1080/12269328.2014.959623) — фазовое превращение жидкость↔твёрдое связано с двухфазной (нефть—вода) фильтрацией; «фронт холодового повреждения» с ростом времени закачки смещается вглубь пласта, степень повреждения растёт.
- [He et al. (2020), «Cold damage from wax deposition in a shallow, low-temperature, and high-wax reservoir in Changchunling Oilfield», *Scientific Reports* 10:14595](https://www.nature.com/articles/s41598-020-71065-z) (open access) — уравнение фазового превращения + двухфазная нефть-водяная модель фильтрации; экспериментально показано резкое падение радиуса поровых каналов и **относительных фазовых проницаемостей** с понижением пластовой температуры.
- [Wax Precipitation Modelling In Low-Temperature Reservoir Under Different Production Regimes, EAGE/Earthdoc (2018)](https://www.earthdoc.org/content/papers/10.3997/2214-4609.201802189) — моделирование выпадения парафина в низкотемпературном пласте при разных режимах добычи.
- [Numerical prediction / Duan et al. (2016), *AIChE J.*, оил-газ стратифицированное течение](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.15223) — слой отложения как растущая пористая среда (диффузия парафина сквозь пористый слой); полезная аналогия пористого отложения.

**Сопряжённые «скважина—пласт» симуляторы** (важны для ПЗП):
- [Development of a Coupled Wellbore/Reservoir Simulator for Damage Prediction and Remediation, SPE-156393](https://onepetro.org/SPEATCE/proceedings-abstract/12ATCE/All-12ATCE/156393) и [диссертация UT Austin](https://repositories.lib.utexas.edu/items/dc8b355f-7486-42c9-b001-1e88295f8293) — термический, многофазный, многокомпонентный симулятор ствола, связанный с компонентным пластовым; уравнения сохранения массы каждого компонента, импульса фаз, энергии смеси и условия равновесия фугитивностей между фазами (нефть, газ, парафин, асфальтен).
- [Wax Deposition Pattern in Wellbore Region of Deep Condensate Gas Reservoir, SPE ATCE 2022](https://onepetro.org/SPEATCE/proceedings-abstract/22ATCE/2-22ATCE/509324) и [Prediction of wax precipitation region in wellbore during deep water oil well testing, *Petroleum* (2018)](https://www.sciencedirect.com/science/article/pii/S1876380418300399) — тепловая и молекулярная диффузия как главные механизмы в околоскважинной зоне.

---

## 8. Русскоязычные работы по моделированию АСПО в скважинах и пласте

- **Тронов В.П. Механизм образования смоло-парафиновых отложений и борьба с ними. — М.: Недра, 1969/1970.** Классическая отечественная монография по механизму АСПО.
- **Глущенко В.Н., Силин М.А. (2009). Нефтепромысловая химия (многотомник).** Базовый источник по механизмам и природе АСПО ([контекст, Кищенко и др.](https://cyberleninka.ru/article/n/modelirovanie-protsessa-obrazovaniya-organicheskih-otlozheniy-parafinovogo-tipa-pri-ekspluatatsii-skvazhin-elektrotsentrobezhnymi)).
- **Ибрагимов Н.Г., Тронов В.П., Гуськова И.А. (2011).** Методы борьбы с отложениями.
- **Кищенко М.А., Александров А.Н., Рогачёв М.К., Кибирев Е.А. (2018). Моделирование процесса образования органических отложений парафинового типа при эксплуатации скважин ЭЦН. — *Экспозиция Нефть Газ*.** Гидродинамика «пласт—скважина—насос», профили T и p (методики Ляпкова, Мищенко, модель OLGAS), фазовая диаграмма по Multiflash Wax; межфазный массоперенос, роль газовых глобул ([cyberleninka](https://cyberleninka.ru/article/n/modelirovanie-protsessa-obrazovaniya-organicheskih-otlozheniy-parafinovogo-tipa-pri-ekspluatatsii-skvazhin-elektrotsentrobezhnymi)).
- **Диссертации (ВАК 25.00.17 / 1.6.x, 05.13.18 / матмоделирование):**
  - Исследование условий образования АСПО в скважинах и разработка технологии борьбы с ними ([dissercat](https://www.dissercat.com/content/issledovanie-uslovii-obrazovaniya-asfaltosmoloparafinovykh-otlozhenii-v-skvazhinakh-i-razrab)).
  - Обоснование комплексной технологии предупреждения АСПО при добыче высокопарафинистой нефти ЭЦН из многопластовых залежей (СПГУ/СПМИ; влияние WAT, устьевого давления, частоты вращения вала насоса на глубину отложения) ([dissercat](https://www.dissercat.com/content/obosnovanie-kompleksnoi-tekhnologii-preduprezhdeniya-obrazovaniya-asfaltosmoloparafinovykh)).
  - Научно-методические основы моделирования управления эксплуатационными характеристиками осложнённых скважин (моделирование термобарических условий и прогноз скорости отложения АСПО) ([dissercat](https://www.dissercat.com/content/nauchno-metodicheskie-osnovy-modelirovaniya-protsessov-upravleniya-ekspluatatsionnymi-kharak-0)).
  - Численное моделирование тепломассопереноса в нефтяной скважине с греющим кабелем (05.13.18) ([dissercat](https://www.dissercat.com/content/chislennoe-modelirovanie-protsessov-teplomassoperenosa-v-neftyanoi-skvazhine-s-greyushchim)).
  - Механизм и условия формирования АСПО на поздней стадии разработки (НГДУ «Джалильнефть») ([dissercat](https://www.dissercat.com/content/mekhanizm-i-usloviya-formirovaniya-asfalto-smolo-parafinovykh-otlozhenii-na-pozdnei-stadii-r)).
- **ВКР/учебные источники:** [ВКР, КФУ (2015)](https://kpfu.ru/portal/docs/F_1151855890/VKR_2015_Burdin_SA.pdf); диссертация СПМИ (2022) ([PDF](https://spmi.ru/sites/default/files/imci_images/sciens/dissertacii/2022/sandyga_dissertaciya.pdf)).
- Прямые/обратные задачи тепломассопереноса в нефтяных пластах ([Cyberleninka, ПДФ](https://cyberleninka.ru/article/n/chislennoe-reshenie-pryamyh-i-obratnyh-zadach-teplomassoperenosa-v-neftyanyh-plastah/pdf)).

---

## 9. Современные обзоры (2023–2025)

- [Yu et al. (2024). A comprehensive review of wax deposition in crude oil systems: Mechanisms, influencing factors, prediction and inhibition techniques. *Fuel* 357:129676](https://www.sciencedirect.com/science/article/abs/pii/S0016236123022901) — наиболее полный современный обзор; классифицирует прогнозные модели на термодинамические и кинетические.
- [Ma et al. (2025). Research progress on wax deposition mechanisms, influencing factors, prediction models, and wax removal/prevention. *Energy Exploration & Exploitation* (SAGE)](https://journals.sagepub.com/doi/10.1177/01445987251343732).
- [Progress on Wax Deposition Characteristics and Prediction Methods for Crude Oil Pipelines. *Processes* (MDPI) 13(6):1651 (2025)](https://www.mdpi.com/2227-9117/13/6/1651) — систематизация термодинамических и кинетических моделей.
- [Wax Deposition during the Transportation of Waxy Crude Oil: Mechanisms, Influencing Factors, Modeling, and Outlook. *Energy & Fuels* (2024)](https://pubs.acs.org/doi/10.1021/acs.energyfuels.3c04687).
- [Characterization of Wax Precipitation and Deposition Behavior of Condensate Oil in Wellbore … *Energies* (MDPI) 15(11):4018 (2022)](https://www.mdpi.com/1996-1073/15/11/4018) — моделирование + эксперимент + молекулярная динамика для околоскважинной зоны.
- [Review of wax deposition in subsea oil pipeline systems and mitigation technologies. *Petroleum* (2021)](https://www.sciencedirect.com/science/article/pii/S266682112100020X).

---

## 10. Пробелы в знаниях и направления для диссертации

1. **Дефицит чисто пластовых моделей.** Зрелые диффузионно-тепловые модели (Singh–Fogler, Huang–Fogler) разработаны для труб; в пористой среде доминируют асфальтеновые модели повреждения, а парафиновые — реже и проще. Перенос аппарата Banki–Hoteit–Firoozabadi (необратимая термодинамика + изменение проницаемости) на пласт — перспективен и пока недоиспользован.
2. **Полная связка.** Немного работ, где одновременно решаются: (а) термодинамика выпадения w_s(T,p,состав) с реальной EOS/PC-SAFT; (б) неизотермическая многофазная фильтрация с энергетическим уравнением; (в) кинетика осаждения/реэнтрейнмента; (г) обратная связь φ→k и на ОФП. Это естественная архитектура вашего симулятора.
3. **Неньютоновские эффекты и начальный градиент** в зоне геля редко включаются совместно с термодинамикой выпадения.
4. **Верификация на промысловых данных** холодного заводнения высокопарафинистых залежей (Wang 2014; Changchunling 2020) — готовые тест-кейсы для валидации.
5. **Околоскважинная локализация.** Поскольку выпадение — преимущественно околоскважинный эффект, оправдано локальное измельчение сетки/связка «скважина—ПЗП—пласт».

---

## Список наиболее значимых источников (структурированная библиография)

### A. Термодинамика выпадения твёрдой фазы (парафин/асфальтен)
1. Won K.W. (1986). Thermodynamics for solid solution-liquid-vapor equilibria: wax phase formation from heavy hydrocarbon mixtures. *Fluid Phase Equilibria* 30:265–279.
2. Hansen J.H., Fredenslund Aa., Pedersen K.S., Rønningsen H.P. (1988). *AIChE Journal* — модель выпадения парафина.
3. Pedersen K.S. et al. (1991). Wax precipitation from North Sea crude oils. 4. Thermodynamic modeling. *Energy & Fuels* 5(6):924–932. — [Springer-контекст](https://link.springer.com/article/10.1007/s12182-015-0071-4)
4. Lira-Galeana C., Firoozabadi A., Prausnitz J.M. (1996). Thermodynamics of wax precipitation in petroleum mixtures. *AIChE Journal* 42(1):239–248 (multi-solid). — [обзор](https://www.sciencedirect.com/science/article/abs/pii/S0009250901003384)
5. Coutinho J.A.P. (1998). Predictive UNIQUAC. *Ind. Eng. Chem. Res.* 37:4870–4875. — [ACS](https://pubs.acs.org/doi/full/10.1021/ie980340h)
6. PC-SAFT-моделирование выпадения парафина (2015, 2021). — [Springer](https://link.springer.com/article/10.1007/s12182-015-0071-4); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0016236121020810)

### B. Кинетика/перенос и рост отложения
7. Burger E.D., Perkins T.K., Striegler J.H. (1981). Studies of wax deposition in the Trans Alaska pipeline. *JPT* 33(6):1075–1086. — [OnePetro](https://www.onepetro.org/journal-paper/SPE-8788-PA)
8. Singh P., Venkatesan R., Fogler H.S., Nagarajan N. (2000). Formation and aging of incipient thin film wax-oil gels. *AIChE J.* 46(5):1059–1074. — [Deep Blue](https://deepblue.lib.umich.edu/handle/2027.42/34241)
9. Singh P. et al. (2001). Morphological evolution of thick wax deposits during aging. *AIChE J.* 47(1):6–18. — [Wiley](https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.690470103)
10. Ramirez-Jaramillo E., Lira-Galeana C., Manero O. (2004). Modeling wax deposition in pipelines. *Pet. Sci. Technol.* — [RG](https://www.researchgate.net/publication/232874607_Modeling_Wax_Deposition_in_Pipelines)
11. Huang Z., Lee H.S., Senra M., Fogler H.S. (2011). A fundamental model of wax deposition in subsea oil pipelines. *AIChE J.* 57(11):2955–2964. — [PDF](https://deepblue.lib.umich.edu/bitstream/handle/2027.42/87116/12517_ftp.pdf)
12. Banki R., Hoteit H., Firoozabadi A. (2008). Mathematical formulation and numerical modeling of wax deposition … enthalpy–porosity approach and irreversible thermodynamics. *Int. J. Heat Mass Transfer* 51:3387–3398. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0017931007006722)
13. Correra S. et al. (2007). Wax diffusivity under given thermal gradient. *ZAMM* 87. — [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/zamm.200510293)

### C. Пористая среда / пласт: фильтрация, повреждение, ОФП
14. Civan F. Reservoir Formation Damage, 3rd ed. (2016), Gulf Professional Publishing. — [vdoc.pub](https://vdoc.pub/documents/reservoir-formation-damage-fundamentals-modeling-assessment-and-mitigation-7d5nok4gg810)
15. Civan F. (2001). Scale effect on porosity and permeability: kinetics, model, and correlation. *AIChE J.* 47(2). — [Wiley](https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.690470206)
16. Wang S., Civan F. Modeling Formation Damage due to Asphaltene Deposition in the Porous Media. *Energy & Fuels*. — [ACS](https://pubs.acs.org/doi/abs/10.1021/ef101195a)
17. Nghiem L. et al. (2004). Modelling Asphaltene Precipitation and Deposition in a Compositional Reservoir Simulator. SPE-89437. — [OnePetro](https://onepetro.org/SPEIOR/proceedings-abstract/04IOR/04IOR/SPE-89437-MS/71368)
18. New model for permeability impairment due to asphaltene deposition (2019). *Fuel*. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S001623611831278X)

### D. Парафиноотложение непосредственно в пласте/ПЗП
19. Simulation of Paraffin Deposition in Petroleum Reservoirs. — [academia.edu](https://www.academia.edu/91089623/Simulation_of_Paraffin_Deposition_in_Petroleum_Reservoirs_3_Subscripts)
20. Wang et al. (2014). Numerical simulation of paraffin wax deposition in waxy crude oil reservoir with cold water flooding. *Geosystem Engineering* 17(4). — [T&F](https://www.tandfonline.com/doi/abs/10.1080/12269328.2014.959623)
21. He K. et al. (2020). Cold damage from wax deposition in a shallow, low-temperature, high-wax reservoir, Changchunling Oilfield. *Scientific Reports* 10:14595. — [Nature](https://www.nature.com/articles/s41598-020-71065-z)
22. Wax Precipitation Modelling In Low-Temperature Reservoir Under Different Production Regimes (2018), EAGE. — [Earthdoc](https://www.earthdoc.org/content/papers/10.3997/2214-4609.201802189)
23. Coupled Wellbore/Reservoir Simulator for Damage Prediction and Remediation. SPE-156393. — [OnePetro](https://onepetro.org/SPEATCE/proceedings-abstract/12ATCE/All-12ATCE/156393); [UT Austin](https://repositories.lib.utexas.edu/items/dc8b355f-7486-42c9-b001-1e88295f8293)
24. Duan J. et al. (2016). Wax deposition modeling of oil/gas stratified pipe flow. *AIChE J.* (растущий пористый слой отложения). — [Wiley](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.15223)

### E. Неизотермическая фильтрация — российская школа
25. Чекалюк Э.Б. (1965). Термодинамика нефтяного пласта. — М.: Недра. — [GeoKniga](https://www.geokniga.org/books/4848)
26. Алишаев М.Г., Розенберг М.Д., Теслюк Е.В. (1985). Неизотермическая фильтрация при разработке нефтяных месторождений. — М.: Недра.
27. Намиот А.Ю. — теплопередача при подъёме нефти; фазовые равновесия.
28. Плохотников С.П., Фатыхов Р.Х. Математическое моделирование фильтрации в слоистых пластах. — [PDF](https://www.geokniga.org/bookfiles/geokniga-matematicheskoe-modelirovanie-filtracii-v-sloistyh-plastah.pdf)
29. Автомодельные задачи неизотермической двухфазной фильтрации (дисс., 05.13.16). — [dissercat](https://www.dissercat.com/content/avtomodelnye-zadachi-neizotermicheskoi-dvukhfaznoi-filtratsii-i-teplovogo-pogranichnogo-sloy)
30. Исследование процессов неизотермической фильтрации жидкости и газа с фазовыми переходами (дисс.). — [dissercat](https://www.dissercat.com/content/issledovanie-protsessov-neizotermicheskoi-filtratsii-zhidkosti-i-gaza-s-fazovymi-perekhodami)

### F. Российские работы по моделированию АСПО
31. Тронов В.П. (1969). Механизм образования смоло-парафиновых отложений и борьба с ними. — М.: Недра.
32. Глущенко В.Н., Силин М.А. (2009). Нефтепромысловая химия. — М.
33. Кищенко М.А., Александров А.Н., Рогачёв М.К., Кибирев Е.А. (2018). Моделирование … парафиновых отложений при эксплуатации скважин ЭЦН. *Экспозиция Нефть Газ*. — [Cyberleninka](https://cyberleninka.ru/article/n/modelirovanie-protsessa-obrazovaniya-organicheskih-otlozheniy-parafinovogo-tipa-pri-ekspluatatsii-skvazhin-elektrotsentrobezhnymi)
34. Обоснование комплексной технологии предупреждения АСПО … ЭЦН (дисс., СПГУ). — [dissercat](https://www.dissercat.com/content/obosnovanie-kompleksnoi-tekhnologii-preduprezhdeniya-obrazovaniya-asfaltosmoloparafinovykh)
35. Научно-методические основы моделирования управления … осложнённых скважин (дисс.). — [dissercat](https://www.dissercat.com/content/nauchno-metodicheskie-osnovy-modelirovaniya-protsessov-upravleniya-ekspluatatsionnymi-kharak-0)

### G. Современные обзоры (2021–2025)
36. Yu et al. (2024). Comprehensive review of wax deposition in crude oil systems. *Fuel* 357:129676. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0016236123022901)
37. Ma et al. (2025). Research progress on wax deposition … *Energy Exploration & Exploitation*. — [SAGE](https://journals.sagepub.com/doi/10.1177/01445987251343732)
38. Progress on Wax Deposition … for Crude Oil Pipelines (2025). *Processes* 13(6):1651. — [MDPI](https://www.mdpi.com/2227-9117/13/6/1651)
39. Wax Deposition during the Transportation of Waxy Crude Oil (2024). *Energy & Fuels*. — [ACS](https://pubs.acs.org/doi/10.1021/acs.energyfuels.3c04687)
40. Non-Isothermal Flow in a Porous Medium Considering Gas Hydrate Decomposition: A Review (2022). *Mathematics* 10(24):4674. — [MDPI](https://www.mdpi.com/2227-7390/10/24/4674)

---

## Методология обзора
Выполнено ~12 поисковых запросов (англо- и русскоязычных) по подзадачам: (1) термодинамика выпадения парафина; (2) кинетика/перенос отложения; (3) неизотермическая фильтрация и тепломассоперенос; (4) повреждение коллектора и связь φ–k; (5) пластовые/околоскважинные численные модели; (6) русскоязычная школа АСПО; (7) современные обзоры. Проработано ~45 источников; для ряда ключевых проведено полнотекстовое чтение. Часть библиографических деталей русскоязычных монографий (том, страницы, год переиздания) рекомендуется сверить по eLibrary/РГБ перед включением в диссертацию.
