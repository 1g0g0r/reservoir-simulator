# Компонентное (composition-based) математическое моделирование осаждения тяжёлых углеводородов при изменении термобарических условий

*Углублённый обзор. Подготовлено: 2026-06-27 · Проработано источников: ~75 суммарно по теме · Фокус: модели, явно разрешающие компонентный состав нефти (н-алканы CₙH₂ₙ₊₂, SCN-фракции, C7+) и зависимость осаждения от (T, p). Достоверность: высокая по международной рецензируемой литературе; точные числа цитирований и часть выходных данных рекомендуется сверить.*

> Методологическая оговорка: поиск выполнен WebSearch/WebFetch (Firecrawl/Exa не сконфигурированы). Полнотекстовое чтение PC-SAFT-главы Rice не удалось (бинарный PDF) — её содержание реконструировано по аннотациям нескольких источников и помечено соответствующе.

---

## 0. Что такое «компонентная» модель и при чём тут термобарика

**Компонентный (compositional) подход** описывает нефть не как «чёрную» жидкость с эмпирическими свойствами (black-oil), а как смесь N компонентов и псевдокомпонентов {CH₄, …, н-алканы CₙH₂ₙ₊₂, нафтены, ароматика, смолы, асфальтены}. Для каждого компонента i записываются химический потенциал μᵢ и фугитивность fᵢ, а равновесие фаз (пар–жидкость–твёрдое, V–L–S, и L–L для асфальтенов) определяется условием равенства фугитивностей:
**fᵢ^L(T,p,**x**) = fᵢ^S(T,p,**z**) = fᵢ^V(T,p,**y**)** для всех i.

**Термобарический путь.** При разработке (T, p) в каждой точке пласта и ствола меняются: депрессуризация, охлаждение у стенки/у забоя, закачка газа/CO₂/растворителя. Эти изменения смещают флюид через **границы фазовой устойчивости** — температуру начала кристаллизации парафина (WAT) и давление начала осаждения асфальтенов (Asphaltene Onset Pressure, AOP). Поэтому компонентная модель осаждения = **термодинамика фазового равновесия с твёрдой/тяжёлой фазой**, вычисляемая вдоль термобарического пути на каждом узле фильтрационной сетки.

Ключевое отличие парафинов и асфальтенов по термобарике:
- **Парафины** выпадают в основном при **снижении T** (ниже WAT); давление влияет слабее, но для «живой» нефти существенно (растворённый газ).
- **Асфальтены** выпадают в основном при **снижении p** к давлению насыщения (максимум у P_b) и при закачке газа; T влияет менее монотонно.

---

# ЧАСТЬ I. Характеризация компонентного состава — фундамент всех моделей

Точность любой компонентной модели осаждения определяется прежде всего **характеризацией тяжёлого «хвоста» C7+** и распределением н-парафинов внутри него. Это самый чувствительный этап ([15 псевдокомпонентов до 1200 г/моль для точной WAT, RG](https://www.researchgate.net/publication/241790086_C7_Characterization_of_Heavy_Oil_Based_on_Crude_Assay_Data)).

### 1.1. Распределения молярного состава C7+
- **Katz (экспоненциальное распределение, 1978/1983)** — простейшая модель убывания мольной доли с номером SCN.
- **Whitson (гамма-распределение, 1983)** — трёхпараметрическое (α, β, η) распределение по молекулярной массе; гибко описывает и лёгкие, и тяжёлые нефти; стандарт де-факто; разбиение на псевдокомпоненты квадратурой Гаусса–Лагерра ([гамма-распределение C7+, RG](https://www.researchgate.net/publication/308358826_C7_Characterization_of_Related_Equilibrium_Fluids_Using_the_Gamma_Distribution); [Characterizing Hydrocarbon Plus Fractions, Whitson](https://www.researchgate.net/publication/250090595_Characterizing_Hydrocarbon_Plus_Fractions)).
- **Pedersen, Ahmed, Riazi** — альтернативные/непрерывные модели; SCN-фракции расширяют до C90+ ([Γ-распределение для MW и Tкип, RG](https://www.researchgate.net/publication/244595959_Application_of_the_G-Distribution_Model_to_Molecular_Weight_and_Boiling_Point_Data_for_Petroleum_Fractions); [непрерывная модель C7+, Riazi PDF](http://www.riazim.com/sample/C7EC97.pdf)). Сравнение точности: новые оптимизационные методы дают среднюю ошибку ~25,8 % против 76 % (Katz), 33,6 % (Ahmed), 45,9 % (Whitson) на ряде нефтей ([C7+ по crude assay, RG](https://www.researchgate.net/publication/241790086_C7_Characterization_of_Heavy_Oil_Based_on_Crude_Assay_Data)).
- Критические свойства и ацентрический фактор псевдокомпонентов — по корреляциям (Lee–Kesler, Riazi–Daubert, Twu) ([оценка корреляций, RG](https://www.researchgate.net/publication/322996666_Evaluation_of_Different_Correlation_Performance_for_the_Calculation_of_the_Critical_Properties_and_Acentric_Factor_of_Petroleum_Heavy_Fractions)).

### 1.2. Распределение н-парафинов (CₙH₂ₙ₊₂) внутри фракций
Поскольку именно н-алканы образуют парафиновую твёрдую фазу, нужно знать **долю н-парафинов в каждом SCN** (не всю фракцию). Методы: высокотемпературная газовая хроматография (HTGC), DSC-деконволюция, корреляции экспоненциального спада ([методики определения n-парафинового распределения, RG](https://www.researchgate.net/publication/256712596_Evaluation_of_different_methodologies_to_determine_the_n-paraffin_distribution_of_petroleum_fractions)).

### 1.3. Свойства чистых н-алканов как функции номера углерода n
Для расчёта SLE нужны для каждого н-алкана:
- **температура плавления T_fus(n)** и **энтальпия плавления ΔH_fus(n)** — монотонно растут с n; ΔH между упорядоченной и жидкой фазами **линейна по n** (Broadhurst); сильный **odd-even (чётно-нечётный) эффект** для коротких цепей ([температуры/энтальпии переходов n-алканов, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S002196140290978X); [нечётные C21–C29, RG](https://www.researchgate.net/publication/238127049_Temperatures_and_Enthalpies_of_Solid-Solid_and_Melting_Transitions_of_the_Odd-Numbered_n-_Alkanes_C_21_C_23_C_25_C_27_and_C_29));
- **твёрдотельные переходы (solid–solid)** и теплоёмкости ΔCp; для n>9 четыре кристаллические формы — **гексагональная, триклинная, моноклинная, орторомбическая** ([термофизика n-алканов C17–C50, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0378381224000839));
- **корреляции**: T_fus прогнозируется по n/MW с приемлемой точностью; ΔH_fus и solid–solid переходы предсказываются хуже — существенный источник погрешности ([валидация корреляций C17–C50, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0378381224000839); [групповой вклад для T_m и ΔH_fus, ACS CGD](https://pubs.acs.org/doi/10.1021/acs.cgd.5c00684)).

> Именно зависимости T_fus(n), ΔH_fus(n) для гомологического ряда CₙH₂ₙ₊₂ замыкают термодинамику: они задают идеальную растворимость твёрдой фазы и точку WAT.

---

# ЧАСТЬ II. Классические (старые) компонентные подходы (≈1980–2000)

### 2.1. Каркас EOS + G^E
Стандартная схема: **кубическое уравнение состояния (SRK / Peng–Robinson)** для пар-жидкостного равновесия (VLE) + **модель коэффициентов активности G^E** для твёрдо-жидкого (SLE). Фугитивность твёрдого компонента связывается с жидким через термодинамику плавления:
ln(fᵢ^S/fᵢ^L) = −ΔH_fus,i/RT·(1 − T/T_fus,i) − (1/RT)∫ΔCp dT + … (+ члены solid–solid переходов).

### 2.2. Модели твёрдого раствора (Solid-Solution, SS)
- **Won (1986)** — твёрдая фаза = **единый однородный твёрдый раствор**; модифицированный регулярный раствор; ввёл корреляции T_fus, ΔH_fus(n) ([контекст: Won — единый твёрдый раствор vs Coutinho — несколько, RG](https://www.researchgate.net/publication/379929573_Thermophysical_properties_of_n-alkanes_from_C17_to_C50_and_validation_of_available_correlations)).
- **Hansen et al. (1988)**, **Pedersen et al. (1991)** — теория **полимерных растворов (Флори–Хаггинса)**; Pedersen ввёл эмпирическую поправку, ограничивающую долю компонента, переходящего в твёрдую фазу (борьба с переоценкой) ([улучшенная модель, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0009250901003384)).
- **Erickson, Brown** — инженерные SS-варианты.
- *Ограничение SS:* систематическая **переоценка выпадения лёгких компонентов**; давление практически не учитывается.

### 2.3. Классические термодинамические модели асфальтенов
- **Hirschberg et al. (1984)** — растворная модель: **Флори–Хаггинс + параметр растворимости (Скэтчард–Хильдебранд)**; асфальтены растворены, выпадение **обратимо** ([контекст-обзор моделей, MDPI Processes 11(3):765](https://www.mdpi.com/2227-9117/11/3/765)).
- **Leontaritis & Mansoori (1987)** — **коллоидная** модель: асфальтены — твёрдые частицы, стабилизированные смолами; выпадение часто **необратимо** ([SPE-16258, RG](https://www.researchgate.net/publication/237196338_Asphaltene_Flocculation_During_Oil_Recovery_and_Processing_A_Thermodynamic-Colloidal_Model)). *Противоречие с Hirschberg по обратимости.*

### 2.4. Непрерывная и полунепрерывная термодинамика (continuous thermodynamics)
Альтернатива дискретным псевдокомпонентам: состав описывается **непрерывной функцией распределения** F(I) по индексу I (молекулярная масса / Tкип / параметр растворимости).
- **Rätzsch & Kehlen (1983)** — непрерывная термодинамика; **Cotterman & Prausnitz (1985)** — полунепрерывная; flash двумя способами: **метод моментов** (приближённый) и **квадратурный** (гауссово интегрирование) ([flash по непрерывной термодинамике, RG](https://www.researchgate.net/publication/232992857_Flash_Calculations_for_a_Crude_Oil_by_Continuous_Thermodynamics); [фазовое равновесие непрерывных/полунепрерывных смесей, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0009250987801446)).
- **Применение к тяжёлым компонентам:** моделирование точек флокуляции асфальтенов в рамках Скэтчарда–Хильдебранда с **гауссовыми распределениями по параметру растворимости** (мальтены/асфальтены — отдельные распределения); депарафинизация (de-oiling) восков ([полунепрерывная для многофазного равновесия, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0378381213006493); [de-oiling восков непрерывной термодинамикой, RG](https://www.researchgate.net/publication/245332538_Modelling_of_the_solvent-deoiling_process_of_waxes_by_continuous_thermodynamics)).
- *Достоинство:* малое число параметров для очень многокомпонентных систем; *ограничение:* приближённость, трудность с резкими границами твёрдой фазы.

---

# ЧАСТЬ III. Multi-solid и явный учёт давления (1996+)

### 3.1. Многотвёрдофазная модель
- **Lira-Galeana, Firoozabadi, Prausnitz (1996)** — каждый компонент, прошедший **критерий стабильности** (по фугитивности), образует **отдельную чистую твёрдую фазу**; пар/жидкость по PR-EOS. Точнее SS для тяжёлых фракций; даёт ступенчатую кривую выпадения ([контекст-обзор, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0009250901003384)).
- **Coutinho (predictive UNIQUAC/Wilson, 1998+)** — компромисс: **несколько сосуществующих твёрдых растворов**; устраняет переоценку лёгких ([ACS IECR](https://pubs.acs.org/doi/full/10.1021/ie980340h)).

### 3.2. Учёт давления — поправка Пойнтинга и «живая» нефть
Давление входит в фугитивность твёрдого через **поправку Пойнтинга**:
fᵢ^S(T,p) = fᵢ^S(T,p*)·exp[ vᵢ^S (p − p*) / (RT) ].
- Включение поправки Пойнтинга + solid–solid переходов подтверждает пригодность multi-solid и для **онсета, и для массы** при высоком давлении ([улучшенная модель с Пойнтингом, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0009250901003384)).
- **«Живая» нефть (live oil):** растворённый газ — растворитель; его наличие **повышает WAT**; депрессуризация ниже P_b (разгазирование) резко меняет WAT ([эффект давления/газа на WAT](https://www.sciencedirect.com/science/article/abs/pii/S0009250901003384)).
- **Газоконденсаты:** эффект давления **немонотонный/обратный** — при изотермическом снижении p масса выпавшего парафина может сперва расти, затем падать и снова расти ([высокое давление, новая предиктивная модель, FPE 2007](https://www.sciencedirect.com/science/article/abs/pii/S0378381207001008); [выпадение в газоконденсатах, SPE-56488](https://onepetro.org/SPEATCE/proceedings-abstract/99ATCE/All-99ATCE/SPE-56488-MS/60045); [температурные паттерны в газоконденсатах, ACS Omega 2024](https://pubs.acs.org/doi/10.1021/acsomega.4c10044)).
- **Измерение при высоком давлении:** HP-μDSC для WAT «живых» нефтей ([контекст](https://www.sciencedirect.com/science/article/abs/pii/S0378381207001008)).

---

# ЧАСТЬ IV. Современные EOS-семейства: SAFT / PC-SAFT / CPA (2000–2025)

Главный сдвиг последних двух десятилетий — переход от «EOS+G^E» к **единым уравнениям состояния на основе статистической теории ассоциирующих флюидов**, описывающим VLE, LLE и осаждение тяжёлой фазы в (T, p, состав) одной моделью.

### 4.1. Основа SAFT/PC-SAFT
- **Chapman, Gubbins, Jackson, Radosz (1990)** — SAFT; **Gross & Sadowski (2001)** — PC-SAFT. Остаточная энергия Гельмгольца:
**a^res = a^hc (твёрдая цепь, hard-chain) + a^disp (дисперсия) + a^assoc (ассоциация/вод. связи)**.
Каждый компонент задаётся тремя параметрами: **m** (число сегментов), **σ** (диаметр сегмента), **ε/k** (энергия дисперсии); для ассоциирующих — доп. параметры энергии/объёма ассоциации.

### 4.2. PC-SAFT для асфальтенов
- **Ting, Hirasaki, Chapman (2003); Gonzalez et al. (2007); Vargas, Chapman (Rice)** — асфальтен как крупный сегмент; **моделирует фазовое поведение в (T, p, состав)**; превосходит кубические EOS даже при минимальных композиционных данных (вплоть до C9+); предсказывает нестабильность при изменении p, T и состава (газ/CO₂) ([выпадение асфальтенов при изменении состава, PC-SAFT, RG](https://www.researchgate.net/publication/231272065_Modeling_of_Asphaltene_Precipitation_Due_to_Changes_in_Composition_Using_the_Perturbed_Chain_Statistical_Associating_Fluid_Theory_Equation_of_State); [фазовое поведение асфальтенов PC-SAFT, RG](https://www.researchgate.net/publication/231273784_Modeling_Asphaltene_Phase_Behavior_in_Crude_Oil_Systems_Using_the_Perturbed_Chain_Form_of_the_Statistical_Associating_Fluid_Theory_PC-SAFT_Equation_of_State); [глава Применение PC-SAFT, Springer](https://link.springer.com/chapter/10.1007/0-387-68903-6_12)).
- Систематические процедуры характеризации крудов/асфальтенов под PC-SAFT и оценка максимума асфальтеновой фазы ([альтернативная характеризация + PC-SAFT, Braz. J. Chem. Eng. 2022](https://link.springer.com/article/10.1007/s43153-022-00296-6); [PC-SAFT для крудов и асфальтенов: систематический обзор, Fuel 2021](https://www.sciencedirect.com/science/article/abs/pii/S0016236121000569)).
- Робастные алгоритмы VLLE/фазовой огибающей на PC-SAFT ([фазовая диаграмма асфальтенов PC-SAFT, FPE 2019](https://www.sciencedirect.com/science/article/abs/pii/S016773221935901X)).

### 4.3. CPA (Cubic-Plus-Association)
- **Li & Firoozabadi (2010)** — **SRK + ассоциативный член**; делит C12+ на тяжёлые УВ и асфальтен; учитывает **само-ассоциацию** асфальтена и **кросс-ассоциацию** асфальтен–тяжёлые УВ; прогноз AOP и влияния CO₂ при изменении давления ([контекст: Li & Firoozabadi 2010, CPA](https://www.sciencedirect.com/science/article/abs/pii/S0378381219301852)).
- **CPA vs PC-SAFT** для AOP и фазовой огибающей — сравнительные исследования (оба точнее кубических solid-моделей; выбор зависит от данных) ([сравнение CPA и PC-SAFT, FPE 2019](https://www.sciencedirect.com/science/article/abs/pii/S0378381219301852); [моделирование онсета CPA и PC-SAFT, RG](https://www.researchgate.net/publication/305313997_Modeling_of_Asphaltene_Onset_Precipitation_Conditions_with_Cubic_Plus_Association_CPA_and_Perturbed_Chain_Statistical_Associating_Fluid_Theory_PC-SAFT_Equation_of_States)).
- Новейшее (2024): параметры ассоциации CPA из **молекулярной динамики и петролеомики**; неизотермические алгоритмы градиента асфальтенов ([ассоциация из MD/петролеомики, Fuel 2024](https://www.sciencedirect.com/science/article/abs/pii/S0016236123031903)); HPHT-эксперимент + PC-SAFT при депрессуризации и закачке газа в «живой» нефти ([HPHT live oil, FPE 2022](https://www.sciencedirect.com/science/article/abs/pii/S0378381222001704)).
- Улучшение кубической PR-solid модели через термодинамические корреляции/усреднение ([cubic-PR solid, Petroleum Science 2019](https://link.springer.com/article/10.1007/s12182-019-00377-1)).

### 4.4. PC-SAFT для парафинов
PC-SAFT и multi-solid + PC-SAFT применяются и к парафину (WAT, масса), особенно для тяжёлого «хвоста» ([MS + PC-SAFT для разных крудов, Fuel 2021](https://www.sciencedirect.com/science/article/abs/pii/S0016236121020810)).

---

# ЧАСТЬ V. Многофазный flash с твёрдыми фазами — вычислительное ядро

При заданных (T, p, **z**) на каждом узле сетки нужно определить **число, тип и составы фаз**, включая твёрдые (wax) и тяжёлую жидкую/твёрдую (asphaltene). Это самая «тяжёлая» вычислительная часть компонентной модели.

### 5.1. Анализ устойчивости фаз
- **Критерий касательной плоскости (Tangent Plane Distance, TPD) — Michelsen (1982)**: фаза устойчива, если TPD ≥ 0 для всех пробных составов. Определяет, нужно ли вводить новую (в т.ч. твёрдую) фазу ([контекст: Michelsen TPD; единый каркас multiphase flash](https://www.academia.edu/8442551/Multiphase_equilibria_calculation_by_direct_minimization_of_Gibbs_free_energy_with_a_global_optimization_method)).
- **Прямая минимизация энергии Гиббса** (Heidemann; Gautam–Seider; Soares и др.) — альтернатива/дополнение TPD; глобальная оптимизация ([минимизация G глобальной оптимизацией, academia](https://www.academia.edu/8442551/Multiphase_equilibria_calculation_by_direct_minimization_of_Gibbs_free_energy_with_a_global_optimization_method)).
- Для асфальтенов: Nghiem и Li применяли TPD; Lira-Galeana — критерий стабильности для твёрдых фаз парафина.

### 5.2. Алгоритмы flash
- Чередование **анализ устойчивости ↔ flash** (Michelsen, унифицированный каркас: nested loop, последовательные подстановки, методы 2-го порядка, частичный Ньютон) ([унифицированный multiphase flash, контекст](https://www.researchgate.net/publication/326870130_Multiphase_isenthalpic_flash_General_approach_and_its_adaptation_to_thermal_recovery_of_heavy_oil)).
- **Trust-region** методы для робастной устойчивости/flash ([trust-region stability & flash, FPE 2013](https://www.sciencedirect.com/science/article/abs/pii/S0378381213004780)).
- Трёхфазный flash с водой; одновременный multiphase flash + устойчивость, включая **твёрдый CO₂** ([одновременный flash с твёрдым CO₂, JCED 2021](https://pubs.acs.org/doi/10.1021/acs.jced.1c00330)).

### 5.3. Изоэнтальпийный (isenthalpic) flash — мост к неизотермике
В тепловых процессах температура — **результат**, а не вход. **Изоэнтальпийный flash** задаёт энтальпию H (а не T) и находит (T, фазы) одновременно — естественная формулировка для **неизотермической** фильтрации/термических МУН.
- **Zhu & Okuno (2016)** — multiphase isenthalpic flash, интегрированный с анализом устойчивости; адаптация к тепловой добыче тяжёлой нефти ([isenthalpic flash + стабильность, Zhu & Okuno, PDF](https://ryosukeokuno.com/wp-content/uploads/2024/10/018_FPE-Multiphase-Isenthalpic-Flash-Zhu-and-Okuno-2016.pdf); [общий подход + тепловая добыча, RG](https://www.researchgate.net/publication/326870130_Multiphase_isenthalpic_flash_General_approach_and_its_adaptation_to_thermal_recovery_of_heavy_oil)).
- *Следствие для диссертации:* связку «энергия ↔ фазовый переход парафина» естественно реализовать через изоэнтальпийный flash с твёрдой фазой.

### 5.4. Численные трудности
Негладкость на границах появления/исчезновения фаз; чувствительность к начальным приближениям; рост стоимости с числом компонентов и фаз — главный барьер для применения SAFT-flash в полномасштабной пластовой симуляции (см. Часть VIII).

---

# ЧАСТЬ VI. Сопряжение компонентной термодинамики с термобарической фильтрацией

### 6.1. Архитектура compositional reservoir simulation
На каждом блоке и временном шаге:
1. перенос компонентов (уравнения сохранения массы каждого i) + Дарси + **уравнение энергии** (для T) → локальные (T, p, **z**);
2. **flash + устойчивость** при локальных (T, p) → доли и составы фаз, в т.ч. твёрдой;
3. **источниковый член осаждения** R_i^dep (кинетика Wang–Civan: поверхность + закупорка + вынос) → изменение φ, k;
4. обновление свойств флюида/породы → следующий шаг.

### 6.2. Реализации
- **Qin, Wang, Sepehrnoori, Pope (2000)** — solid-model асфальтена в компонентной симуляции ([IECR, ACS](https://pubs.acs.org/doi/10.1021/ie990781g)).
- **Nghiem et al.** — solid-model в компонентном симуляторе; **полностью неявные** EOS-симуляторы осаждения при истощении ([FIM-симулятор осаждения, FPE 2015](https://www.sciencedirect.com/science/article/abs/pii/S0378381215001831); [параллельный FIM EOS, SPE-120203](https://onepetro.org/SPEATCE/proceedings-abstract/08ATCE/08ATCE/SPE-120203-STU/145597)).
- **Tabzar et al. (2018)** — полная связанная система (нефть/газ/вода/асфальтен) с Nghiem-термодинамикой и 3-членной кинетикой; Ньютон–Рафсон ([OGST 2018, full text](https://ogst.ifpenergiesnouvelles.fr/articles/ogst/full_html/2018/01/ogst180048/ogst180048.html)).
- **Коммерческие:** **CMG-GEM** (компонентный, модуль асфальтенов — [CMG-GEM кейс, RG](https://www.researchgate.net/publication/267869547_Compositional_simulation_of_deposition_flocculation_and_precipitation_of_Asphaltene_in_oil_reservoir_using_CMG-GEM)); **CMG-STARS** (термический); **PVTsim Nova/Multiflash** (парафин+асфальтен, экспорт в OLGA/PIPESIM — [Calsep](https://calsep.com/pvtsim-nova/find-a-pvtsim-package/flow-assurance/)).
- **Термобарический путь → карта зон осаждения:** осаждение локализуется там, где путь пересекает WAT/AOP (часто **околоскважинная зона**: максимум ∇p и охлаждения).

### 6.3. Современные эффективные реализации (2023–2025)
Эффективные схемы внедрения выпадения асфальтена в симуляцию ([effective approach, 2024](https://www.sciencedirect.com/science/article/pii/S2949891024008777)); CO₂ в глубоких пластах ([SPE 2023](https://onepetro.org/SPEATCE/proceedings-abstract/23ATCE/3-23ATCE/535420)).

---

# ЧАСТЬ VII. Старое vs современное — сравнительная таблица

| Подход | Термодинам. база | Учёт T | Учёт p | Компонентное разрешение | Сильные/слабые стороны | Репрезентативные источники |
|---|---|---|---|---|---|---|
| SS (твёрдый раствор) | EOS+G^E (регуляр./полимер. раствор) | да (главное) | слабо | псевдокомп. C7+ | прост; переоценивает лёгкие | Won 1986; Hansen 1988; Pedersen 1991 |
| Multi-solid | PR-EOS + чистые тв. фазы + Пойнтинг | да | да (Пойнтинг) | покомпонентно | точен для тяжёлых; ступенчатая кривая | Lira-Galeana 1996 |
| Predictive UNIQUAC | неск. твёрдых растворов | да | да | покомпонентно | баланс точности; параметры предиктивны | Coutinho 1998 |
| Непрерывная термодинамика | распределение F(I)+EOS | да | да | непрерывно (моменты/квадратура) | мало параметров; приближённость | Cotterman–Prausnitz 1985; Rätzsch–Kehlen 1983 |
| Асфальтен растворный | Флори–Хаггинс/Скэтчард–Хильдебранд | да | да | мальтены/асфальтены | обратимо; ограничен | Hirschberg 1984 |
| Асфальтен коллоидный | коллоид + смолы-стабилизаторы | частично | да | асфальтен/смолы | необратимость; качественный | Leontaritis–Mansoori 1987 |
| Асфальтен solid-model | PR-EOS + тв. фаза (2 солида) | да | да | C31+ split | стандарт в ПО; калибруемый | Nghiem 1993 |
| Асфальтен мицеллярный | LLE + мицеллы | да | да | асфальтен/смолы/мономеры | физичный; сложный | Pan–Firoozabadi 1997 |
| PC-SAFT | стат. теория (hc+disp+assoc) | да | да | m,σ,ε по компонентам | точен в (T,p,состав); дорог | Ting 2003; Gonzalez 2007 |
| CPA | SRK + ассоциация | да | да | само/кросс-ассоциация | хорош для AOP; калибровка | Li–Firoozabadi 2010 |

---

# Пробелы и противоречия

**Противоречия (фиксировать, не усреднять):**
1. **SS vs Multi-solid vs SAFT** — нет единой «лучшей» термодинамики; SS переоценивает лёгкие, MS даёт негладкую кривую, SAFT точнее, но дороже и чувствительнее к параметрам.
2. **Обратимость осаждения асфальтенов** — Hirschberg (обратимо) vs Leontaritis–Mansoori (необратимо); современные модели вводят (частичную) обратимость как параметр.
3. **Эффект давления не универсален** — для «живой» нефти газ повышает WAT; в газоконденсатах эффект давления немонотонный/обратный.
4. **CPA vs PC-SAFT** для AOP — обе превосходят кубические, но ранжирование зависит от флюида и данных.
5. **Характеризация C7+** — разные методы дают разброс предсказанной WAT/AOP; это **доминирующий источник погрешности**, перекрывающий различия термодинамических моделей.

**Пробелы (открытые задачи — релевантны диссертации):**
1. **Изоэнтальпийный flash с твёрдой фазой в пластовом симуляторе** — почти не реализован для парафина при неизотермической фильтрации (хотя аппарат Zhu–Okuno готов).
2. **SAFT-термодинамика в полномасштабной компонентной симуляции пласта** — барьер вычислительной стоимости flash; нужны суррогаты/таблицы/ML-ускорение.
3. **Совместное (co-precipitation) компонентное моделирование парафин + асфальтен + смолы** вдоль термобарического пути — почти не разработано.
4. **Непрерывная термодинамика для твёрдой фазы в фильтрации** — потенциально дёшева, но мало применяется из-за резких границ S-фазы.
5. **ML/PINN-ускорение flash и характеризации** в связке с термобарической фильтрацией — зарождающееся направление.

---

# Итоговый список ключевых источников (BibTeX — новые относительно прошлых отчётов)

```bibtex
@article{Whitson1983,
  author = {Whitson, C. H.},
  title = {Characterizing hydrocarbon plus fractions},
  journal = {Society of Petroleum Engineers Journal}, volume = {23}, number = {4}, pages = {683--694}, year = {1983}}

@article{KatzFiroozabadi1978,
  author = {Katz, D. L. and Firoozabadi, A.},
  title = {Predicting phase behavior of condensate/crude-oil systems using methane interaction coefficients},
  journal = {Journal of Petroleum Technology}, volume = {30}, number = {11}, pages = {1649--1655}, year = {1978}}

@article{CottermanPrausnitz1985,
  author = {Cotterman, R. L. and Prausnitz, J. M.},
  title = {Flash calculations for continuous or semicontinuous mixtures by use of an equation of state},
  journal = {Industrial \& Engineering Chemistry Process Design and Development}, volume = {24}, number = {2}, pages = {434--443}, year = {1985}}

@article{RatzschKehlen1983,
  author = {R{\"a}tzsch, M. T. and Kehlen, H.},
  title = {Continuous thermodynamics of complex mixtures},
  journal = {Fluid Phase Equilibria}, volume = {14}, pages = {225--234}, year = {1983}}

@article{Broadhurst1962,
  author = {Broadhurst, M. G.},
  title = {An analysis of the solid phase behavior of the normal paraffins},
  journal = {Journal of Research of the National Bureau of Standards A}, volume = {66A}, number = {3}, pages = {241--249}, year = {1962}}

@article{Michelsen1982a,
  author = {Michelsen, M. L.},
  title = {The isothermal flash problem. Part I. Stability},
  journal = {Fluid Phase Equilibria}, volume = {9}, number = {1}, pages = {1--19}, year = {1982}}

@article{Michelsen1982b,
  author = {Michelsen, M. L.},
  title = {The isothermal flash problem. Part II. Phase-split calculation},
  journal = {Fluid Phase Equilibria}, volume = {9}, number = {1}, pages = {21--40}, year = {1982}}

@article{GrossSadowski2001,
  author = {Gross, J. and Sadowski, G.},
  title = {Perturbed-chain SAFT: an equation of state based on a perturbation theory for chain molecules},
  journal = {Industrial \& Engineering Chemistry Research}, volume = {40}, number = {4}, pages = {1244--1260}, year = {2001}}

@article{Chapman1990,
  author = {Chapman, W. G. and Gubbins, K. E. and Jackson, G. and Radosz, M.},
  title = {New reference equation of state for associating liquids},
  journal = {Industrial \& Engineering Chemistry Research}, volume = {29}, number = {8}, pages = {1709--1721}, year = {1990}}

@article{Ting2003,
  author = {Ting, P. D. and Hirasaki, G. J. and Chapman, W. G.},
  title = {Modeling of asphaltene phase behavior with the SAFT equation of state},
  journal = {Petroleum Science and Technology}, volume = {21}, number = {3-4}, pages = {647--661}, year = {2003}}

@article{Gonzalez2007,
  author = {Gonzalez, D. L. and Ting, P. D. and Hirasaki, G. J. and Chapman, W. G.},
  title = {Modeling of asphaltene precipitation due to changes in composition using the perturbed chain SAFT equation of state},
  journal = {Energy \& Fuels}, volume = {21}, number = {3}, pages = {1231--1242}, year = {2007}}

@article{LiFiroozabadi2010,
  author = {Li, Z. and Firoozabadi, A.},
  title = {Cubic-plus-association equation of state for asphaltene precipitation in live oils},
  journal = {Energy \& Fuels}, volume = {24}, number = {5}, pages = {2956--2963}, year = {2010}}

@article{ZhuOkuno2016,
  author = {Zhu, D. and Okuno, R.},
  title = {Multiphase isenthalpic flash integrated with stability analysis},
  journal = {Fluid Phase Equilibria}, volume = {423}, pages = {203--219}, year = {2016}}

@article{Pauly2007,
  author = {Pauly, J. and Daridon, J.-L. and Coutinho, J. A. P.},
  title = {A new predictive thermodynamic model for wax formation phenomena at high pressure},
  journal = {Fluid Phase Equilibria}, volume = {255}, pages = {193--199}, year = {2007}}

@article{Qin2000,
  author = {Qin, X. and Wang, P. and Sepehrnoori, K. and Pope, G. A.},
  title = {Modeling asphaltene precipitation in reservoir simulation},
  journal = {Industrial \& Engineering Chemistry Research}, volume = {39}, number = {8}, pages = {2644--2654}, year = {2000}}
```

> Тома/страницы/годы для ряда классических работ (Whitson 1983; Cotterman–Prausnitz 1985; Michelsen 1982; Li–Firoozabadi 2010; Ting 2003; Gonzalez 2007; Zhu–Okuno 2016) приведены по памяти/аннотациям — сверьте по издателю/DOI.

---

# Выводы для диссертации (компонентный модуль `paraphin`)
1. **Термодинамическое ядро:** для парафина — multi-solid (Lira-Galeana) или predictive UNIQUAC (Coutinho); при высоком давлении/«живой» нефти — добавить **поправку Пойнтинга** и при необходимости PC-SAFT. Для асфальтена — Nghiem solid-model или CPA/PC-SAFT.
2. **Характеризация C7+** (Whitson-гамма + распределение н-парафинов CₙH₂ₙ₊₂ + корреляции T_fus(n), ΔH_fus(n) Broadhurst) — заложить как отдельный, тщательно валидируемый модуль: это главный источник погрешности.
3. **Flash-ядро:** анализ устойчивости TPD (Michelsen) + multiphase flash с твёрдой фазой; для неизотермики — **изоэнтальпийная** формулировка (Zhu–Okuno).
4. **Сопряжение с фильтрацией:** на каждом узле — flash при локальных (T,p) вдоль термобарического пути → источник осаждения → φ,k (Часть VI прошлых отчётов).
5. **Производительность:** предусмотреть кэширование/таблицы/ML-суррогат flash (открытая задача) — иначе SAFT-flash в полномасштабной 3D-симуляции дорог.

---

## Методология
Выполнено ~24 поисковых запроса суммарно (в т.ч. ~11 в этом раунде, англ./рус.); широкий охват → углубление по характеризации, классическим и SAFT/CPA-моделям, flash-алгоритмам и термобарике; полнотекстовое чтение открытых источников (Tabzar et al. 2018 ранее; попытка PC-SAFT-главы Rice — PDF бинарный). Противоречия зафиксированы явно; библиографические детали части классических работ рекомендуется сверить по первоисточникам.
