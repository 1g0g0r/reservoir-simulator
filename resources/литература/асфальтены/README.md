# Асфальтены и смолы: устойчивость, флокуляция, осаждение в порах

Статьи блока асфальтенов (`paraphin/equations/Asphaltene.py`, `Qp_m_k_fi.calc_qp_m_k_fi_2`,
`oil_composition.py`; `docs/модель_АСПО.docx`, разд. 4, 6). Обзор — `твт_статья/full_review.pdf`, разд. 5.
PDF — `fetch_open_access.py`.

## Использованы в модели

| Файл (после скачивания) | Источник | Что взято в модель | Доступ |
|---|---|---|---|
| `hirschberg1984_asphaltene_flocculation.pdf` | Hirschberg A., deJong L.N.J., Schipper B.A., Meijer J.G. Influence of temperature and pressure on asphaltene flocculation // SPEJ. 1984. 24(3):283–293. doi:10.2118/11202-PA | Предельная растворенная доля по Флори–Хаггинсу phi_max = exp[v_a/v_m − 1 − v_a(δ_a − δ_m)²/RT] (`asph_soluble`), калибровка δ_a на давление начала осаждения | платный |
| `leontaritis1987_colloidal_asphaltene.pdf` | Leontaritis K.J., Mansoori G.A. Asphaltene flocculation during oil production and processing: a thermodynamic colloidal model // SPE-16258-MS. 1987. doi:10.2118/16258-MS | Коллоидная природа, пептизация смолами, частичная необратимость (k_redis < k_floc), соосаждение смол | платный |
| `wang_buckley2001_onset_two_component.pdf` | Wang J.X., Buckley J.S. A two-component solubility model of the onset of asphaltene flocculation in crude oils // Energy Fuels. 2001. 15(5):1004–1012. doi:10.1021/ef010012l | δ пропорционален плотности (правило «одной трети») — зависимость δ_m от P и T (`delta_maltene`) | платный |
| `buckley1998_asphaltene_solvent_properties.pdf` | Buckley J.S. et al. Asphaltene precipitation and solvent properties of crude oils // Pet. Sci. Technol. 1998. 16(3–4):251–285. doi:10.1080/10916469808949783 | То же: связь параметра растворимости с плотностью/показателем преломления | платный |
| `akbarzadeh2005_regular_solution_asphaltene.pdf` | Akbarzadeh K., Alboudwarej H., Svrcek W.Y., Yarranton H.W. A generalized regular solution model for asphaltene precipitation from n-alkane diluted heavy oils and bitumens // Fluid Phase Equilib. 2005. 232:159–170. doi:10.1016/j.fluid.2005.03.029 | Параметры растворимости SARA-фракций: насыщенные 16.4, ароматика 20.3, смолы 19.3 МПа^0.5 | открыт (сайт U. Calgary) |
| `yen2001_cii_inhibitors.pdf` | Yen A., Yin Y.R., Asomaning S. Evaluating asphaltene inhibitors: laboratory tests and field studies // SPE-65376-MS. 2001. doi:10.2118/65376-MS | Индекс коллоидной неустойчивости CII, пороги 0.7/0.9 (выгрузка `CII`) | платный |
| `maqbool2009_asphaltene_kinetics.pdf` | Maqbool T., Balgoa A.T., Fogler H.S. Revisiting asphaltene precipitation from crude oils: a case of neglected kinetic effects // Energy Fuels. 2009. 23(7):3681–3686. doi:10.1021/ef9002236 | Кинетика флокуляции (часы — месяцы) — релаксация `floc_relax`, k_floc | платный |
| `mullins2012_yen_mullins_model.pdf` | Mullins O.C. et al. Advances in asphaltene science and the Yen–Mullins model // Energy Fuels. 2012. 26:3986–4003. doi:10.1021/ef300185p | Иерархия молекула — наноагрегат — кластер, мольный объем асфальтенов | платный |
| `wang_civan2001_productivity_decline_asphaltene.pdf` | Wang S., Civan F. Productivity decline of vertical and horizontal wells by asphaltene deposition in petroleum reservoirs // SPE-64991-MS. 2001. doi:10.2118/64991-MS | Поверхностное осаждение + закупорка горл + вынос; наше сужение флокулами — поверхностный член, выведенный из течения в капилляре | платный |
| `tabzar2018_asphaltene_multiphase_ogst.pdf` | Tabzar A. et al. Multiphase flow modeling of asphaltene precipitation and deposition // Oil Gas Sci. Technol. 2018. 73:51. doi:10.2516/ogst/2018039 | Выпадение растет при падении давления к P_b; полная связка Wang–Civan с фильтрацией — образец постановки | открыт |
| `nabzar2008_colloidal_asphaltene_ogst.pdf` | Nabzar L., Aguiléra M.E. The colloidal approach. A promising route for asphaltene deposition modelling // Oil Gas Sci. Technol. 2008. 63(1):21–35. doi:10.2516/ogst:2007083 | Осаждение асфальтенов как коллоидных частиц (диффузия к стенке), обоснование кинетики сужения | открыт |
| `darabi2014_asphaltene_wellbore_reservoir.pdf` | Darabi H., Shirdel M., Kalaei M.H., Sepehrnoori K. Aspects of modeling asphaltene deposition in a compositional coupled wellbore/reservoir simulator // SPE-169121-MS. 2014. doi:10.2118/169121-MS | Зона максимального осаждения — не на забое, а при P ≈ P_b (колокол по давлению) | платный |
| `scirep2022_flory_huggins_inhibitors.pdf` | Modelling the effect of the inhibitors on asphaltene precipitation using Flory–Huggins theory // Sci. Rep. 2022. doi:10.1038/s41598-022-23596-w | Современная формулировка Флори–Хаггинса для асфальтенов — сверка формулы | открыт |
| `kor2017_asphaltene_models_well.pdf` | Kor P., Kharrat R., Ayoubi A. Comparison and evaluation of several models in prediction of asphaltene deposition profile along an oil well // JPEPT. 2017. 7:497–510. doi:10.1007/s13202-016-0269-z | Сравнение моделей осаждения — к ограничениям | открыт |
| `kargarpour2016_marrat_asphaltene.pdf` | Kargarpour M., Dandekar A.Y. Analysis of asphaltene deposition in Marrat oil well string: a new approach // JPEPT. 2016. 6:845–856. doi:10.1007/s13202-015-0221-7 | Промысловый кейс — к ограничениям | открыт |

## Уже в репозитории

| Файл | Что взято |
|---|---|
| `../Model-Assisted Analysis of Simultaneous Paraffin and Asphaltene Deposition in Laboratory Core Tests.pdf` | Wang S., Civan F. // J. Energy Resour. Technol. 2005. 127(4):318–322. doi:10.1115/1.1924466. Совместное осаждение парафина и асфальтенов на опытах Sutton & Roberts (0.7 и 0.1 % асфальтенов, P_b = 2050 psia); растворимость парафина — регулярный раствор, асфальтенов — полимерный; один поровый объем на оба осадка — как в `calc_qp_m_k_fi_2` |
| `../парафиновое/solaimany-nazar2011.pdf` | Модель осаждения асфальтенов в пласте при истощении |
| `../mehanizmy-obrazovaniya-asfaltosmoloparafinovyh-otlozheniy-i-faktory-intensivnosti-ih-formirovaniya.pdf` | Механизмы образования АСПО (обзор) |
