# Реология парафинистой нефти и гелеобразование, течение вязкопластичной нефти в пористой среде

Статьи блока геля (`paraphin/equations/Gel.py`, `utils/math_utils/fluids_correlations.py`;
`docs/модель_АСПО.docx`, разд. 7). PDF — `fetch_open_access.py`.

## Использованы в модели

| Файл (после скачивания) | Источник | Что взято в модель | Доступ |
|---|---|---|---|
| `pedersen_ronningsen2000_wax_viscosity.pdf` | Pedersen K.S., Rønningsen H.P. Effect of precipitated wax on viscosity — a model for predicting non-Newtonian viscosity of crude oils // Energy Fuels. 2000. 14(1):43–51. doi:10.1021/ef9901185 | Ньютоновская часть mu_L·exp(D·phi) (`wax_viscosity = 1`, `visc_D`); показано, что их слагаемые ~phi⁴/γ̇ при пластовых скоростях сдвига расходятся — их роль играет предел текучести | платный |
| `letoffe1995_wax_dsc_thermomicroscopy.pdf` | Létoffé J.M., Claudy P., Kok M.V., Garcin M., Volle J.L. Crude oils: characterization of waxes precipitated on cooling by d.s.c. and thermomicroscopy // Fuel. 1995. 74(6):810–817. doi:10.1016/0016-2361(94)00006-D | Застывание при нескольких процентах выпавшего парафина — существование порога `gel_phi` | платный |
| `venkatesan2005_paraffin_gel_strength.pdf` | Venkatesan R. et al. The strength of paraffin gels formed under static and flow conditions // Chem. Eng. Sci. 2005. 60(13):3587–3598. doi:10.1016/j.ces.2005.02.045 | Прочность геля растет с долей твердой фазы; при сдвиге гель слабее — поэтому порог по реометру (150 1/с) выше статического | платный |
| `visintin2005_waxy_gel_rheology.pdf` | Visintin R.F.G. et al. Rheological behavior and structural interpretation of waxy crude oil gels // Langmuir. 2005. 21(14):6240–6249. doi:10.1021/la050705k | Гель парафинистой нефти как коллоидный (фрактальный) гель — степенной закон tau_y(phi) | платный |
| `shih1990_colloidal_gel_scaling.pdf` | Shih W.-H., Shih W.Y., Kim S.-I., Liu J., Aksay I.A. Scaling behavior of the elastic properties of colloidal gels // Phys. Rev. A. 1990. 42(8):4772–4779. doi:10.1103/PhysRevA.42.4772 | Степенная зависимость прочности геля от доли частиц (`yield_stress`) | платный |
| `dimitriou2014_waxy_crude_thixotropy.pdf` | Dimitriou C.J., McKinley G.H. A comprehensive constitutive law for waxy crude oil: a thixotropic yield stress fluid // Soft Matter. 2014. 10(35):6619–6644. doi:10.1039/C4SM00578C | Тиксотропия — релаксация множителя подвижности к равновесному (`gel_time`) | платный |
| `singh2000_wax_oil_gel_aging.pdf` | Singh P., Venkatesan R., Fogler H.S., Nagarajan N. Formation and aging of incipient thin film wax–oil gels // AIChE J. 2000. 46(5):1059–1074. doi:10.1002/aic.690460517 | Осадок на стенке — гель с захваченной нефтью: в поре сетку образуют и осадок, и взвесь (`pore_solid_fraction`) | платный |
| `wu_pruess1992_bingham_porous_media.pdf` | Wu Y.-S., Pruess K., Witherspoon P.A. Flow and displacement of Bingham non-Newtonian fluids in porous media // SPE Reserv. Eng. 1992. 7(3):369–376. doi:10.2118/20051-PA (отчет LBL — OSTI 7081417) | Бингамовская нефть в пористой среде: начальный градиент давления из пучка капилляров | платный (отчет OSTI открыт) |
| `chevalier2013_darcy_yield_stress.pdf` | Chevalier T. et al. Darcy's law for yield stress fluid flowing through a porous medium // J. Non-Newtonian Fluid Mech. 2013. 195:57–66. doi:10.1016/j.jnnfm.2012.12.005 | Закон Дарси для вязкопластичной жидкости: критический перепад + расходная часть — множитель Phi(\|grad p\|) | платный |
| `liu2019_darcy_yield_stress_prl_arxiv.pdf` | Liu C., De Luca A., Rosso A., Talon L. Darcy's law for yield stress fluids // Phys. Rev. Lett. 2019. 122:245502. doi:10.1103/PhysRevLett.122.245502; arXiv:1811.09494 | Постепенное включение каналов по мере роста перепада — как в интеграле по fi(r) | открыт (arXiv) |
| `jtac2014_solid_fraction_gel_strength.pdf` | Thermal behavior and solid fraction dependent gel strength model of waxy oils // J. Therm. Anal. Calorim. 2014. doi:10.1007/s10973-014-3660-3 | Прочность геля как функция доли твердой фазы по ДСК | платный |
| — | Buckingham E. On plastic flow through capillary tubes // Proc. ASTM. 1921. 21:1154–1156 | Расход бингамовской жидкости в трубе: F(ξ) = 1 − 4ξ/3 + ξ⁴/3 (`br_factor`) | классика, в открытом доступе не найдена |

## Данные калибровки и проверки (уже в репозитории)

| Файл | Что взято |
|---|---|
| `../парафиновое/li2024_zhetybai_cooling_damage.pdf` | Вязкость при охлаждении, 150 1/с (рис. 4а) — `visc_D`, `gel_phi`, `gel_tau_ref`, `gel_n` бингамовским разложением (`experiments/calibrate.py`, разд. C) |
| `../парафиновое/togasheva2026_uzen_cooling_WAT.pdf` | Узень: объемная кристаллизация 33.5–35 C, потеря текучести 25–34 C, температура застывания ~30 C — качественная проверка |
| `../парафиновое/he2020_changchunling_cold_damage.pdf` | Резкое падение проницаемости у температуры застывания 14.6–18.2 C — качественная проверка |
| `../парафиновое/abileva2025_phd_satbayev.pdf` | Фильтрация вязкопластичных нефтей (разд. 3.1–3.2) |
| — | «Special features in exploitation of Uzen field, caused by persistence of initial pressure gradient» (ETDEWEB 6603213): начальный градиент давления на Узене — промысловый признак геля |
