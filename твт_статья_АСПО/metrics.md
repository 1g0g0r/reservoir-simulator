# Числа статьи

Создается `python твт_статья_АСПО/make_article.py`, руками не править.

- `SR_OWN1`: 0.034
- `SR_WC1`: 0.040
- `SR_LEG1`: 0.048
- `SR_NOISE1`: 0.029
- `SR_VISC1`: 0.037
- `SR_EARLY_OWN`: 0.70
- `SR_WC_01`: 0.89
- `SR_WC_025`: 0.75
- `SR_EXP_PV1`: 0.25
- `SR_EXP_K1`: 0.70
- `SR_OWN2`: 0.037
- `SR_WC2`: 0.060
- `SR_LEG2`: 0.248
- `SR_NOISE2`: 0.025
- `SR_VISC2`: 0.054
- `LI_90`: 0.019
- `LI_NOISE90`: 0.006
- `LI_65`: 0.027
- `LI_NOISE65`: 0.007
- `LI_45`: 0.042
- `LI_NOISE45`: 0.007
- `LI_25`: 0.013
- `LI_NOISE25`: 0.005
- `LI_LEG90`: 0.222
- `LI_DH`: 137
- `LI_FILM`: 0.082
- `LI_DCR`: 13.5
- `LI_DIFF`: 0.10
- `LI_KCR`: 0.016
- `LI_TCR`: 63
- `LI_HOLD`: 1
- `LI_CUM25`: 0.012
- `LI_NET_FIT45`: 0.428
- `LI_NET_D`: 2–7
- `LI_NET45`: 0.034–0.037
- `LI_NET_DALL`: 2–10
- `LI_NET25`: 0.24–0.33
- `SD_RMS`: 0.036
- `SD_RMSLG`: 0.034
- `SD_NOISE`: 0.119
- `SD_LEG`: 0.255
- `SD_WAT`: 34.8
- `SD_WAT_AUTH`: 33.8
- `SD_MCOND`: 0.17
- `SD_MEXP`: 0.23
- `SD_FACTOR`: 1.19
- `SD_KCR`: 3.7e-04
- `SD_TCR`: 45
- `SD_KWALL`: 1.1e-03
- `SD_T10_0_25`: 34.2
- `SD_FIN_0_25`: 959
- `SD_DA_0_25`: 5.3
- `SD_T10_0_5`: 33.9
- `SD_FIN_0_5`: 982
- `SD_DA_0_5`: 2.6
- `SD_T10_1`: 33.5
- `SD_FIN_1`: 54
- `SD_DA_1`: 1.3
- `SD_T10_2`: 32.9
- `SD_FIN_2`: 14
- `SD_DA_2`: 0.7
- `SD_T10_4`: —
- `SD_FIN_4`: 3
- `SD_DA_4`: 0.3
- `SD_DT10`: 1.2
- `SD_DT10_LO`: 0.25
- `SD_DT10_HI`: 2
- `SD_RATE_SPAN`: 3.1
- `SD_RATE_THR`: 0.16
- `SD_RATE_RMS`: 0.2
- `SD_RATE_LI_THR`: 10
- `SD_RATE_LI_RMS`: 0.4
- `SD_BULK20_RHEO`: 30.4
- `SD_BULK20_CORE`: 34.4
- `SD_BULK20_SHIFT`: 4.0
- `SD_WAT_BULK`: 30.0
- `HE_BUNDLE`: 0.281
- `HE_NET`: 0.044
- `HE_NET0`: 0.046
- `HE_CONSTR`: 0.037
- `HE_POWER`: 0.039
- `HE_POWER_N`: 5.7
- `HE_LATTICE`: 0.008
- `ASPH_LIN_M`: 0.61
- `ASPH_LIN_K`: 0.034
- `ASPH_LIN_BUNDLE`: 0.51
- `ASPH_LIN_NET`: 0.026
- `ASPH_LIN_LAT`: 0.032
- `ASPH_ST_M`: 0.80
- `ASPH_ST_K`: 0.89
- `ASPH_ST_BUNDLE`: 0.73
- `ASPH_ST_NET`: 0.254
- `ASPH_ST_LAT`: 0.266
- `HE_N`: 19
- `ML_WAX_HI`: 24.9
- `ML_WAX_LO`: 5
- `ML_K_HI`: 0.98
- `ML_K_LO`: 1.00
- `ML_M_HI`: 1.2
**T1**

| Опыт | Шумовой порог | Упрощенная модель | Настоящая модель | Другие модели |
|---|---|---|---|---|
| Sutton, Roberts, опыт 1 | 0.029 | 0.048 | 0.034 | 0.040 |
| Sutton, Roberts, опыт 2 | 0.025 | 0.248 | 0.037 | 0.060 |
| Li и др., 90°C | 0.006 | 0.222 | 0.019 | — |
| Li и др., 65°C | 0.007 | закупорка | 0.027 | — |
| Li и др., 45°C | 0.007 | закупорка | 0.042 | — |
| Li и др., 25°C | 0.005 | закупорка | 0.013 | — |
| Sandyga и др., охлаждение | — | 0.255 | 0.036 | — |
| He и др., k(m) | — | 0.281 | 0.044 | 0.039 |

- `RMS_MIN`: 0.013
- `RMS_MAX`: 0.044
- `LEG_MIN`: 0.05
- `LEG_MAX`: 0.28
- `F_MAP_SIDE`: 200
- `F_MAP_YEARS`: 1, 3, 5
**T2**

| Вариант | КИН | ΔКИН | Закачано, тыс. м³ | Скин-фактор нагнетательной скважины | Парафин в осадке, т | Ниже WAT, % площади | Гель, % площади | Средняя температура, °C |
|---|---|---|---|---|---|---|---|---|
| базовый: 20°C, Винсом—Вестервельд | 0.329 | — | 95.6 | 1.6 | 390 | 15 | 7 | 49.9 |
| изотермическая закачка 55°C | 0.363 | +0.034 | 142.4 | 0.0 | 0 | 0 | 0 | 55.0 |
| закачка 40°C | 0.358 | +0.029 | 134.3 | 0.0 | 0 | 0 | 0 | 51.8 |
| закачка 5°C | 0.307 | −0.022 | 73.3 | 3.1 | 685 | 14 | 9 | 49.5 |
| без теплообмена с кровлей и подошвой | 0.323 | −0.006 | 88.7 | 1.6 | 1530 | 25 | 26 | 43.6 |
| теплообмен по Ловерье | 0.328 | −0.002 | 93.6 | 1.6 | 576 | 23 | 9 | 47.6 |
| нефть без парафина | 0.348 | +0.019 | 119.5 | 0.0 | 0 | 0 | 0 | 48.4 |
| без парафина и без теплообмена | 0.345 | +0.016 | 115.4 | 0.0 | 0 | 0 | 0 | 39.8 |
| без геля | 0.335 | +0.006 | 102.1 | 1.5 | 416 | 16 | 0 | 49.5 |
| с удержанием смол и асфальтенов | 0.249 | −0.081 | 37.9 | 27.9 | 124 | 5 | 2 | 53.1 |
| без скрытой теплоты | 0.329 | 0.000 | 95.4 | 1.6 | 393 | 15 | 7 | 49.9 |
| равновесная кристаллизация | 0.318 | −0.011 | 83.5 | 3.3 | 330 | 13 | 6 | 50.6 |
| равновесие без давления и растворенного газа | 0.344 | +0.015 | 111.4 | 2.2 | 817 | 23 | 13 | 49.0 |
| тепловое неравновесие (LTNE) | 0.329 | 0.000 | 95.6 | 1.6 | 390 | 15 | 7 | 49.9 |

- `F_BASE_RF`: 0.329
- `F_BASE_DRF`: +0.0000
- `F_BASE_DRF_ABS`: 0.000
- `F_BASE_INJ`: 95.6
- `F_BASE_SKIN`: 1.6
- `F_BASE_KINJ`: 0.568
- `F_BASE_KMIN`: 0.51
- `F_BASE_WAX`: 390
- `F_BASE_RET`: 0.0
- `F_BASE_TMEAN`: 49.9
- `F_BASE_GEL`: 7
- `F_BASE_WCUT`: 91
- `F_BASE_Q0`: 103
- `F_BASE_QMIN`: 103
- `F_BASE_QEND`: 309
- `F_BASE_COLD`: 14.0
- `F_BASE_BELOW`: 15.2
- `F_BASE_WAXMAX`: 6.0
- `F_BASE_PMEAN`: 13.3
- `F_BASE_KPROD`: 0.51
- `F_BASE_ASPHPROD`: 36
- `F_BASE_ASPH`: 23
- `F_BASE_QDROP`: —
- `F_BASE_PB`: —
- `F_BASE_WAT0`: 40.0
- `F_TISO_RF`: 0.363
- `F_TISO_DRF`: +0.0338
- `F_TISO_DRF_ABS`: 0.034
- `F_TISO_INJ`: 142.4
- `F_TISO_SKIN`: 0.0
- `F_TISO_KINJ`: 1.000
- `F_TISO_KMIN`: 0.63
- `F_TISO_WAX`: 0
- `F_TISO_RET`: 0.0
- `F_TISO_TMEAN`: 55.0
- `F_TISO_GEL`: 0
- `F_TISO_WCUT`: 94
- `F_TISO_Q0`: 103
- `F_TISO_QMIN`: 103
- `F_TISO_QEND`: 549
- `F_TISO_COLD`: 0.0
- `F_TISO_BELOW`: 0.0
- `F_TISO_WAXMAX`: 0.0
- `F_TISO_PMEAN`: 15.0
- `F_TISO_KPROD`: 0.63
- `F_TISO_ASPHPROD`: 26
- `F_TISO_ASPH`: 16
- `F_TISO_QDROP`: —
- `F_TISO_PB`: —
- `F_TISO_WAT0`: 40.0
- `F_T40_RF`: 0.358
- `F_T40_DRF`: +0.0287
- `F_T40_DRF_ABS`: 0.029
- `F_T40_INJ`: 134.3
- `F_T40_SKIN`: 0.0
- `F_T40_KINJ`: 1.000
- `F_T40_KMIN`: 0.61
- `F_T40_WAX`: 0
- `F_T40_RET`: 0.0
- `F_T40_TMEAN`: 51.8
- `F_T40_GEL`: 0
- `F_T40_WCUT`: 94
- `F_T40_Q0`: 103
- `F_T40_QMIN`: 103
- `F_T40_QEND`: 503
- `F_T40_COLD`: 20.6
- `F_T40_BELOW`: 0.0
- `F_T40_WAXMAX`: 0.0
- `F_T40_PMEAN`: 14.7
- `F_T40_KPROD`: 0.61
- `F_T40_ASPHPROD`: 27
- `F_T40_ASPH`: 17
- `F_T40_QDROP`: —
- `F_T40_PB`: —
- `F_T40_WAT0`: 40.0
- `F_T5_RF`: 0.307
- `F_T5_DRF`: −0.0224
- `F_T5_DRF_ABS`: 0.022
- `F_T5_INJ`: 73.3
- `F_T5_SKIN`: 3.1
- `F_T5_KINJ`: 0.404
- `F_T5_KMIN`: 0.40
- `F_T5_WAX`: 685
- `F_T5_RET`: 0.0
- `F_T5_TMEAN`: 49.5
- `F_T5_GEL`: 9
- `F_T5_WCUT`: 88
- `F_T5_Q0`: 103
- `F_T5_QMIN`: 103
- `F_T5_QEND`: 217
- `F_T5_COLD`: 10.3
- `F_T5_BELOW`: 13.8
- `F_T5_WAXMAX`: 10.7
- `F_T5_PMEAN`: 12.0
- `F_T5_KPROD`: 0.54
- `F_T5_ASPHPROD`: 33
- `F_T5_ASPH`: 33
- `F_T5_QDROP`: —
- `F_T5_PB`: —
- `F_T5_WAT0`: 40.0
- `F_HL0_RF`: 0.323
- `F_HL0_DRF`: −0.0065
- `F_HL0_DRF_ABS`: 0.006
- `F_HL0_INJ`: 88.7
- `F_HL0_SKIN`: 1.6
- `F_HL0_KINJ`: 0.563
- `F_HL0_KMIN`: 0.49
- `F_HL0_WAX`: 1530
- `F_HL0_RET`: 0.0
- `F_HL0_TMEAN`: 43.6
- `F_HL0_GEL`: 26
- `F_HL0_WCUT`: 90
- `F_HL0_Q0`: 103
- `F_HL0_QMIN`: 103
- `F_HL0_QEND`: 276
- `F_HL0_COLD`: 31.9
- `F_HL0_BELOW`: 25.1
- `F_HL0_WAXMAX`: 6.2
- `F_HL0_PMEAN`: 13.1
- `F_HL0_KPROD`: 0.49
- `F_HL0_ASPHPROD`: 37
- `F_HL0_ASPH`: 25
- `F_HL0_QDROP`: —
- `F_HL0_PB`: —
- `F_HL0_WAT0`: 40.0
- `F_HL1_RF`: 0.328
- `F_HL1_DRF`: −0.0019
- `F_HL1_DRF_ABS`: 0.002
- `F_HL1_INJ`: 93.6
- `F_HL1_SKIN`: 1.6
- `F_HL1_KINJ`: 0.567
- `F_HL1_KMIN`: 0.50
- `F_HL1_WAX`: 576
- `F_HL1_RET`: 0.0
- `F_HL1_TMEAN`: 47.6
- `F_HL1_GEL`: 9
- `F_HL1_WCUT`: 91
- `F_HL1_Q0`: 103
- `F_HL1_QMIN`: 103
- `F_HL1_QEND`: 300
- `F_HL1_COLD`: 21.3
- `F_HL1_BELOW`: 23.4
- `F_HL1_WAXMAX`: 6.1
- `F_HL1_PMEAN`: 13.3
- `F_HL1_KPROD`: 0.50
- `F_HL1_ASPHPROD`: 36
- `F_HL1_ASPH`: 24
- `F_HL1_QDROP`: —
- `F_HL1_PB`: —
- `F_HL1_WAT0`: 40.0
- `F_NOWAX_RF`: 0.348
- `F_NOWAX_DRF`: +0.0188
- `F_NOWAX_DRF_ABS`: 0.019
- `F_NOWAX_INJ`: 119.5
- `F_NOWAX_SKIN`: 0.0
- `F_NOWAX_KINJ`: 1.000
- `F_NOWAX_KMIN`: 0.57
- `F_NOWAX_WAX`: 0
- `F_NOWAX_RET`: 0.0
- `F_NOWAX_TMEAN`: 48.4
- `F_NOWAX_GEL`: 0
- `F_NOWAX_WCUT`: 93
- `F_NOWAX_Q0`: 103
- `F_NOWAX_QMIN`: 103
- `F_NOWAX_QEND`: 425
- `F_NOWAX_COLD`: 18.3
- `F_NOWAX_BELOW`: 0.0
- `F_NOWAX_WAXMAX`: 0.0
- `F_NOWAX_PMEAN`: 14.2
- `F_NOWAX_KPROD`: 0.57
- `F_NOWAX_ASPHPROD`: 31
- `F_NOWAX_ASPH`: 19
- `F_NOWAX_QDROP`: —
- `F_NOWAX_PB`: —
- `F_NOWAX_WAT0`: -62.4
- `F_HL0_NOWAX_RF`: 0.345
- `F_HL0_NOWAX_DRF`: +0.0156
- `F_HL0_NOWAX_DRF_ABS`: 0.016
- `F_HL0_NOWAX_INJ`: 115.4
- `F_HL0_NOWAX_SKIN`: 0.0
- `F_HL0_NOWAX_KINJ`: 1.000
- `F_HL0_NOWAX_KMIN`: 0.56
- `F_HL0_NOWAX_WAX`: 0
- `F_HL0_NOWAX_RET`: 0.0
- `F_HL0_NOWAX_TMEAN`: 39.8
- `F_HL0_NOWAX_GEL`: 0
- `F_HL0_NOWAX_WCUT`: 93
- `F_HL0_NOWAX_Q0`: 103
- `F_HL0_NOWAX_QMIN`: 103
- `F_HL0_NOWAX_QEND`: 402
- `F_HL0_NOWAX_COLD`: 42.9
- `F_HL0_NOWAX_BELOW`: 0.0
- `F_HL0_NOWAX_WAXMAX`: 0.0
- `F_HL0_NOWAX_PMEAN`: 14.1
- `F_HL0_NOWAX_KPROD`: 0.56
- `F_HL0_NOWAX_ASPHPROD`: 31
- `F_HL0_NOWAX_ASPH`: 20
- `F_HL0_NOWAX_QDROP`: —
- `F_HL0_NOWAX_PB`: —
- `F_HL0_NOWAX_WAT0`: -62.4
- `F_NOGEL_RF`: 0.335
- `F_NOGEL_DRF`: +0.0056
- `F_NOGEL_DRF_ABS`: 0.006
- `F_NOGEL_INJ`: 102.1
- `F_NOGEL_SKIN`: 1.5
- `F_NOGEL_KINJ`: 0.573
- `F_NOGEL_KMIN`: 0.53
- `F_NOGEL_WAX`: 416
- `F_NOGEL_RET`: 0.0
- `F_NOGEL_TMEAN`: 49.5
- `F_NOGEL_GEL`: 0
- `F_NOGEL_WCUT`: 91
- `F_NOGEL_Q0`: 103
- `F_NOGEL_QMIN`: 103
- `F_NOGEL_QEND`: 338
- `F_NOGEL_COLD`: 15.0
- `F_NOGEL_BELOW`: 16.4
- `F_NOGEL_WAXMAX`: 5.9
- `F_NOGEL_PMEAN`: 13.5
- `F_NOGEL_KPROD`: 0.53
- `F_NOGEL_ASPHPROD`: 34
- `F_NOGEL_ASPH`: 22
- `F_NOGEL_QDROP`: —
- `F_NOGEL_PB`: —
- `F_NOGEL_WAT0`: 40.0
- `F_RET_RF`: 0.249
- `F_RET_DRF`: −0.0806
- `F_RET_DRF_ABS`: 0.081
- `F_RET_INJ`: 37.9
- `F_RET_SKIN`: 27.9
- `F_RET_KINJ`: 0.069
- `F_RET_KMIN`: 0.07
- `F_RET_WAX`: 124
- `F_RET_RET`: -0.4
- `F_RET_TMEAN`: 53.1
- `F_RET_GEL`: 2
- `F_RET_WCUT`: 75
- `F_RET_Q0`: 103
- `F_RET_QMIN`: 75
- `F_RET_QEND`: 95
- `F_RET_COLD`: 5.0
- `F_RET_BELOW`: 5.2
- `F_RET_WAXMAX`: 6.0
- `F_RET_PMEAN`: 9.2
- `F_RET_KPROD`: 1.10
- `F_RET_ASPHPROD`: 0
- `F_RET_ASPH`: 133
- `F_RET_QDROP`: 65
- `F_RET_PB`: 4.0
- `F_RET_WAT0`: 40.0
- `F_NOLATENT_RF`: 0.329
- `F_NOLATENT_DRF`: −0.0002
- `F_NOLATENT_DRF_ABS`: 0.000
- `F_NOLATENT_INJ`: 95.4
- `F_NOLATENT_SKIN`: 1.6
- `F_NOLATENT_KINJ`: 0.567
- `F_NOLATENT_KMIN`: 0.51
- `F_NOLATENT_WAX`: 393
- `F_NOLATENT_RET`: 0.0
- `F_NOLATENT_TMEAN`: 49.9
- `F_NOLATENT_GEL`: 7
- `F_NOLATENT_WCUT`: 91
- `F_NOLATENT_Q0`: 103
- `F_NOLATENT_QMIN`: 103
- `F_NOLATENT_QEND`: 308
- `F_NOLATENT_COLD`: 14.0
- `F_NOLATENT_BELOW`: 15.3
- `F_NOLATENT_WAXMAX`: 6.1
- `F_NOLATENT_PMEAN`: 13.3
- `F_NOLATENT_KPROD`: 0.51
- `F_NOLATENT_ASPHPROD`: 36
- `F_NOLATENT_ASPH`: 23
- `F_NOLATENT_QDROP`: —
- `F_NOLATENT_PB`: —
- `F_NOLATENT_WAT0`: 40.0
- `F_EQUIL_RF`: 0.318
- `F_EQUIL_DRF`: −0.0115
- `F_EQUIL_DRF_ABS`: 0.011
- `F_EQUIL_INJ`: 83.5
- `F_EQUIL_SKIN`: 3.3
- `F_EQUIL_KINJ`: 0.383
- `F_EQUIL_KMIN`: 0.38
- `F_EQUIL_WAX`: 330
- `F_EQUIL_RET`: 0.0
- `F_EQUIL_TMEAN`: 50.6
- `F_EQUIL_GEL`: 6
- `F_EQUIL_WCUT`: 89
- `F_EQUIL_Q0`: 103
- `F_EQUIL_QMIN`: 103
- `F_EQUIL_QEND`: 257
- `F_EQUIL_COLD`: 11.9
- `F_EQUIL_BELOW`: 12.9
- `F_EQUIL_WAXMAX`: 6.1
- `F_EQUIL_PMEAN`: 12.9
- `F_EQUIL_KPROD`: 0.48
- `F_EQUIL_ASPHPROD`: 38
- `F_EQUIL_ASPH`: 26
- `F_EQUIL_QDROP`: —
- `F_EQUIL_PB`: —
- `F_EQUIL_WAT0`: 40.0
- `F_NOPRESS_RF`: 0.344
- `F_NOPRESS_DRF`: +0.0146
- `F_NOPRESS_DRF_ABS`: 0.015
- `F_NOPRESS_INJ`: 111.4
- `F_NOPRESS_SKIN`: 2.2
- `F_NOPRESS_KINJ`: 0.490
- `F_NOPRESS_KMIN`: 0.49
- `F_NOPRESS_WAX`: 817
- `F_NOPRESS_RET`: 0.0
- `F_NOPRESS_TMEAN`: 49.0
- `F_NOPRESS_GEL`: 13
- `F_NOPRESS_WCUT`: 92
- `F_NOPRESS_Q0`: 103
- `F_NOPRESS_QMIN`: 103
- `F_NOPRESS_QEND`: 374
- `F_NOPRESS_COLD`: 16.5
- `F_NOPRESS_BELOW`: 22.9
- `F_NOPRESS_WAXMAX`: 8.2
- `F_NOPRESS_PMEAN`: 12.0
- `F_NOPRESS_KPROD`: 1.00
- `F_NOPRESS_ASPHPROD`: 0
- `F_NOPRESS_ASPH`: 0
- `F_NOPRESS_QDROP`: —
- `F_NOPRESS_PB`: —
- `F_NOPRESS_WAT0`: 45.3
- `F_LTNE_RF`: 0.329
- `F_LTNE_DRF`: +0.0000
- `F_LTNE_DRF_ABS`: 0.000
- `F_LTNE_INJ`: 95.6
- `F_LTNE_SKIN`: 1.6
- `F_LTNE_KINJ`: 0.568
- `F_LTNE_KMIN`: 0.51
- `F_LTNE_WAX`: 390
- `F_LTNE_RET`: 0.0
- `F_LTNE_TMEAN`: 49.9
- `F_LTNE_GEL`: 7
- `F_LTNE_WCUT`: 91
- `F_LTNE_Q0`: 103
- `F_LTNE_QMIN`: 103
- `F_LTNE_QEND`: 309
- `F_LTNE_COLD`: 14.0
- `F_LTNE_BELOW`: 15.2
- `F_LTNE_WAXMAX`: 6.0
- `F_LTNE_PMEAN`: 13.3
- `F_LTNE_KPROD`: 0.51
- `F_LTNE_ASPHPROD`: 36
- `F_LTNE_ASPH`: 23
- `F_LTNE_QDROP`: —
- `F_LTNE_PB`: —
- `F_LTNE_WAT0`: 40.0
- `F_T0`: 55
- `F_INJ_LOSS`: 33
- `F_COOL_LOSS`: 0.034
- `F_WAX_EFF`: 0.019
- `F_WAX_EFF_PCT`: 5.4
- `F_WAX_INJ_PCT`: 20
- `F_WAX_SHARE`: 56
- `F_GEL_SHARE`: 30
**T3**

| Показатель | Эффект теплообмена при наличии парафина | Эффект теплообмена без парафина | Эффект парафина при учете теплообмена | Эффект парафина без учета теплообмена | Взаимодействие |
|---|---|---|---|---|---|
| КИН | +0.006 | +0.003 | −0.019 | −0.022 | +0.003 |
| Закачано, тыс. м³ | +6.8 | +4.2 | −24.0 | −26.7 | +2.7 |

- `F_FX_HL_WAX`: 0.006
- `F_FX_HL_NOWAX`: 0.003
- `F_FX_WAX_HL`: 0.019
- `F_FX_WAX_NOHL`: 0.022
- `F_FX_INTER`: 0.003
- `F_FX_INTER_SIGNED`: +0.003
- `F_FX_INTER_PCT`: 49
- `F_HL0_WAXRATIO`: 3.9
- `F_HL1_WAXPCT`: 48
- `F_HL1_SHARE`: 71
- `F_LTNE_REL`: 1·10⁻¹³
- `F_EQUIL_REL`: 3·10⁻¹
- `F_NOLATENT_REL`: 8·10⁻³
- `F_WAT0`: 38.7–40.0
- `F_PB_YEARS`: —
- `F_QDROP_DAYS`: —
- `F_FRONT_W`: 198
- `F_FRONT_T`: 34
- `F_FRONT_RATIO`: 0.17
- `F_WIDTH`: 28
- `F_GROUPS`: 1 / 24 / 30 / 45
- `F_LTNE_DT`: 0
- `F_PE`: 118
- `F_VT_RATIO`: 0.42
- `F_STE`: 22
- `F_WPREC`: 11.4
- `F_WAX0`: 25.0
- `F_KRATIO`: 4·10²
- `F_SIG0`: 0.96
- `F_D0`: 0.04
- `F_TAU_LTNE`: 0.01
- `F_DG`: 0.2
- `F_DA`: 4·10⁵
- `F_TPASS`: 304
- `F_WALL_SHARE`: 39
- `TH_WON1_WAT`: 43.7
- `TH_WON1_35`: 15.8
- `TH_WON1_25`: 22.0
- `TH_EXP_35`: 5.0
- `TH_EXP_25`: 13.0
- `TH_WAT_EXP`: 45.65
- `TH_RMS_GROUPS`: 0.39
- `TH_RMS_LEGACY`: 0.62
- `TH_WAT_GROUPS`: 45.2
- `TH_WAT_LEGACY`: 38.8
- `TH_SLOPE`: 0.07
- `TH_DH`: 41.9
- `TH_SHIFT`: 52.4
- `EOS_LI_WAT_PR`: 67.6
- `EOS_LI_RMS_PR`: 4.6
- `EOS_LI_WAT_IDEAL`: 64.1
- `EOS_LI_RMS_IDEAL`: 4.2
- `EOS_LI_WAT_SS`: 64.9
- `EOS_LI_RMS_SS`: 7.3
- `EOS_LI_DWAT`: 22
- `EOS_LI_PR_IDEAL`: 3.5
- `EOS_SYN_DPR`: −9.6
- `EOS_SYN_DIDEAL`: −11.3
- `EOS_SYN_DSS_RNG`: −1.4…+3.0
- `EOS_SYN_DAUTH_MS`: −3.1
- `EOS_SYN_DAUTH_SS`: +5.7
- `EOS_SYN_DAUTH_UQ`: +1.5
- `EOS_SYN_DPR_ABS`: 9.6
- `EOS_SYN_DIDEAL_ABS`: 11.3
- `EOS_SYN_DAUTH_MS_ABS`: 3.1
- `EOS_SYN_SHIFT`: 1.7
- `EOS_SYN_RMS_PR`: 6.7–8.8
- `EOS_SYN_RMS_IDEAL`: 7.1–9.3
- `EOS_SYN_RMS_SS`: 5.6–8.8
- `EOS_SYN_AAD_UQ`: 0.84–1.45
- `EOS_SYN_DMSC_RNG`: −4.6…−2.5
- `EOS_SYN_RMS_MSC`: 5.4–6.7
- `EOS_SYN_DMSCP_RNG`: −7.6…−4.8
- `EOS_SYN_RMS_MSCP`: 7.2–8.8
- `EOS_SYN_DUQ_RNG`: −0.4…+0.8
- `EOS_SYN_RMS_UQ`: 0.5–3.9
- `EOS_SYN_DUQID_RNG`: −1.7…−0.7
- `EOS_SYN_RMS_UQID`: 0.9–4.0
- `EOS_SYN_DPR_RNG`: −10.4…−7.7
- `EOS_SYN_DEFF_RNG`: −24.2…−13.4
- `EOS_SYN_RMS_EFF`: 3.1–9.3
- `EOS_SYN_DUQ_ABS`: 0.8
- `EOS_SYN_DAUTH_UQ_RNG`: +1.0…+2.2
- `EOS_SYN_AAD_UQ_OWN`: 0.40–2.35
- `EOS_LI_WAT_MSC`: 71.2
- `EOS_LI_RMS_MSC`: 5.6
- `EOS_LI_WAT_UQ`: 71.3
- `EOS_LI_RMS_UQ`: 4.1
- `EOS_LI_UQFIT_RMS`: 3.2
- `EOS_LI_UQFIT_WAT`: 62
- `EOS_LI_UQFIT_S`: 0.080
- `EOS_LI_UQFIT_LAST`: 48
- `EOS_LI_WAT_HI`: 26
- `BR_C24_WON`: 65
- `BR_C24_F`: 54.9
- `BR_C24_T`: 31.3
- `BR_C24_TOT`: 86
- `BR_C24_TR_PCT`: 36
- `PRESS_SANDYGA`: 0.15–0.23
- `PRESS_SL_EFF`: 0.187
- `PRESS_SL_PR`: 0.28
- `PRESS_SL_PR0`: 0.017
- `PRESS_DROP_EFF`: 6.6
- `PRESS_RISE_EFF`: 0.18
- `PRESS_DROP_PR`: 1.3
- `PRESS_RISE_PR`: 0.28
- `PRESS_KIJ`: −0.003
- `PRESS_DV_EFF`: 0.028
- `PRESS_DV_EOS`: 0.13
- `ASPH_H_MAX`: 11
- `ASPH_H_P`: 9.5
- `ASPH_H_WIN`: 8.0–10.5
- `ASPH_N_MAX`: 3
- `ASPH_N_P`: 9.5
- `ASPH_N_WIN`: 9.0–10.5
- `ASPH_N0_MAX`: 50
- `ASPH_N0_P`: 1.0
- `ASPH_N0_WIN`: 1.0–10.5
- `ASPH_LAB_DEAD`: 0.0
- `ASPH_LAB_PLATEAU`: 0.33–0.77
- `ASPH_DELTA_A`: 21.54
- `TZ_ASPH_WT`: 16.1
- `TZ_T`: 100
- `TZ_PB`: 14.1
- `TZ_PON`: 34.5
- `TZ_PB_PR0`: 18.4
- `TZ_EXP_RNG`: 0.40–1.04
- `TZ_ONSET_RNG`: 8–13
- `TZ_KIJ_AUTH`: 0.4
- `TZ_VS_AUTH`: 0.8
- `TZ_ONSET_RMS`: 10.1
- `TZ_VS04_RMS`: 0.55
- `TZ_VS04_LOW`: 0.00
- `TZ_KIJ`: 0.035
- `TZ_VS`: 0.654
- `TZ_VBAR`: 0.646
- `TZ_VS_DEV_PCT`: 1.1
- `TZ_RMS`: 0.04
- `PVT_RS_DEV`: 14
- `PVT_RS_DEV_P`: 0.5
- `PVT_FREE6`: 0.22
- `PVT_BO_B`: 1.086
- `VISC_RMS`: 0.039
- `VISC_RMS_KD`: 0.72
- `VISC_KD_UNDER`: 6.6
- `VISC_PR_OVER`: 6·10⁵
- `VISC_T0`: 24.4
- `VISC_MU25`: 46.6
- `VISC_E`: 21.3
- `VISC_D`: 3.76
- `GEL_PHI`: 6.9
- `GEL_TAU`: 17.4
- `GEL_N`: 1.88
- `GEL_SR_LEG`: 0.048
- `GEL_SR_1`: 0.143
- `GEL_SR_01`: 0.058
- `GEL_SR_003`: 0.049
- `GEL_SR2_LEG`: 0.246
- `GEL_SR2_003`: 0.256
- `GEL_LI_COMP`: 0.081
- `GEL_LI_1`: 0.077
- `GEL_SD_STATIC`: 12.1
- `GEL_SD_LEG`: 4.1
- `SR_JOINT`: 0.11
- `SR_CROSS`: 0.16–0.18
- `SR_MULT1`: 3.1
- `SR_MULT2`: 0.30
- `SR_MU1`: 1
- `SR_MU2`: 10
- `SD_LOSS_NARROW`: 0.79–0.92
- `SD_LOSS_WIDE`: 0.35–0.63
- `SD_LOSS_EXP`: 0.76–0.88
- `F_RPASS`: 16.9
- `F_VOL_BLOCK`: 53
- `F_COND_BLOCK`: 25
- `SR_RPASS`: 19.5
- `SR_VOL_BLOCK`: 67
- `SR_COND_BLOCK`: 38
- `F_MU_RATIO_25`: 3.0
- `F_MU_RATIO_FIT`: 2.5
- `OLD_FX_WAX_HL`: 0.0004
- `OLD_FX_WAX_NOHL`: 0.017
- `OLD_FX_INTER_PCT`: 81
**T4**

| Измерение | Величина | Опыт | Настоящая модель | Уравнение состояния | Другие модели |
|---|---|---|---|---|---|
| Кривая выпадения, нефть Жетыбая [19] | СКО, % масс. | — | 0.39 | 4.6 | один псевдокомпонент: 0.62; PR + UNIQUAC: 4.1, с подобранным распределением н-алканов: 3.2 |
| То же | WAT, °C | 45.65 | 45.2 | 67.6 | один псевдокомпонент: 38.8; по Вону без подбора: 43.7; PR + UNIQUAC: 71.3 |
| Синтетические смеси C18–C36 [30] | WDT − WDT опыта, °C | — | −11.3 | −9.6 | PR + переход по Coutinho: −4.6…−2.5; PR + UNIQUAC: −0.4…+0.8; PR + multi-solid [30]: −3.1; PR + UNIQUAC [30]: +1.5 |
| То же | отклонение доли твердого, % масс. | — | 7.1–9.3 | 6.7–8.8 | PR + переход по Coutinho: 5.4–6.7; PR + UNIQUAC: 0.5–3.9 (среднее абсолютное 0.40–2.35); PR + UNIQUAC [30]: 0.84–1.45 |
| Сдвиг WAT с давлением [46] | dWAT/dP, °C/МПа | 0.15–0.23 | 0.187 | 0.28 | PR без скачка объема: 0.017 |
| Вязкость при охлаждении [19] | СКО ln μ | — | 0.039 | — | Кригер—Догерти: 0.72 |
| Асфальтены у давления насыщения | выпало, % содержания | нет замеров | 11 | 3 | Nghiem, kij = 0: 50 при 1.0 МПа |
| Асфальтены, живая нефть [34] | СКО выпавших, % масс. | — | — | 0.04 | Нгхайем по точке начала осаждения: 10.1; V_s по кривой при k_ij = 0.4: 0.55 |

**T5**

| Данные | Что варьируется | Метод | Критерий | Результат |
|---|---|---|---|---|
| Кривая выпадения и WAT нефти Жетыбая [19] | наклон распределения s, эффективная теплота ΔH_eff, сдвиг T_m; перебор разбиений на группы | метод наименьших квадратов | доля выпавшего при 10–45°C и WAT | s = 0.07, ΔH_eff = 41.9 кДж/моль, сдвиг 52.4 K |
| Сдвиг WAT с давлением [46] | доля скачка объема Δv/v_L в (11) | по среднему наклону регрессий для 10–60% парафина | dWAT/dP | Δv/v_L = 0.028 |
| Вязкость при 24–100°C [19] | μ_ref, E_a (выше 40°C); D, φ_gel, τ_ref, n_g (ниже) | наименьшие квадраты по ln μ, μ = μ_p + τ_y/γ̇ | СКО ln μ | μ(25°C) = 46.6 мПа·с, E_a = 21.3 кДж/моль, D = 3.76, φ_gel = 6.9%, τ_ref = 17.4 Па, n_g = 1.88 |
| Давление начала осаждения асфальтенов | δ_a (Хиршберг) или f_s* (Нгхайем) | насыщение в одной точке | — | δ_a = 21.54 МПа⁰·⁵ |
| Кривая выпавших асфальтенов [34] | k_ij асфальтены—легкие 0–0.5, V_s (Нгхайем); v_a (Хиршберг) | перебор k_ij с шагом 0.005, V_s — метод наименьших квадратов; k_ij метан—C7+ по давлению насыщения | СКО выпавших, % масс. | k_ij = 0.035, V_s = 0.654 л/моль |
| Sutton и Roberts [27] | d_p = 2–50 мкм, f_D = 10⁻³–10³, порог выноса 0.1–5 Па и его скорость 10⁻⁶–10⁻¹ с⁻¹ | метод наименьших квадратов по расчетам керна: общий набор, набор на опыт, перекрестный прогноз | СКО k/k₀ | общий набор: d_p = 15.6 мкм, f_D = 0.97, порог 1.5 Па, скорость 3.0·10⁻⁴ с⁻¹ |
| Li и др., 90–45°C [19] | Γ_max = 10⁻⁷–10⁻² кг/м², K_ref = 10⁻²–10⁶, −ΔH_ads = 0–150 кДж/моль, k_ads = 10⁻⁶–10⁻¹ с⁻¹ | точно по трем плато, затем метод наименьших квадратов; две формы кинетики | СКО k/k₀ ступеней | Γ_max = 1.2·10⁻⁴ кг/м², K_ref = 225, ΔH_ads = −137 кДж/моль, k_ads = 9.2·10⁻⁴ с⁻¹ |
| Li и др., 25°C [19] | d_p = 1–50 мкм, f_D = 10⁻³–10², k_cr = 10⁻⁵–10⁻¹ с⁻¹; тиксотропное время геля 5, 60 мин | метод наименьших квадратов | СКО k/k₀ | d_p = 13.5 мкм, f_D = 0.10, 1/k_cr = 63 с, время геля 60 мин |
| Sandyga и др. [2] | WAT пористой среды 33.8, 34.3, 34.8°C; k_w = 10⁻⁶–1 с⁻¹, C_0 = 10⁻²–1, k_cr = 10⁻⁵–1 с⁻¹ | сетка, затем метод наименьших квадратов | СКО lg(∇p/∇p₀) | WAT 34.8°C, k_w = 1.1·10⁻³ с⁻¹, C_0 = 1.0, 1/k_cr = 45 мин |
| He и др. [28] | отношение горла к поре (сеть), показатель n (степенная зависимость) | одномерная минимизация | СКО k/k₀ | горло/пора 0.40 при z = 8; n = 5.7 |
| Вариант на уравнении состояния | k_ij газ—нефть | по давлению насыщения при T_0 | — | k_ij = −0.003; T_m и ΔH — по Вону или Coutinho с переходом, без подбора |

