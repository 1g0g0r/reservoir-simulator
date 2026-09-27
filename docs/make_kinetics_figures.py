"""Рисунки к описанию кинетики осаждения (`docs/кинетика_осаждения.md`): аналитические проверки моделей.

    python docs/make_kinetics_figures.py   # секунды, прогонов не требует -> docs/figures/kin*.png

Каждый рисунок строится функциями пакета (`paraphin/equations/Kinetics_math.py`, `Thermal_ltne.py`), а не их
копиями, поэтому он проверяет код, а не формулу из текста:
  kin01 - перенос частиц к стенке капилляра: вклады броуновской и сдвиговой диффузии и оседания от размера;
  kin02 - агрегация по Смолуховскому: численные шаги против точного решения, размер фрактальных флокул;
  kin03 - изотерма удержания смол и асфальтенов от температуры: подобранные по Li et al. (2024) параметры;
  kin04 - замыкания проницаемости модели глубинной фильтрации против пучка капилляров;
  kin05 - тепловое неравновесие: численная схема (перенос + точный обмен) против решения Шумана;
  kin06 - сеть пор и горл (эффективная среда) против пучка: k(m) и порог протекания.
Стиль - общий с рисунками описания модели (`make_model_figures.py`).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from make_model_figures import SERIES, STYLES, INK, INK2, _save, plt  # noqa: E402
from paraphin.constants import k_B, Lk, ro_p, ro_o, ro_asph, R  # noqa: E402
from paraphin.equations.Kinetics_math import (brownian_diffusivity, shear_diffusivity, stokes_velocity,  # noqa: E402
                                              leveque_velocity, coagulation_kernel, smoluchowski_step, floc_size,
                                              langmuir_constant, langmuir_eq, perm_kozeny_carman, perm_power,
                                              perm_damage)
from paraphin.equations.Thermal_ltne import exchange_step  # noqa: E402

EXPERIMENTS = ROOT / 'experiments'


def fig_transport():
    """Скорость переноса частиц к стенке канала r = 10 мкм от диаметра частицы при градиенте 1 МПа/м."""
    T_K, mu, r, grad = 293.15, 5e-3, 10e-6, 1e6
    um = grad * r * r / (8.0 * mu)  # средняя скорость в капилляре по Пуазейлю
    gamma_w = 4.0 * um / r
    d = np.logspace(-8, -4.3, 200)
    brown = np.array([leveque_velocity(um, r, brownian_diffusivity(T_K, mu, x)) for x in d])
    shear = np.array([leveque_velocity(um, r, shear_diffusivity(0.5, 0.01, gamma_w, x / 2)) for x in d])
    settle = np.array([stokes_velocity(x / 2, ro_p - ro_o, mu) for x in d])
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for y, label, color, ls in ((brown, 'броуновская диффузия (Левек)', SERIES[0], STYLES[0]),
                                (shear, 'сдвиговая дисперсия, φ = 1 %', SERIES[1], STYLES[1]),
                                (settle, 'оседание по Стоксу', SERIES[2], STYLES[2])):
        ax.loglog(d * 1e6, y, color=color, ls=ls, label=label)
    ax.axvline(15.0, color=INK2, lw=0.8, ls=':')
    ax.text(15.0, ax.get_ylim()[0] * 3, ' d_p = 15 мкм\n (калибровка)', fontsize=7, color=INK2)
    ax.set_xlabel('диаметр частицы, мкм')
    ax.set_ylabel('скорость к стенке, м/с')
    ax.set_title(f'канал r = 10 мкм, |∇p| = 1 МПа/м, μ = 5 мПа·с, L_k = {Lk * 1e3:.1f} мм', fontsize=8, color=INK2)
    ax.legend(fontsize=7, handlelength=3.2)
    _save(fig, 'kin01_transport')


def fig_smoluchowski():
    """Число флокул N/N0 и диаметр флокулы: шаги `smoluchowski_step` против N0/(1 + K*N0*t/2)."""
    T_K, mu, d0, df = 343.15, 5e-3, 1e-7, 2.0
    phi = 1e-3  # объемная доля выпавших асфальтенов
    n0 = phi / (math.pi * d0 ** 3 / 6.0)
    t = np.logspace(-2, 4, 60)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for n_w, (w, label) in enumerate(((1.0, 'DLCA, W = 1'), (1e3, 'RLCA, W = 10³'))):
        k = coagulation_kernel(T_K, mu, w)
        exact = 1.0 / (1.0 + k * n0 * t / 2.0)
        num, n, t_prev = [], n0, 0.0
        for x in t:
            n = smoluchowski_step(n, k, x - t_prev)
            t_prev = x
            num.append(n / n0)
        ax1.loglog(t, exact, color=SERIES[n_w], ls=STYLES[0], label=f'{label}: точное')
        ax1.loglog(t[::4], np.array(num)[::4], 'o', mfc='white', mec=SERIES[n_w], ms=4, label=f'{label}: шаги')
        m_floc = ro_asph * phi / (np.array(num) * n0)
        for n_d, dfx in enumerate((1.8, 2.5)):
            ax2.loglog(t, [floc_size(m, d0, dfx, ro_asph) * 1e6 for m in m_floc], color=SERIES[n_w],
                       ls=STYLES[n_d + 1], label=f'{label}, D_f = {dfx}')
    ax1.set_xlabel('t, с')
    ax1.set_ylabel('N / N₀')
    ax1.legend(fontsize=6.5, handlelength=3.2)
    ax2.set_xlabel('t, с')
    ax2.set_ylabel('диаметр флокулы, мкм')
    ax2.legend(fontsize=6.5, handlelength=3.2)
    _save(fig, 'kin02_smoluchowski')


def _li_params():
    path = EXPERIMENTS / 'results' / 'li2024.json'
    if not path.is_file():
        return None
    out = json.loads(path.read_text(encoding='utf-8'))
    return out.get('dynamic', {}).get('kin'), out.get('cumulative')


def fig_langmuir():
    """Равновесное удержание (доля sigma_max) от температуры: Ленгмюр с K(T) по Вант-Гоффу, параметры Li et al."""
    got = _li_params()
    if got is None or got[0] is None:
        print('   kin03 пропущен: нет подбора в experiments/results/li2024.json (python experiments/li2024.py)')
        return
    kin, cum = got
    from paraphin import fi_0, w1_cv, w2_cv
    data = json.loads((EXPERIMENTS / 'data' / 'li2024.json').read_text(encoding='utf-8'))
    m0 = data['core']['porosity']
    a0 = 2.0 * m0 * float((w1_cv * fi_0).sum()) / float((w2_cv * fi_0).sum())
    c_a, c_r = data['oil']['asphaltenes'], data['oil']['resins']
    t = np.linspace(20, 95, 151)
    sig = []
    for tc in t:
        k_l = langmuir_constant(kin['ADS_K'], kin['ADS_DH'], tc + 273.15, 70.0 + 273.15, R)
        g = langmuir_eq(kin['ADS_GMAX'] * a0, k_l, c_a) + langmuir_eq(0.5 * kin['ADS_GMAX'] * a0, k_l, c_r)
        sig.append(g / 1200.0 / (kin['PERM_SMAX'] * m0))
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.plot(t, 1.0 - np.minimum(sig, 1.0), color=SERIES[0], label='модель: 1 − σ/σ_max при равновесии')
    temps = sorted((float(x) for x in cum), reverse=True)
    ax.plot([x for x in temps if x > 30], [cum[str(x)] if str(x) in cum else cum[x] for x in temps if x > 30], 'o',
            mfc='white', mec=INK, ms=5, label='опыт: накопленное k/k₀ ступеней')
    ax.set_xlabel('T, °C')
    ax.set_ylabel('k / k₀ (удержание смол и асфальтенов)')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, handlelength=3.2)
    _save(fig, 'kin03_langmuir')


def fig_permeability():
    """k/k0 от m/m0: замыкания модели глубинной фильтрации и пучок капилляров при сужении."""
    m0 = 0.2
    x = np.linspace(0.3, 1.0, 141)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.semilogy(x, [perm_kozeny_carman(v * m0, m0) for v in x], color=SERIES[0], ls=STYLES[0], label='Козени–Карман')
    for n_c, n in enumerate((6.0, 10.0, 19.0)):
        ax.semilogy(x, [perm_power(v * m0, m0, n) for v in x], color=SERIES[1 + n_c], ls=STYLES[1],
                    label=f'степенной, n = {n:g}')
    ax.semilogy(x, [perm_damage((1.0 - v) / 0.3, 1.0, 2.0) for v in x], color=SERIES[4], ls=STYLES[2],
                label='функция повреждения, σ_max = 0.3 m₀, γ = 2')
    r = np.linspace(1e-3, 5.0, 20000)
    fi = np.exp(-0.5 * (np.log(r) / 0.4) ** 2) / r
    pts = []
    for h in np.linspace(0.0, 2.0, 200):
        rr = np.maximum(r - h, 0.0)
        pts.append((np.sum(rr ** 2 * fi) / np.sum(r ** 2 * fi), np.sum(rr ** 4 * fi) / np.sum(r ** 4 * fi)))
    pts = np.array(pts)
    ax.semilogy(pts[:, 0], pts[:, 1], color=INK, ls=STYLES[3], label='пучок капилляров: сужение')
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(1e-4, 1.2)
    ax.set_xlabel('m / m₀')
    ax.set_ylabel('k / k₀')
    ax.legend(fontsize=6.5, handlelength=3.2)
    _save(fig, 'kin04_permeability')


def fig_schumann():
    """Выходная температура флюида: перенос с Курантом 1 + точный обмен за шаг против решения Шумана."""
    from scipy.special import i0
    from scipy.integrate import quad
    L, u, c_f, c_s = 1.0, 1e-3, 1.0e6, 2.0e6
    n = 400
    dt = L / n / u
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    for n_h, h in enumerate((1000.0, 5000.0, 20000.0)):
        y = h * L / (c_f * u)
        tf, ts = np.zeros(n), np.zeros(n)
        times, out = [], []
        for step in range(1, int(6000.0 / dt) + 1):
            tf[1:] = tf[:-1].copy()
            tf[0] = 1.0
            for k in range(n):
                tf[k], ts[k] = exchange_step(tf[k], ts[k], c_f, c_s, h, dt)
            if step % 20 == 0:
                times.append(step * dt)
                out.append(tf[-1])
        exact = []
        for t in times:
            if t <= L / u:
                exact.append(0.0)
                continue
            z = h * (t - L / u) / c_s
            val, _ = quad(lambda s: math.exp(-s) * i0(2.0 * math.sqrt(s * z)), 0.0, y, limit=200)
            exact.append(1.0 - math.exp(-z) * val)
        tt = np.array(times) * u / L
        ax.plot(tt, exact, color=SERIES[n_h], ls=STYLES[0], label=f'Шуман, y = {y:g}')
        ax.plot(tt[::6], np.array(out)[::6], 'o', mfc='white', mec=SERIES[n_h], ms=3.5, label=f'схема, y = {y:g}')
    ax.set_xlabel('t·u / L (прокачано объемов пор)')
    ax.set_ylabel('(T_f − T₀) / (T_вх − T₀) на выходе')
    ax.legend(fontsize=6.5, ncol=2, handlelength=3.2)
    _save(fig, 'kin05_schumann')


def fig_network():
    """Сеть пор и горл (эффективная среда) против пучка: k(m) при равномерном сужении и порог протекания."""
    from paraphin.equations.Kinetics_math import ema_conductance
    r = np.linspace(1e-3, 5.0, 3000)
    fi = np.exp(-0.5 * (np.log(r) / 0.4) ** 2) / r
    w0 = fi / fi.sum()
    gam = 0.4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.3))
    bundle = []
    for h in np.linspace(0.0, 2.0, 200):
        rr = np.maximum(r - h, 0.0)
        bundle.append((np.sum(rr ** 2 * fi) / np.sum(r ** 2 * fi), np.sum(rr ** 4 * fi) / np.sum(r ** 4 * fi)))
    bundle = np.array(bundle)
    ax1.semilogy(bundle[:, 0], bundle[:, 1], color=INK, ls=STYLES[3], label='пучок капилляров')
    for n_z, z in enumerate((3.0, 4.0, 6.0, 1e6)):
        g0 = ema_conductance((gam * r) ** 4, w0, 0.0, z)
        pts = []
        for h in np.linspace(0.0, 2.0, 300):
            rt = gam * r - h
            op = rt > 0
            if not op.any():
                break
            gm = ema_conductance(np.where(op, rt, 0.0) ** 4, np.where(op, w0, 0.0), float(w0[~op].sum()), z)
            pts.append((np.sum(np.maximum(r - h, 0.0) ** 2 * fi) / np.sum(r ** 2 * fi), max(gm / g0, 1e-6)))
        pts = np.array(pts)
        label = 'сеть, z → ∞' if z > 1e5 else f'сеть, z = {z:g}'
        ax1.semilogy(pts[:, 0], pts[:, 1], color=SERIES[n_z], ls=STYLES[n_z % 3], label=label)
    ax1.set_xlim(0.3, 1.0)
    ax1.set_ylim(1e-3, 1.2)
    ax1.set_xlabel('m / m₀')
    ax1.set_ylabel('k / k₀')
    ax1.legend(fontsize=6.5, handlelength=3.2)
    p = np.linspace(0.0, 1.0, 201)
    for n_z, z in enumerate((3.0, 4.0, 6.0)):
        ax2.plot(p, [ema_conductance(np.array([1.0]), np.array([x]), 1.0 - x, z) for x in p], color=SERIES[n_z],
                 ls=STYLES[n_z % 3], label=f'z = {z:g}, порог {2 / z:.2f}')
    ax2.plot(p, p, color=INK, ls=STYLES[3], label='пучок')
    ax2.set_xlabel('доля открытых горл')
    ax2.set_ylabel('g_m / g')
    ax2.legend(fontsize=6.5, handlelength=3.2)
    _save(fig, 'kin06_network')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    fig_transport()
    fig_smoluchowski()
    fig_langmuir()
    fig_permeability()
    fig_schumann()
    fig_network()


if __name__ == '__main__':
    main()
