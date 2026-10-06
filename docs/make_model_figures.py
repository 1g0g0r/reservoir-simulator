"""Рисунки к математическому описанию модели детального состава нефти (`docs/модель_АСПО.md`).

    python experiments/calibrate.py     # сначала: калибровка -> experiments/results/calibration.json
    python demo_composition.py          # затем: демонстрационные расчеты -> outputs/data/demo_*
    python docs/make_model_figures.py   # рисунки -> docs/figures/*.png

Цвета - фиксированный порядок категориальной палитры (проверена на различимость при нарушениях
цветового зрения); группы парафина дополнительно различаются типом линии и подписаны прямо на графике.
"""
import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from paraphin.geometry import fi_0  # noqa: E402
from paraphin import oil_composition as oc  # noqa: E402
from paraphin.layout import N_W  # noqa: E402
from paraphin.constants import (init_T, P_onset_asph, P_bubble, sara_asphaltenes, sara_resins, v_asph, R,  # noqa: E402
                                ro_asph, ro_o, gel_phi, gel_phi_ref, gel_tau_ref, gel_n, X_max, Y_max,
                                geological_reserves, gel_mobility_min, alpha_p_visc)
from paraphin.equations.Gel import br_factor, gel_phi_eq  # noqa: E402

FIG = HERE / 'figures'
DATA = ROOT / 'outputs' / 'data'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
STYLES = ['-', '--', '-.', ':']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
GROUP_NAMES = ['C17–22 мягкие', 'C23–30 твердые', 'C31–40', 'C41–60 воски']
BLUES = LinearSegmentedColormap.from_list('blues', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])

plt.rcParams.update({
    'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False, 'lines.linewidth': 2.0, 'savefig.dpi': 200,
    'figure.dpi': 100, 'font.family': 'DejaVu Sans',
})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ', name)


def _label_end(ax, x, y, text, color, dx=0.0, dy=0.0):
    """Прямая подпись у конца линии - цветом только маркер, текст - чернилами."""
    ax.annotate(text, (x, y), xytext=(x + dx, y + dy), color=INK, fontsize=8, va='center',
                arrowprops=dict(arrowstyle='-', color=color, lw=0.8) if (dx or dy) else None)


# --- 1. Схема связей ---------------------------------------------------------------------------------------

def fig_scheme():
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.2, 7.0)
    w, h = 2.7, 1.45
    boxes = {
        'comp': (0.1, 5.1, 'Состав нефти\nSARA + SCN: 4 группы\nпарафинов, асфальтены,\nсмолы'),
        'thermo': (4.15, 5.1, 'Равновесие\nmulti-solid (T, P, газ)\nФлори–Хаггинс (δ, P_b)'),
        'kin': (8.2, 5.1, 'Кинетика\nфлокуляция, осаждение\nв пучке капилляров\nfi(r)'),
        'pore': (8.2, 2.55, 'Пористость и\nпроницаемость\nm, k из ∫r²fi, ∫r⁴fi'),
        'rheo': (4.15, 2.55, 'Реология\nμ(T, φ), предел τ_y,\nБукингем–Райнер Φ(∇p)'),
        'flow': (0.1, 2.55, 'Фильтрация и тепло\nДарси, IMPES, перенос\nкомпонентов, скрытая\nтеплота'),
        'out': (4.15, 0.0, 'Выход: WAT(P), осадок\nпо компонентам, КИН'),
    }
    for key, (x, y, text) in boxes.items():
        out = key == 'out'
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.12',
                                    fc='#fdf3ee' if out else '#f5f8fd', ec=SERIES[1] if out else SERIES[0], lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=8, color=INK)

    def arrow(p, q, text=None, dx=0.0, dy=0.22, ha='center'):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=11, color=INK2, lw=1.1))
        if text:
            ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, text, ha=ha, fontsize=7.2, color=INK2)

    top, mid = 5.1 + h / 2, 2.55 + h / 2
    arrow((0.1 + w, top), (4.15, top), 'w_k, x_k')
    arrow((4.15 + w, top), (8.2, top), 'взвесь,\nфлокулы', dy=-0.62)
    arrow((8.2 + w / 2, 5.1), (8.2 + w / 2, 2.55 + h), 'q_p1, q_p2, q_pa', dx=0.12, dy=0.0, ha='left')
    arrow((8.2, mid), (4.15 + w, mid), 'fi, m')
    arrow((4.15, mid), (0.1 + w, mid), 'μ_эфф, Φ')
    arrow((0.1 + w / 2, 2.55 + h), (0.1 + w / 2, 5.1), 'T, P, перенос', dx=0.12, dy=0.0, ha='left')
    arrow((4.15 + w / 2, 2.55), (4.15 + w / 2, h))
    _save(fig, 'fig01_scheme')


# --- 2-4. Состав, выпадение, давление ---------------------------------------------------------------------------

def fig_scn(cal):
    a = cal['scn']
    n, w = np.array(a['scn_n']), np.array(a['scn_w']) * 100
    edges = [oc.scn_first] + [b + 1 for b in oc.scn_bounds] + [oc.scn_last + 1]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        s = (n >= lo) & (n < hi)
        ax.bar(n[s], w[s], width=0.8, color=SERIES[k], edgecolor='white', linewidth=0.6,
               label=f'{GROUP_NAMES[k]}: {100 * a["group_w"][k]:.1f} % масс.')
    ax.set_xlabel('число атомов углерода n')
    ax.set_ylabel('доля C_n в нефти, % масс.')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='x', visible=False)
    _save(fig, 'fig02_scn')


def fig_precip(cal):
    a = cal['scn']
    g = oc.group_properties(oc.scn_slope, a['alpha'], a['shift'], total=0.2493)
    tt = np.linspace(-20.0, 55.0, 301)
    by = np.array([g['w'] - oc.sle_split_np(g['w'], g['M'], g['Tm'], g['dH'], g['dv'], t) for t in tt]) * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    ax1.plot(tt, by.sum(axis=1), color=SERIES[0], label='4 группы (multi-solid)')
    ax1.plot(a['T'], a['legacy'], color=INK2, ls='--', lw=1.4, label='прежняя модель (6.1)')
    ax1.plot(a['T'], a['exp'], 'o', ms=5, mfc='white', mec=INK, mew=1.2, label='Li et al., 2024 (ДСК)')
    ax1.set_xlabel('T, °C')
    ax1.set_ylabel('выпало парафина, % масс.')
    ax1.legend(fontsize=7.5)
    ax1.set_title('Нефть Жетыбая, 24.93 % парафина', fontsize=9, color=INK)
    for k in range(by.shape[1]):
        ax2.plot(tt, by[:, k], color=SERIES[k], ls=STYLES[k], label=GROUP_NAMES[k])
    ax2.set_xlabel('T, °C')
    ax2.set_ylabel('выпало, % масс. нефти')
    ax2.set_title('По группам', fontsize=9, color=INK)
    ax2.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    _save(fig, 'fig03_precipitation')


def fig_pressure(cal):
    b = cal['pressure']
    p = np.array(b['P'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    shift = np.array(b['wat_poynting']) - b['wat_poynting'][0]
    ax1.plot(p, shift, color=SERIES[0], label=f'модель (Пойнтинг), dv/v_L = {b["dv_frac"]:.3f}')
    for k, (c, v) in enumerate(b['sandyga_shift'].items()):
        ax1.plot(p, v, color=SERIES[k + 1], ls=STYLES[k + 1], lw=1.4,
                 label=f'Sandyga et al., 2020: {100 * float(c):.0f} % парафина')
    ax1.set_xlabel('P, МПа')
    ax1.set_ylabel('WAT(P) − WAT(0.1 МПа), °C')
    ax1.legend(fontsize=7)
    ax1.set_title('Дегазированная нефть', fontsize=9, color=INK)
    ax2.plot(p, b['wat_poynting'], color=INK2, ls='--', lw=1.4, label='без газа')
    ax2.plot(p, b['wat_gas'], color=SERIES[0], label='с растворенным газом')
    ax2.axvline(P_bubble / 1e6, color=GRID, lw=1.2)
    ax2.text(P_bubble / 1e6 + 0.4, max(b['wat_poynting']) - 0.3, 'P_b', color=INK2, fontsize=8, va='top')
    ax2.set_xlabel('P, МПа')
    ax2.set_ylabel('WAT, °C')
    ax2.legend(fontsize=7.5, loc='lower right')
    ax2.set_title(f'Газосодержание {oc.N_GAS_B * 1e3:.2f} моль/кг при P_b', fontsize=9, color=INK)
    fig.tight_layout()
    _save(fig, 'fig04_wat_pressure')


# --- 5. Асфальтены --------------------------------------------------------------------------------------------

def _soluble(p, t, w_res=sara_resins):
    rest = 1.0 - oc.WAX_TOTAL - sara_asphaltenes - w_res
    d_m = oc.delta_maltene_py(rest * oc.F_SAT_REST + oc.WAX_TOTAL, rest * (1.0 - oc.F_SAT_REST), w_res, p, t)
    dd = (oc.DELTA_ASPH - d_m) * 1e3
    phi = min(1.0, math.exp(v_asph / oc.V_M - 1.0 - v_asph * dd * dd / (R * (t + 273.15))))
    return phi * ro_asph / (phi * ro_asph + (1.0 - phi) * ro_o)


def fig_asphaltenes():
    p = np.linspace(0.5e6, 20e6, 300)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for k, t in enumerate((init_T - 20.0, init_T, init_T + 20.0)):
        prec = [100 * max(0.0, sara_asphaltenes - _soluble(x, t)) / sara_asphaltenes for x in p]
        ax1.plot(p / 1e6, prec, color=SERIES[k], ls=STYLES[k], label=f'T = {t:.0f} °C')
    ax1.axvline(P_bubble / 1e6, color=GRID, lw=1.2)
    ax1.axvline(P_onset_asph / 1e6, color=GRID, lw=1.2)
    ax1.text(P_bubble / 1e6 - 0.2, 102, 'P_b', ha='right', color=INK2, fontsize=8)
    ax1.text(P_onset_asph / 1e6 + 0.2, 102, 'AOP', ha='left', color=INK2, fontsize=8)
    ax1.set_xlabel('P, МПа')
    ax1.set_ylabel('выпало асфальтенов, % от содержания')
    ax1.set_ylim(0, 108)
    ax1.legend(fontsize=7.5)
    for k, mult in enumerate((0.5, 1.0, 2.0)):
        prec = [100 * max(0.0, sara_asphaltenes - _soluble(x, init_T, sara_resins * mult)) / sara_asphaltenes for x in p]
        ax2.plot(p / 1e6, prec, color=SERIES[k], ls=STYLES[k], label=f'смолы {100 * sara_resins * mult:.1f} %')
    ax2.set_xlabel('P, МПа')
    ax2.set_ylabel('выпало асфальтенов, %')
    ax2.set_ylim(0, 108)
    ax2.legend(fontsize=7.5)
    ax2.set_title(f'T = {init_T:.0f} °C', fontsize=9, color=INK)
    fig.tight_layout()
    _save(fig, 'fig05_asphaltenes')


# --- 6-7. Реология и гель -------------------------------------------------------------------------------------

def fig_viscosity(cal):
    c = cal['rheology']
    t = np.array(c['T'])
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.semilogy(t, c['kd'], color=INK2, ls='--', lw=1.4, label='Кригер–Догерти (прежняя)')
    ax.semilogy(t, c['pr_d'], color=SERIES[1], ls='-.', lw=1.6, label='Pedersen–Rønningsen, D = 18.12')
    full = np.minimum(np.array(c['pr_full']), 5e3)
    ax.semilogy(t, full, color=SERIES[2], ls=':', lw=1.6, label='P–R, полная формула (обрезано 5000)')
    ax.semilogy(t, c['bingham'], color=SERIES[0], label=f'модель: μ_p·e^(Dφ) + τ_y/γ̇, D = {c["visc_D"]:.2f}')
    ax.semilogy(t, c['mu'], 'o', ms=5, mfc='white', mec=INK, mew=1.2, label='Li et al., 2024, 150 1/с')
    ax.set_xlabel('T, °C')
    ax.set_ylabel('вязкость, мПа·с')
    ax.legend(fontsize=7.2)
    _save(fig, 'fig06_viscosity')


def fig_gel(cal):
    c = cal['rheology']
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.0, 2.8))
    phi = np.linspace(0.0, 0.2, 200)
    tau = np.where(phi > gel_phi, gel_tau_ref * np.clip((phi - gel_phi) / (gel_phi_ref - gel_phi), 0, None) ** gel_n, 0.0)
    ax1.plot(100 * phi, tau, color=SERIES[0])
    phi_pts = np.array(c['phi'])
    tau_pts = np.maximum(0.0, (np.array(c['mu']) - np.array(c['mu_l']) * np.exp(c['visc_D'] * phi_pts)) * 1e-3 * 150.0)
    ax1.plot(100 * phi_pts[tau_pts > 0], tau_pts[tau_pts > 0], 'o', ms=4.5, mfc='white', mec=INK, mew=1.1)
    ax1.set_xlabel('φ_s, % об.')
    ax1.set_ylabel('τ_y, Па')
    ax1.set_title('Предел текучести', fontsize=9, color=INK)
    xi = np.linspace(0, 1.2, 200)
    ax2.plot(xi, [br_factor(x) for x in xi], color=SERIES[0])
    ax2.set_xlabel('ξ = τ_y / τ_w')
    ax2.set_ylabel('F(ξ)')
    ax2.set_title('Букингем–Райнер', fontsize=9, color=INK)
    fi = np.tile(fi_0, (1, 1, 1))
    g = np.logspace(3, 8, 200)
    for k, ty in enumerate((0.1, 1.0, 10.0)):
        ax3.semilogx(g, [gel_phi_eq(0, 0, fi, x, ty) for x in g], color=SERIES[k], ls=STYLES[k], label=f'τ_y = {ty:g} Па')
    ax3.set_xlabel('|∇p|, Па/м')
    ax3.set_ylabel('Φ')
    ax3.set_title('Подвижность пучка fi_0', fontsize=9, color=INK)
    ax3.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, 'fig07_gel')


# --- 8. Демонстрационный расчет ----------------------------------------------------------------------------------

# Демонстрационные варианты (`demo_composition.VARIANTS`): механизмы добавляются по одному
DEMO_VARIANTS = [('legacy', 'прежняя модель'), ('wax', 'группы, WAT(P), вязкость P–R'), ('asph', '+ асфальтены, смолы'),
                 ('nogel', '+ вязкость от давления'), ('full', '+ гель')]


def _load(name):
    for path in (DATA / f'demo_{name}_processed_data.pkl', DATA / f'demo_{name}_comp_processed_data.pkl'):
        if path.is_file():
            with open(path, 'rb') as f:
                return pickle.load(f)[1]
    return None


def fig_demo():
    full = _load('full')
    if full is None:
        print('   нет расчетов demo_* - сначала python demo_composition.py')
        return
    comp = full['Composition']
    ext = [0, X_max, 0, Y_max]
    idx = len(full['Time']) - 1
    days = full['Time'][idx] / 86400.0

    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.9))
    # Асфальтены копятся в нескольких ячейках у добывающей скважины - их карты показаны в углу 40 x 40 м
    n = max(2, full['k'].shape[1] // 5)
    zoom = [X_max * (1.0 - n / full['k'].shape[1]), X_max, Y_max * (1.0 - n / full['k'].shape[2]), Y_max]
    corner = (slice(-n, None), slice(-n, None))
    panels = [
        # Где кристаллы есть, нефть насыщена и T = WAT: ниже нуля разность не опускается
        ('T − WAT, °C\n(0 — в нефти есть кристаллы)', full['Temperature'][idx] - comp['WAT'][idx], ext),
        ('осадок парафина,\nдоля m0', full['Wps dep'][idx], ext),
        ('осадок асфальтенов и смол,\nдоля m0 (угол у добывающей)', comp['Asph dep'][idx][corner], zoom),
        ('множитель подвижности Φ', full['Gel']['Phi'][idx], ext),
        ('k / k0', full['k'][idx], ext),
        ('флокулы в нефти, ppm масс.\n(угол у добывающей)', 1e6 * comp['Asph flocs'][idx][corner], zoom),
    ]
    for ax, (title, field, extent) in zip(axes.flat, panels):
        im = ax.imshow(field.T, origin='lower', extent=extent, cmap=BLUES, interpolation='nearest')
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
    fig.suptitle(f'Полная модель, {days:.0f} сут: нагнетательная (0, 0), добывающая ({X_max:.0f}, {Y_max:.0f})',
                 fontsize=9, color=INK)
    fig.tight_layout()
    _save(fig, 'fig08_demo_maps')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.1))
    styles = STYLES + [(0, (6, 1.5, 1, 1.5, 1, 1.5))]
    for k, (name, label) in enumerate(DEMO_VARIANTS):
        data = _load(name)
        if data is None:
            continue
        t = data['Time'] / 86400.0 / 365.0
        # Геометрия и начальная насыщенность в демо те же, что в constants.py (правятся только сетка и давления)
        kin = np.abs(data['Wells_accumulated']['Producer_Q_oil']) / geological_reserves
        # Пять кривых сходятся к концу, прямые подписи наложились бы - итоговый КИН в легенде
        ax1.plot(t, 100 * kin, color=SERIES[k], ls=styles[k], label=f'{label}: {100 * kin[-1]:.1f} %')
    ax1.set_xlabel('время, лет')
    ax1.set_ylabel('КИН, %')
    ax1.legend(fontsize=6.5, loc='lower right')
    tot = full['Totals']
    t = full['Time'] / 86400.0 / 365.0
    dep_wax = sum(tot[f'wax {k + 1} deposited'] for k in range(N_W))
    # Цвета 6-7: первые пять слева заняты вариантами расчета
    ax2.plot(t, dep_wax / 1e3, color=SERIES[5], label='парафин (все группы)')
    ax2.plot(t, (tot['asph flocs deposited'] + tot['resins deposited']) / 1e3, color=SERIES[6], ls='--',
             label='асфальтены + смолы')
    ax2.set_xlabel('время, лет')
    ax2.set_ylabel('в отложениях, т')
    ax2.legend(fontsize=7.5)
    fig.tight_layout()
    _save(fig, 'fig09_demo_totals')


def demo_text():
    """Текст раздела «Демонстрационный расчет» по результатам `demo_composition.py` -> docs/demo_results.md.

    Числа в тексте берутся из расчетов, поэтому текст пересобирается вместе с рисунками.
    """
    runs = {name: _load(name) for name, _ in DEMO_VARIANTS}
    if any(d is None for d in runs.values()):
        print('   demo_results.md не записан: нужны все варианты demo_composition.py')
        return
    full = runs['full']

    def kin(d):
        return float(np.abs(d['Wells_accumulated']['Producer_Q_oil'][-1]) / geological_reserves)

    def inj(d):
        return float(np.abs(d['Wells_accumulated']['Injector_Q_water'][-1]))

    rows, prev = [], None
    for name, label in DEMO_VARIANTS:
        d = runs[name]
        delta = '' if prev is None else f'{100 * (kin(d) - kin(prev)):+.1f}'
        rows.append(f'| {label} | {100 * kin(d):.1f} | {delta} | {inj(d) / 1e3:.0f} | '
                    f'{100 * d["Wps dep"][-1].max():.1f} | {d["k"][-1][0, 0]:.2f} | {d["k"][-1][-1, -1]:.2f} |')
        prev = d
    table = '\n'.join(rows)
    d_kin = {name: 100 * kin(runs[name]) for name, _ in DEMO_VARIANTS}

    comp, tot = full['Composition'], full['Totals']
    idx = len(full['Time']) - 1
    years = full['Time'][idx] / 86400.0 / 365.0
    wat, wat0 = comp['WAT'][idx], comp['WAT'][0]
    phi = full['Gel']['Phi'][idx]
    dep_wax = sum(tot[f'wax {g + 1} deposited'][idx] for g in range(N_W))
    dep_asph = tot['asph flocs deposited'][idx] + tot['resins deposited'][idx]
    by_group = [tot[f'wax {g + 1} deposited'][idx] / max(dep_wax, 1e-30) for g in range(N_W)]
    asph_dep = comp['Asph dep'][idx]
    p_min = float(full['Pressure'][idx].min() / 1e6)

    # Выпадение при закачиваемой температуре: дегазированная нефть (прежняя модель) и живая при 12 МПа
    from paraphin.constants import P_ref_wax, Twater
    from paraphin.equations.Wp_balance import _wp_saturated
    w0 = oc.WAX_W0.sum()
    live = w0 - oc.sle_split_np(oc.WAX_W0, oc.WAX_M, oc.WAX_TM_K, oc.WAX_DH, oc.WAX_DV, Twater,
                                dp=12e6 - P_ref_wax, n_g=oc.gas_moles(12e6)).sum()
    dead = w0 - _wp_saturated(w0, Twater)
    visc = math.exp(alpha_p_visc * (12e6 - P_ref_wax))

    text = f"""Постановка — элемент заводнения, как в `start.py`: пласт 200 x 200 x 10 м, сетка 50 x 50. Нагнетательная
скважина в углу (0, 0) закачивает воду при {Twater:.0f} °C и забойном давлении 17 МПа, добывающая в углу (200, 200)
работает при 7 МПа. Начальное давление 12 МПа, начальная температура 70 °C. Давление насыщения 9.5 МПа
(Узень XIII), давление начала осаждения асфальтенов 11 МПа (демонстрационное): у добывающей скважины давление
проходит оба порога. Состав — нефть Жетыбая (раздел 2). Расчет на {years:.0f} лет (`python demo_composition.py`)
выполнен в пяти вариантах: к прежней модели механизмы добавляются по одному, и разность соседних строк
табл. 2 — вклад одного механизма.

Таблица 2. Результаты на конец расчета: КИН и его изменение относительно предыдущей строки, закачано воды,
наибольший осадок парафина (доля начального порового объема), проницаемость у нагнетательной и у добывающей
скважин.

| Вариант | КИН, % | ΔКИН, п.п. | закачано, тыс. м³ | осадок парафина, % m0 | k/k0 нагнет. | k/k0 добыв. |
|---|---|---|---|---|---|---|
{table}

Карты полной модели на конец расчета — рис. 8, КИН и отложения во времени — рис. 9.

- **Группы парафина, WAT(P), вязкость P–R** ({d_kin["wax"] - d_kin["legacy"]:+.1f} п.п.). Прежняя модель
  описывает дегазированную нефть, а в пласте нефть живая: растворенный газ разбавляет раствор. При
  {Twater:.0f} °C и 12 МПа выпадает {100 * live:.1f} % масс. парафина против {100 * dead:.1f} % у дегазированной
  нефти. Осадок у нагнетательной скважины меньше, ее приемистость выше, КИН растет.
- **Асфальтены и смолы** ({d_kin["asph"] - d_kin["wax"]:+.1f} п.п.). Флокулы образуются там, где давление
  опускается к давлению насыщения, — в окрестности добывающей скважины (наименьшее давление в пласте
  {p_min:.1f} МПа). Они оседают почти сразу, как образовались (осаждение ограничено подводом), поэтому их доля
  в нефти не превышает {100 * comp['Asph flocs'][idx].max():.1e} % масс. Зато осадок асфальтенов со смолами
  копится в ячейке скважины: в полной модели это {100 * asph_dep[-1, -1]:.0f} % начального порового объема.
  Это классическое повреждение призабойной зоны асфальтенами при снижении давления. Всего в осадке
  {dep_asph / 1e3:.0f} т асфальтенов и смол.
- **Вязкость от давления** ({d_kin["nogel"] - d_kin["asph"]:+.1f} п.п.). Множитель Баруса при 12 МПа —
  {visc:.2f}. Это оценка сверху: снижение вязкости растворенным газом не учтено (раздел 12).
- **Гель** ({d_kin["full"] - d_kin["nogel"]:+.1f} п.п.). Множитель подвижности Φ < 0.5 на
  {100 * float((phi < 0.5).mean()):.0f} % площади — в остывшей и закольматированной зоне у нагнетательной
  скважины. Там нефть почти неподвижна (подвижность ограничена снизу долей {gel_mobility_min} от пластической),
  и вода ее обходит. Вклад геля в КИН мал: неподвижная нефть заперта в зоне, которую вода уже прошла.
  Предел текучести в поре взят объемным (`gel_tau_mult` = 1): так его подтверждает керн той же нефти
  (`experiments/состав/валидация_АСПО.docx`, раздел 5). Для легких нефтей он в десятки раз меньше.

WAT в полной модели. В начале расчета WAT составляет {wat0.min():.1f}–{wat0.max():.1f} °C: при 12 МПа
поправка Пойнтинга не перекрывает понижения растворенным газом (рис. 4), и WAT ниже, чем у дегазированной нефти
(45.3 °C). К концу расчета WAT лежит в пределах {wat.min():.1f}–{wat.max():.1f} °C. Минимум — в холодной зоне у
нагнетательной скважины: там тяжелые группы выпали и осели, а оставшаяся нефть насыщена при местной
температуре, так что ее WAT совпадает с температурой пласта.

Состав осадка. Всего осело {dep_wax / 1e3:.0f} т парафина, по группам C17–22 / C23–30 / C31–40 / C41–60 —
{" / ".join(f"{100 * x:.0f}" for x in by_group)} %. Больше всего оседает тяжелых парафинов и восков, а мягкие
парафины при {Twater:.0f} °C почти целиком остаются в растворе.

![](figures/fig08_demo_maps.png)

Рис. 8. Полная модель на конец расчета: превышение температуры над местной WAT (ноль — в нефти есть
кристаллы); осадок парафина (доля начального порового объема); множитель подвижности геля Φ; проницаемость
k/k0. Осадок асфальтенов со смолами и флокулы в нефти сосредоточены у добывающей скважины и показаны для
угла 40 x 40 м.

![](figures/fig09_demo_totals.png)

Рис. 9. КИН пяти вариантов (слева) и масса отложений в пласте в полной модели (справа).
"""
    (HERE / 'demo_results.md').write_text(text, encoding='utf-8')
    print('   demo_results.md')


def main():
    cal = json.loads((ROOT / 'experiments' / 'results' / 'calibration.json').read_text(encoding='utf-8'))
    print('Рисунки:')
    fig_scheme()
    fig_scn(cal)
    fig_precip(cal)
    fig_pressure(cal)
    fig_asphaltenes()
    fig_viscosity(cal)
    fig_gel(cal)
    fig_demo()
    demo_text()


if __name__ == '__main__':
    main()
