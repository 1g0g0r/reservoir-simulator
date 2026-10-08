"""Граничные условия на границе области: paraphin против MRST (incomp, ad-blackoil, geothermal) и Баклея-Леверетта.

Одномерная полоса 200x200x10 м из 1x100 ячеек (Nx = 1, поток вдоль j, как у керна в experiments/core_runner.py) без
скважин: течение задают границы Left (j < 0; у MRST - ymin) и Right (ymax). Добывающая в последней ячейке - с
ничтожной продуктивностью (Solver.initialize требует ровно одну), дебиты - поток через правую границу. Варианты:
    dirichlet - давление Дирихле Pw на входе и Po на выходе, S = 1 на входе, начальная S = S_min; 300 сут. Против
                incomp и ad-blackoil и против аналитики Баклея-Леверетта до прорыва (решение зависит только от
                накопленной закачки, поэтому годится и при переменном притоке);
    neumann   - градиент давления g = (Pw - Po)/L на входе (наш Нейман: поток g*A*lambda), Po на выходе, полоса
                водонасыщена: подвижность постоянна, и MRST получает тот же расход g*A*k/mu_w через fluxside; 5 сут;
    thermal   - одна вода (S_max = 1), давление и температура Дирихле на обеих границах (вход - Pw и Twater, выход -
                Po и init_T), против geothermal; 150 сут - холодный фронт выходит из полосы.
Проводимость граничной грани на притоке у кодов разная (README.md, «Чем модели отличаются»): paraphin - гармоническое
среднее подвижностей ячейки и фиктивной ячейки при S из ГУ, incomp - подвижность ячейки, ad - втекающей воды. На
полосе из 100 ячеек половина граничной ячейки - 1/200 сопротивления, и это соглашение почти не сказывается.
MRST-часть - mrst_boundary_conditions.m (MATLAB/Octave; без аргументов - ручной запуск, возвращает результат).

    python -m pytest tests/mrst_tests/test_boundary_conditions.py -s       # ~2 мин
    python -m tests.mrst_tests.test_boundary_conditions --mrst              # эталон: 5 прогонов octave-cli, ~10 мин
"""
import argparse

import numpy as np
import pytest

from tests.mrst_tests import common as c

TEST = 'test_boundary_conditions'
N = 100
PATCH = {'Nx, Ny': f'1, {N}', **c.PATCH_TWO_PHASE}
# Без нефти геологические запасы нулевые, а КИН (solver._update_wells_data) делит на них
PATCH_THERMAL = {**PATCH, 'S_max': '1.0', 'init_S': 'S_max', 'geological_reserves': '1.0'}
# вариант: (сутки расчета, сутки полей, шаг MRST по решателям, [сут])
VARIANTS = {'dirichlet': (300, (20, 40, 100, 300), {'incomp': 0.05, 'ad': 0.25}),
            'neumann': (5, (1, 2, 3, 5), {'incomp': 0.05, 'ad': 0.25}),
            'thermal': (150, (25, 50, 100, 150), {'thermal': 0.1})}
BL_DAYS = (20, 40)  # до прорыва (~60 сут)


def geometry():
    """Длина полосы, сечение, [м], [м^2]; градиент Неймана, [Па/м], и равный ему расход MRST, [м^3/с]."""
    from paraphin.constants import X_min, X_max, Y_min, Y_max, h, Pw, Po, init_k, init_T
    from paraphin.utils.math_utils.fluids_correlations import calc_mu_w
    length, area = Y_max - Y_min, (X_max - X_min) * h
    grad = (Pw - Po) / length
    return length, area, grad, grad * area * init_k / float(calc_mu_w(init_T))


def physics() -> dict:
    from paraphin.constants import init_T, Twater, c_w, c_f, K_w, K_f, ro_f
    return c.physics(T0=init_T + 273.15, T_inj=Twater + 273.15, Cp_w=c_w, lambda_w=K_w, lambda_R=K_f, rho_R=ro_f,
                     Cp_R=c_f, ny=N, grad=geometry()[2])


def cases() -> dict:
    out = {}
    for variant, (days, field_days, dts) in VARIANTS.items():
        for solver, dt in dts.items():
            case = c.mrst_case(physics(), solver, N, days, dt, field_days)
            case.update(nx=1, ny=N, variant=variant, q_bc=geometry()[3])
            out[f'{solver}_n{N}_{variant}'] = case
    return out


def paraphin_jobs() -> dict:
    from paraphin.constants import Pw, Po, Twater, init_T, S_max
    jobs = {}
    for variant, (days, field_days, _) in VARIANTS.items():
        inlet = ['Pressure', 'Left', 'Neumann', geometry()[2]] if variant == 'neumann' else \
            ['Pressure', 'Left', 'Dirichlet', Pw]
        bc = [inlet, ['Pressure', 'Right', 'Dirichlet', Po], ['Saturation', 'Left', 'Dirichlet', 1.0]]
        opts = {'days': days, 'field_days': field_days, 'bc': bc}
        copy, patch = f'mrst_bc_n{N}', PATCH
        if variant == 'neumann':
            opts['S0'] = S_max
        if variant == 'thermal':
            bc += [['Temperature', 'Left', 'Dirichlet', Twater], ['Temperature', 'Right', 'Dirichlet', init_T]]
            opts['T_inj'] = Twater
            copy, patch = f'mrst_bc_thermal_n{N}', PATCH_THERMAL
        jobs[f'paraphin_n{N}_{variant}'] = (copy, patch, opts)
    return jobs


def buckley_leverett(Q, y):
    """Водонасыщенность по Баклею-Леверетту в точках y, [м], после закачки Q, [м^3]; положение и S фронта."""
    from paraphin.constants import S_min, S_max, n_power, init_m, init_T
    from paraphin.utils.math_utils.fluids_correlations import calc_mu_o, calc_mu_w
    s = np.linspace(S_min, S_max, 4001)
    x = (s - S_min) / (S_max - S_min)
    fw = x ** n_power / (x ** n_power + (1.0 - x) ** n_power * float(calc_mu_w(init_T)) / float(calc_mu_o(init_T, 0.0)))
    dfw = np.gradient(fw, s)
    i_f = 1 + np.argmax(fw[1:] / (s[1:] - S_min))  # касательная Велге из (S_min, 0)
    y_s = Q / (geometry()[1] * init_m) * dfw[i_f:]   # положение насыщенности s[i_f:], убывает с S
    S = np.full_like(y, S_min)
    behind = y <= y_s[0]
    S[behind] = np.interp(y[behind], y_s[::-1], s[i_f:][::-1])
    return S, y_s[0], s[i_f]


def plot_strip(file, runs, names, field_days, rows, title, analytic=None) -> None:
    """Профили вдоль полосы: строки rows [(ключ, множитель, подпись)], столбцы - сутки; analytic(сутки) -> S
    Баклея-Леверетта поверх водонасыщенности (черный пунктир)."""
    plt = c._plt()
    y = (np.arange(N) + 0.5) * geometry()[0] / N
    fig, axs = plt.subplots(len(rows), len(field_days), figsize=(3.8 * len(field_days), 3.2 * len(rows)),
                            layout='constrained', sharex=True, squeeze=False)
    for col, day in enumerate(field_days):
        for row, (key, scale, label) in enumerate(rows):
            ax = axs[row, col]
            for k, name in enumerate(names):
                c._draw(ax, y, runs[name][key][col] * scale, name, k)
            if analytic and key == 'sw':
                ax.plot(y, analytic(day), color='#0b0b0b', ls=':', lw=1.5, label='Баклей–Леверетт')
            ax.set_title(f'{day} сут: {label}', loc='left')
        axs[-1, col].set_xlabel('расстояние от входа, м')
    axs[0, 0].legend(frameon=False)
    fig.suptitle(title, x=0.01, ha='left')
    fig.savefig(c.FIG / file, dpi=150)
    plt.close(fig)


@pytest.fixture(scope='module')
def result():
    runs = c.load_reference(TEST, physics())
    runs.update(c.paraphin_runs(paraphin_jobs()))
    diffs = {}
    for variant, (days, field_days, dts) in VARIANTS.items():
        ref = f'paraphin_n{N}_{variant}'
        diffs.update(c.report(f'{TEST}_{variant}', runs, {f'{s}_n{N}_{variant}': ref for s in dts}, field_days))
    y = (np.arange(N) + 0.5) * geometry()[0] / N

    def bl(day):
        return buckley_leverett(runs[f'paraphin_n{N}_dirichlet']['Q_inj'][day - 1], y)[0]

    names = [f'paraphin_n{N}_dirichlet', f'incomp_n{N}_dirichlet', f'ad_n{N}_dirichlet']
    c.plot_series(f'{TEST}_dirichlet_rates.png', [('Приток через границу, м³/сут', 'q_inj', names, 1.0),
                                                  ('Отток нефти, м³/сут', 'q_o', names, 1.0),
                                                  ('Обводненность оттока, доли', 'eta', names, 1.0)], runs)
    plot_strip(f'{TEST}_dirichlet_profiles.png', runs, names, VARIANTS['dirichlet'][1][:3],
               [('sw', 1.0, 'S_w'), ('p', 1e-5, 'p, бар')], 'Дирихле по давлению, S = 1 на входе', bl)
    names = [f'paraphin_n{N}_neumann', f'incomp_n{N}_neumann', f'ad_n{N}_neumann']
    plot_strip(f'{TEST}_neumann_profiles.png', runs, names, VARIANTS['neumann'][1][-1:],
               [('p', 1e-5, 'p, бар')], f'Нейман: градиент {geometry()[2] / 1e5:g} бар/м на входе')
    names = [f'paraphin_n{N}_thermal', f'thermal_n{N}_thermal']
    for name in names:  # температура для рисунков - в °C
        runs[name]['T_C'] = runs[name]['T'] - 273.15
        runs[name]['T_prod_C'] = runs[name]['T_prod'] - 273.15
    c.plot_series(f'{TEST}_thermal_rates.png', [('Приток через границу, м³/сут', 'q_inj', names, 1.0),
                                                ('Температура на выходе, °C', 'T_prod_C', names, 1.0)], runs)
    plot_strip(f'{TEST}_thermal_profiles.png', runs, names, VARIANTS['thermal'][1][:3],
               [('T_C', 1.0, 'T, °C'), ('p', 1e-5, 'p, бар')], 'Дирихле по давлению и температуре')
    return runs, diffs


def _pick(diffs, variant):
    return {name: d for name, d in diffs.items() if name.endswith(variant)}


def test_dirichlet_rates(result):
    """Приток через границу Дирихле и отток нефти (норма L2 суточных), накопленная нефть."""
    c.assert_limits(_pick(result[1], 'dirichlet'), {'q_inj': 0.02, 'q_o': 0.03, 'Qo': 0.005})


def test_dirichlet_water_cut(result):
    c.assert_limits(_pick(result[1], 'dirichlet'), {'eta_mean': 0.005, 't_bt': 5})


def test_dirichlet_fields(result):
    c.assert_limits(_pick(result[1], 'dirichlet'), {'dS_mean': 0.005, 'dS_max': 0.1, 'dp_rms': 1.0, 'dp_max': 2.0})


def test_inflow_face_mobility(result):
    """Подвижность на грани притока у кодов разная, но на полосе из 100 ячеек это 1/200 сопротивления: приток
    первых суток у всех в пределах 1 %."""
    c.assert_limits(_pick(result[1], 'dirichlet'), {'q_inj_1': 0.01})


def test_buckley_leverett(result):
    """До прорыва профиль водонасыщенности paraphin - решение Баклея-Леверетта по его же накопленной закачке:
    средняя разность по полосе и положение фронта (S = (S_ф + S_min)/2) с точностью до размытия схемы."""
    from paraphin.constants import S_min
    runs, _ = result
    y = (np.arange(N) + 0.5) * geometry()[0] / N
    r = runs[f'paraphin_n{N}_dirichlet']
    for day in BL_DAYS:
        k = VARIANTS['dirichlet'][1].index(day)
        S_bl, y_front, s_front = buckley_leverett(r['Q_inj'][day - 1], y)
        S = r['sw'][k]
        y_num = y[np.argmax(S < 0.5 * (s_front + S_min))]
        err = np.abs(S - S_bl).mean()
        assert err < 0.02, f'{day} сут: средняя |S - S_БЛ| = {err:.4f}'
        assert abs(y_num - y_front) < 0.05 * geometry()[0], f'{day} сут: фронт {y_num:.1f} м против {y_front:.1f} м'


def test_neumann(result):
    """Нейман: один и тот же расход и проводимости - поле давления и поток совпадают до точности решателей."""
    c.assert_limits(_pick(result[1], 'neumann'), {'q_inj': 1e-6, 'dp_max': 1e-3, 'dS_max': 1e-6})


def test_thermal_boundary(result):
    """Дирихле по температуре: допуски как в test_thermal - расхождение температуры от нагрева трением в MRST."""
    from tests.mrst_tests.test_thermal import dT_friction
    c.assert_limits(_pick(result[1], 'thermal'), {'q_inj': 0.05, 'T_prod_max': dT_friction() + 1.0,
                                                   'dT_mean': dT_friction(), 'dT_max': dT_friction() + 1.0,
                                                   'dp_rms': 2.0, 'dp_max': 4.0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--mrst', action='store_true', help='снять эталон MRST в Octave')
    if parser.parse_args().mrst:
        c.save_reference(TEST, physics(), cases())
