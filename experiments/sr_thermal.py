"""Тепловая постановка опытов Sutton & Roberts: горячая нефть входит в керн, который охлаждается снаружи.

    python experiments/sr_thermal.py        # прогоны (кэш) -> results/sr_thermal.json

У Ring et al. (1994) нефть подается при 54.4 C в керн при 21.1 C. Режим 'thermal' `core_runner.py` задавал вдоль
керна линейный профиль температуры, неизменный во времени, и с ним опыты описываются хуже (СКО 0.16), чем в
изотермической постановке, где холодная нефть уже несет равновесную взвесь (0.034). Здесь профиль не задается,
а считается уравнением энергии с теплообменом через боковую поверхность керна с термостатом при T керна:
q = hv*(T - T_core), hv = U*P/A (`heat_losses = 3`, `hv_bath`), где U - коэффициент теплопередачи через манжету и
кернодержатель, P/A = 4/a у квадратного сечения со стороной a. Нефть остывает на входной длине
L = u*rho_o*c_o/hv: при U ~ 10 Вт/(м^2*C) - около 10 см из 30. U в опыте не измерен, поэтому показан ряд значений:
от U -> 0 (нефть остывает только в конце керна) до U -> бесконечности (изотермический керн, режим 'rate').
Параметры кинетики - итоговые наборы (`params.json`): свой на каждый опыт.
"""
import sys

import sutton_roberts as sr
from common import run_many, load_params, save_results, rms_k

FLAGS = sr.LSQ_FAMILIES[sr.MAIN_FAMILY]['flags']  # основная модель опытов (сеть пор и горл + вынос)
U_BATH = (3.0, 10.0, 30.0, 100.0, 300.0)  # [Вт/(м^2*C)]


def _hv(n, u):
    return 4.0 * u / sr.EXPS[n]['core']['side']


def entrance_length(n, u):
    """Входная тепловая длина L = u*rho_o*c_o/hv, [м] (c_o нефти - из constants.py)."""
    from paraphin.constants import c_o
    e = sr.exp_dict(n)
    darcy = e['q'] / e['side'] ** 2
    return darcy * e['ro_o'] * c_o / _hv(n, u)


def jobs(kin_of):
    out = []
    for n in (1, 2):
        out.append(sr.job('ent', n, FLAGS, kin_of(n)))  # изотермический керн - предел U -> inf
        for u in U_BATH:
            flags = dict(FLAGS, heat_losses='3', hv_bath=repr(_hv(n, u)))
            out.append(sr.job(f'bath{u:g}', n, flags, kin_of(n), mode='thermal_bath'))
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = load_params('sutton_roberts_lsq')[sr.MAIN_FAMILY]
    own = lambda n: sr.lsq_kin(sr.MAIN_FAMILY, p[f'own{n}'])
    res = run_many(jobs(own))
    out = {'U': list(U_BATH), 'L': {n: [entrance_length(n, u) for u in U_BATH] for n in (1, 2)}, 'own': {}}
    per = 1 + len(U_BATH)
    for k, n in enumerate((1, 2)):
        block = res[k * per:(k + 1) * per]
        out['own'][n] = dict(iso=rms_k(block[0], sr.EXPS[n]['k_pv']),
                             bath=[rms_k(r, sr.EXPS[n]['k_pv']) for r in block[1:]],
                             curves=[dict(pv=r['pv'], k=r['k']) for r in block])
        print(f'опыт {n}: изотермический {out["own"][n]["iso"]:.3f}; с термостатом',
              {f'U={u:g} (L={l * 100:.0f} см)': round(v, 3)
               for u, l, v in zip(U_BATH, out['L'][n], out['own'][n]['bath'])}, flush=True)
    save_results('sr_thermal', out)


if __name__ == '__main__':
    main()
