"""Твердый раствор групп парафина в ядре (`wax_solid_solution`, `Thermo_wax.ss_split`) против эталона
`thermo.solids.UniquacSolidWax` с идеальной жидкостью на тех же группах, в копии пакета с группами SCN.

    python -m pytest tests/test_solid_solution.py -q
"""
import json
import sys
import time

PATCH = {'wax_characterization': "'scn'", 'wax_solid_solution': 'True'}
TEMPS = (70.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0)


def _work(out_path):
    import numpy as np
    from paraphin.constants import P_ref_wax, ro_wax_liq, ro_p
    from paraphin.equations.Thermo_wax import ss_split, wat_cell
    from paraphin.oil_composition import WAX_W0, WAX_M
    from paraphin.thermo import tables as tb
    from paraphin.thermo.characterization import coutinho_nalkane
    from paraphin.thermo.solids import MultiSolidWax, UniquacSolidWax

    eos, n, mw, _, iw, _ = tb.oil_fluid(0.0, gas=False, w_wax=WAX_W0, m_wax=WAX_M)
    g = np.arange(iw, iw + WAX_W0.size)
    nc = (WAX_M - 2.016) / 14.027
    t_m, dh_m, t_tr, dh_tr = coutinho_nalkane(nc)
    ms = MultiSolidWax(eos, g, t_m, dh_m, WAX_M * 1e-3 * (1.0 / ro_wax_liq - 1.0 / ro_p), P_ref_wax, t_tr, dh_tr)
    uq = UniquacSolidWax(ms, nc, liquid='ideal')
    z = n / n.sum()
    out = {'dev': [], 'dev_buf': [], 'sus': []}
    for t in TEMPS:
        ref = uq.precipitate(t + 273.15, P_ref_wax, z)[0][g] * mw[g] * n.sum()
        cold = np.full(WAX_W0.size, -1.0)
        ss_split(WAX_W0.copy(), t, P_ref_wax, cold)
        buf = np.abs(ref[::-1]) + 1e-3  # прежнее содержимое буфера на ответ не влияет
        ss_split(WAX_W0.copy(), t, P_ref_wax, buf)
        out['dev'].append(float(np.abs(cold - ref).max()))
        out['dev_buf'].append(float(np.abs(buf - cold).max()))
        out['sus'].append(float(ref.sum()))
    out['wat'] = [float(wat_cell(WAX_W0.copy(), P_ref_wax)), uq.wat(P_ref_wax, z) - 273.15]
    sus = np.full(WAX_W0.size, -1.0)
    t0 = time.perf_counter()
    for _ in range(2000):
        ss_split(WAX_W0.copy(), 20.0, P_ref_wax, sus)
    out['us_per_call'] = (time.perf_counter() - t0) / 2000 * 1e6
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f)


def test_solid_solution_kernel():
    from pathlib import Path
    from tests._patched_copy import run_worker
    r = run_worker('solid_solution', PATCH, Path(__file__), timeout=900.0)
    print(r)
    assert max(r['sus']) > 0.05 and min(r['sus']) == 0.0       # проверены и выпадение, и полное растворение
    assert max(r['dev']) < 1e-8                                  # массовые доли взвеси групп
    assert max(r['dev_buf']) == 0.0
    assert abs(r['wat'][0] - r['wat'][1]) < 0.01                 # бисекция WAT ядра против brentq эталона


if __name__ == '__main__':
    _work(sys.argv[1])
