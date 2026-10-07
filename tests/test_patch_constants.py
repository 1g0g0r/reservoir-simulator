"""Наборы подстановок проекта находят свои строки в constants.py, и подставленный файл исполняется.

Флаги и константы меняются только подстановкой текста в копию пакета (`tests/_patched_copy.py`), поэтому
переименование или перенос константы ломает чужие прогоны молча - до их запуска. Здесь каждый набор, которым
пользуются тесты, опыты, демо и статья, применяется к constants.py и исполняется вместе с проверками
согласованности флагов в конце файла - без компиляции и прогонов.
"""
import importlib.util

from tests._patched_copy import ROOT, patch_constants

EXPERIMENTS = ROOT / 'experiments'


def _module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_sets() -> dict:
    from tests.test_composition_flags_on import ALL_ON, KIN_ALL, EOS_ON
    demo = _module(ROOT / 'demo_composition.py')
    thermal = _module(ROOT / 'твт_статья_АСПО' / 'run_thermal.py')
    cf = _module(EXPERIMENTS / 'исходная_модель' / 'core_flood.py')
    common = _module(EXPERIMENTS / 'common.py')
    sr = _module(EXPERIMENTS / 'sutton_roberts.py')
    sd = _module(EXPERIMENTS / 'sandyga2020.py')
    li = _module(EXPERIMENTS / 'li2024.py')
    ml = _module(EXPERIMENTS / 'maloney2004.py')
    val = _module(EXPERIMENTS / 'состав' / 'validate.py')

    core = common.core_constants(sr.exp_dict(1))
    flood = cf._case_constants(cf.EXPERIMENTS[1], 15e-6, 3e-4, cf.NY, cf.DT)
    sd_exp, sd_extra = sd.exp_case()
    sets = {
        'ALL_ON': ALL_ON, 'KIN_ALL': KIN_ALL, 'EOS_ON': EOS_ON,
        'single': {'wax_components': 'True', 'wax_characterization': "'single'"},
        'run_grid_study': {'Nx, Ny': '100, 100'}, 'run_grid_study_nr': {'Nr': '45'},
        'run_cases': {'init_Wp': '0.0', 'heat_losses': '1'},
        'core_flood': dict(flood, Nr='21', r_m='16e-6', sigma_r='0.6', r_max='5e-5', init_T='54.4'),
        'li2024': li.constants(dict(li.RETENTION, wax_kinetics='True')),
        'maloney2004': ml.job(ml.WAX[1])[1],
        'sandyga2020': {**common.core_constants(sd_exp, dt=2.0), **sd_extra, **sd.KINETICS, 'pore_network': 'True'},
        'validate': {**flood, **val.COMP, **val.GEL, 'gel_tau_mult': '0.1', 'gel_deposit_weight': '0.0', 'gel_phi': '0.02'},
    }
    sets.update({f'sutton_roberts_{n}': dict(core, **getattr(sr, n))
                 for n in ('SINGLE', 'KERNEL', 'ENTRAINMENT', 'FILTRATION')})
    sets.update({f'demo_{n}': values for n, values in demo.VARIANTS.items()})
    sets.update({f'thermal_{n}': dict(thermal.COMMON, **values) for n, values in thermal.VARIANTS.items()})
    from tests.mrst_tests import common as mrst
    from tests.mrst_tests.test_thermal import PATCH as MRST_THERMAL
    sets['mrst_two_phase'] = {'Nx, Ny': '40, 40', **mrst.PATCH_TWO_PHASE}
    sets['mrst_thermal'] = {'Nx, Ny': '20, 20', **MRST_THERMAL}
    return sets


def test_patch_sets_apply(monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENTS / 'исходная_модель'))  # `import core_flood` модулей опытов
    monkeypatch.syspath_prepend(str(EXPERIMENTS))                      # `from common import ...`
    path = ROOT / 'paraphin' / 'constants.py'
    text = path.read_text(encoding='utf-8')
    for name, values in _patch_sets().items():
        patched = patch_constants(text, values)  # KeyError, если строки нет или она не одна
        exec(compile(patched, f'constants.py [{name}]', 'exec'), {'__file__': str(path)})
