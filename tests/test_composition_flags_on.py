"""Детальный состав нефти со включенными флагами: законы сохранения, тождества, отсутствие гонок.

Флаги - константы уровня модуля (numba вшивает их в машинный код), поэтому расчет со включенными флагами
идет в копии пакета с поправленным constants.py (`tests/_patched_copy.py`), отдельным процессом. Первый
запуск компилирует копию (~1 мин на сценарий), дальше ее кеш переиспользуется.
"""
import numpy as np
import pytest

from tests._patched_copy import ROOT, run_worker
from tests.regression_baseline import BASELINE

WORKER = ROOT / 'tests' / '_flags_on_worker.py'

# Все механизмы сразу на небольшой сетке. Давления выбраны так, чтобы поле давления пересекало и давление
# начала осаждения асфальтенов (11 МПа), и давление насыщения (9.5 МПа): пласт 12 МПа, отбор при 7 МПа.
ALL_ON = {'Nx, Ny': '20, 20', 'wax_components': 'True', 'wax_pressure': 'True', 'asphaltenes': 'True',
          'gelation': 'True', 'wax_viscosity': '1', 'Pw': '170 * bar_to_pa', 'Po': '70 * bar_to_pa'}


# Все модели кинетики осаждения поверх ALL_ON (`equations/Deposition.py`): кристаллизация, перенос к стенке, вынос,
# агрегация, снежный ком, адсорбция, старение гель-отложения
KIN_ALL = dict(ALL_ON, wax_kinetics='True', wall_transport='True', entrainment='True', asph_aggregation='True',
               snowball='True', adsorption='True', deposit_aging='True')


@pytest.fixture(scope='module')
def all_on():
    return run_worker('all', ALL_ON, WORKER, args=(300.0, 2))['runs']


def test_conservation(all_on):
    r = all_on[0]
    assert r['worst_q'] < 1e-6, 'закачка не равна отбору'
    assert r['worst_pore'] < 1e-10, '(q_p1 + q_p2 + q_pa)*dt != m - m_new'
    assert r['worst_dep'] < 1e-10, 'объем отложений не равен потере пористости'
    # Балансы массы по каждой группе парафина, асфальтенам (растворенные + флокулы) и смолам
    assert max(abs(x) for x in r['balance']) < 1e-8, r['balance']


def test_closure(all_on):
    r = all_on[0]
    assert r['worst_closure'] < 1e-12, 'w_o + w_p + w_ps != 1'
    assert r['worst_groups'] < 1e-12, 'сумма групп парафина != w_p + w_ps'
    assert r['min_frac'] >= 0.0


def test_mechanisms_active(all_on):
    """Проверки выше не пустые: парафин и асфальтены оседают, гель образуется."""
    r = all_on[0]
    assert r['dep_wax'] > 0.0 and r['dep_asph'] > 0.0 and r['dep_resin'] > 0.0
    assert r['max_flocs'] > 0.0 or r['dep_asph'] > 0.0
    assert r['phi_min'] < 0.5, 'гель не образовался'
    assert 0.0 <= r['phi_min'] <= r['phi_max'] <= 1.0


def test_stability(all_on):
    r = all_on[0]
    assert r['phi_flips'] <= 10, 'множитель подвижности геля колеблется'
    assert r['min_dt'] > r['dt_min'], 'шаг упал до нижней границы'


def test_no_race(all_on):
    """Два одинаковых прогона побитово равны: гонок в prange (буферы Fo[i], new_Ws[i, j]) нет."""
    assert all_on[0]['fingerprint'] == all_on[1]['fingerprint']


def test_single_group_reproduces_legacy(tmp_path):
    """Детальный состав с одной группой (режим 'single') - прежняя термодинамика, но через перенос по
    потокам граней, стоки по группам и носитель скрытой теплоты Hl. Совпасть побитово он не обязан
    (другой порядок сложения), но обязан совпасть с эталоном до округления, накопленного за 100 сут
    (фактически ~1e-13)."""
    if not BASELINE.is_file():
        pytest.skip('нет эталона')
    fields = tmp_path / 'single.npz'
    r = run_worker('single', {'wax_components': 'True', 'wax_characterization': "'single'"}, WORKER,
                   args=(100.0, 1, fields))['runs'][0]
    assert max(abs(x) for x in r['balance']) < 1e-8
    ref, got = np.load(BASELINE), np.load(fields)
    assert np.isclose(got['KIN'], ref['KIN'], rtol=1e-9)
    for name in ('p', 'S', 'T', 'm', 'k', 'Wp', 'Wps'):
        scale = np.abs(ref[name]).max()
        assert np.abs(got[name] - ref[name]).max() <= 1e-9 * scale, name


@pytest.fixture(scope='module')
def kin_all():
    return run_worker('kin_all', KIN_ALL, WORKER, args=(120.0, 2))['runs']


def test_kinetics_conservation(kin_all):
    """Балансы массы каждой группы, асфальтенов (с адсорбированными) и смол; тождества пористости и осадка
    с новыми каналами (стеночная кристаллизация, старение, адсорбция, вынос)."""
    r = kin_all[0]
    assert max(abs(x) for x in r['balance']) < 1e-8, r['balance']
    assert r['worst_pore'] < 1e-10
    assert r['worst_dep'] < 1e-10
    assert r['worst_closure'] < 1e-12 and r['worst_groups'] < 1e-12
    assert r['min_frac'] >= 0.0


def test_kinetics_active_and_stable(kin_all):
    r = kin_all[0]
    assert r['dep_wax'] > 0.0 and r['dep_asph'] > 0.0
    assert r['min_dt'] > r['dt_min'], 'шаг упал до нижней границы'
    assert kin_all[0]['fingerprint'] == kin_all[1]['fingerprint'], 'два прогона разошлись: гонка в prange'
