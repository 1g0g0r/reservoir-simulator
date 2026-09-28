"""Все сравнения с опытами одной командой: прогоны, подбор, рисунки, сводные таблицы отчета.

    python experiments/run_all.py                  # полный: подборы по сеткам и least_squares (часы на 4 ядрах)
    python experiments/run_all.py --quick          # только итоговые наборы из params.json (~20 мин на 4 ядрах)
    python experiments/run_all.py --plot           # только рисунки и таблицы по готовым results/*.json
    python experiments/run_all.py li2024 he2020    # выборочно (в любом режиме)

Прогоны кэшируются в results/cache (по хешу констант, параметров, опыта и исходников пакета), поэтому повторный
запуск без изменений ничего не считает, а после правки кода пересчитывает только затронутое. Результаты - в
results/<опыт>.json, рисунки - в figures/, итоговые параметры подборов - в params.json, таблицы для отчета
`сравнение_с_опытами.md` - в results/tables.json. Отчет в docx: `python docs/build_docx.py сравнение_с_опытами`.

Порядок важен: he2020 берет долю парафина в гель-отложении C0 из подбора sandyga2020.
"""
import json
import math
import sys
import time

from common import RESULTS, mode_from_argv

ORDER = ('sutton_roberts', 'li2024', 'sandyga2020', 'he2020', 'pore_models')
TARGET = 0.03  # СКО k/k0 (решение автора); где шумовой порог опыта выше, цель - порог


def _f(x, digits=3):
    return '—' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.{digits}f}'


def _table(head, rows):
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


def tables(results: dict) -> dict:
    """Markdown-таблицы отчета по результатам модулей: {ключ подстановки: таблица}."""
    import he2020
    import pore_models
    import li2024
    import sandyga2020
    import sutton_roberts
    out = {}
    if 'sutton_roberts' in results:
        rows = sutton_roberts.summary(results['sutton_roberts'])
        out['SR'] = _table(['Вариант', 'СКО k/k₀, опыт 1', 'СКО k/k₀, опыт 2', 'Подобрано по'],
                           [(n, _f(a), _f(b), c) for n, a, b, c in rows])
    if 'li2024' in results:
        o = results['li2024']
        rows = li2024.summary(o)
        # nan у ступени - прогон до нее не дошел: керн закупорен на предыдущей
        plug = lambda v: 'закупорка' if isinstance(v, float) and math.isnan(v) else _f(v)
        out['LI'] = _table(['Вариант', '90 °C', '65 °C', '45 °C', '25 °C'],
                           [(n, *([_f(v) for v in vals] if i == 0 else [plug(v) for v in vals]))
                            for i, (n, vals) in enumerate(rows)])
        calib = lambda rms: math.sqrt(sum(rms[t] ** 2 for t in ('90.0', '65.0', '45.0')) / 3.0)
        out['LI_SOLID'] = _f(calib(o['forms']['solid']['rms']))
        out['LI_FILM'] = _f(calib(o['forms']['film']['rms']))
        out['LI_DH'] = f"{-o['forms'][o['best_form']]['kin']['ADS_DH'] / 1e3:.0f}"
        if 'cold' in o:
            c = o['cold']
            r25 = c['rms']['25.0']
            out['LI_HOLD_25'] = f'СКО ступени 25 °C {_f(r25)}'
            k = c['kin']
            out['LI_COLD_TEXT'] = (
                f"семейство подобрано отдельно при удержании из подбора выше WAT: тиксотропное время геля "
                f"{c['gel_time']:g} с, диаметр кристалла {k['D_CRYST'] * 1e6:.1f} мкм (на Berea - 15 мкм, у керна Li "
                f"поры мельче), множитель броуновской диффузии {k['DIFF_MULT']:.2f}, константа кристаллизации в "
                f"объеме {k['K_CRYST']:.1e} 1/с. СКО ступени {_f(r25)} при шумовом пороге {_f(o['noise']['25'])}.")
    if 'sandyga2020' in results:
        rows = sandyga2020.summary(results['sandyga2020'])
        out['SD'] = _table(['Вариант', 'СКО lg(∇p/∇p₀)', 'СКО k/k₀', 'наибольшее расхождение в 5 точках, раз',
                            'проводящая пористость, доли m₀'],
                           [(n, _f(a), _f(k), _f(b, 1), _f(c, 2)) for n, a, k, b, c in rows])
        best = results['sandyga2020']['best']
        if 'cooling' in results['sandyga2020']:
            rates = sorted(results['sandyga2020']['cooling'].values(), key=lambda e: e['rate'])
            out['SD_RATES'] = _table(['Скорость охлаждения, °C/ч', 'Da на 1 °C', 'рост в 2 раза, °C', 'рост в 10 раз, °C',
                                      'рост к 32.8 °C, раз'],
                                     [(f"{e['rate']:g}", _f(e['damkohler'], 2), _f(e['t2'], 2), _f(e['t10'], 2),
                                       f"{e['final']:.0f}") for e in rates])
        out['SD_PORES'] = _table(['Диаметр пор, мкм'] + [str(d) for d in sandyga2020.DATA['pores']['d_um']],
                                 [['томография'] + [_f(v, 2) for v in best['pore_loss_exp']],
                                  ['модель'] + [_f(v, 2) for v in best['pore_loss']]])
    if 'he2020' in results:
        rows = he2020.summary(results['he2020'])
        out['HE'] = _table(['Зависимость k(m)', 'СКО lg(k/k₀)', 'СКО k/k₀', 'точек вне досягаемости'],
                           [(n, _f(e), _f(l), f'{m} из {tot}') for n, e, l, m, tot in rows])
    if results:
        out['SUMMARY'] = summary_table(results)
    if 'pore_models' in results:
        rows = pore_models.summary(results['pore_models'])
        out['PORE'] = _table(['Модель порового пространства', 'СКО k/k₀', 'СКО lg(k/k₀)', 'подобрано параметров',
                              'примечание'], [(n, _f(a), _f(b), c, d) for n, a, b, c, d in rows])
        out['EMA_LATTICE'] = _f(results['pore_models']['ema_vs_lattice'])
    return out


def summary_table(results: dict) -> str:
    """Сводка по всем опытам в одной шкале - СКО k/k0 (цель 0.05 или шумовой порог опыта, если он выше)."""
    g = lambda d, n: d[n] if n in d else d[str(n)]
    rows = []
    if 'sutton_roberts' in results:
        import sutton_roberts as sr
        o = results['sutton_roberts']
        for n in (1, 2):
            own = g(o['lsq']['rate_entrainment'][f'own{n}']['rms'], n)
            shared = g(o['visc']['rms'], n) if 'visc' in o else None
            rows.append((f'Sutton & Roberts, опыт {n}', _f(g(o['noise'], n)), _f(sr.rms(g(o['legacy'], n), n)),
                         _f(own), _f(shared), _f(g(o['others']['Wang & Civan (2005)'], n))))
    if 'li2024' in results:
        import li2024 as li
        o = results['li2024']
        legacy = li.stage_rms(o['legacy'])
        # лучшая модель - удержание выше WAT и кинетика кристаллизации ниже (подбор 25 C) в одном прогоне
        best = o['cold']['rms'] if 'cold' in o else o['forms'][o['best_form']]['rms']
        get = lambda d, t: d[t] if t in d else d.get(float(t))
        for t in ('90.0', '65.0', '45.0', '25.0'):
            old_v = legacy.get(float(t))
            rows.append((f'Li et al., {float(t):.0f} °C', _f(o['noise'][str(int(float(t)))]),
                         _f(old_v) if old_v is not None else 'закупорка', _f(get(best, t)), '—', '—'))
    if 'sandyga2020' in results:
        o = results['sandyga2020']
        rows.append(('Sandyga et al., ∇p(T)', '—', _f(o['legacy'].get('rms_k')), _f(o['best'].get('rms_k')), '—', '—'))
    if 'pore_models' in results:
        o = {r['name']: r for r in results['pore_models']['rows']}
        net = [r for name, r in o.items() if name.startswith('4. сеть') and 'подбор' in r['note']]
        bundle = [r for name, r in o.items() if name.startswith('1.')]
        power = [r for name, r in o.items() if name.startswith('6.')]
        rows.append(('He et al., k(m)', '—', _f(bundle[0]['lin']) if bundle else '—', _f(net[0]['lin']) if net else '—',
                     '—', _f(power[0]['lin']) if power else '—'))
    return _table(['Опыт', 'шумовой порог', 'прежняя модель', 'лучшая модель (калибровка)',
                   'общая кинетика, своя вязкость', 'другие авторы / замыкание'], rows)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    mode = mode_from_argv()
    names = [a for a in sys.argv[1:] if not a.startswith('--')] or list(ORDER)
    results = {}
    for name in ORDER:
        if name not in names:
            path = RESULTS / f'{name}.json'
            if path.is_file():
                results[name] = json.loads(path.read_text(encoding='utf-8'))
            continue
        module = __import__(name)
        t0 = time.time()
        print(f'=== {name} ({mode}) ===', flush=True)
        results[name] = json.loads(json.dumps(module.run(mode), default=float))  # ключи - строки, как после чтения
        print(f'=== {name}: {time.time() - t0:.0f} с ===', flush=True)
    fresh = {}
    for name, res in results.items():  # results/*.json прежнего формата - без таблиц, с подсказкой
        try:
            tables({name: res})
            fresh[name] = res
        except (KeyError, TypeError) as err:
            print(f'{name}: в results нет поля {err} - таблицы пропущены, пересчитайте модуль без --plot')
    out = tables(fresh)
    path = RESULTS / 'tables.json'
    old = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    old.update(out)
    path.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding='utf-8')
    for key, table in out.items():
        print(f'\n{key}\n{table}')


if __name__ == '__main__':
    main()
