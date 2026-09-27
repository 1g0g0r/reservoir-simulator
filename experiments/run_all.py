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

ORDER = ('sutton_roberts', 'li2024', 'sandyga2020', 'he2020')
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
    import li2024
    import sandyga2020
    import sutton_roberts
    out = {}
    if 'sutton_roberts' in results:
        rows = sutton_roberts.summary(results['sutton_roberts'])
        out['SR'] = _table(['Вариант', 'СКО k/k₀, опыт 1', 'СКО k/k₀, опыт 2', 'Подобрано по'],
                           [(n, _f(a), _f(b), c) for n, a, b, c in rows])
    if 'li2024' in results:
        rows = li2024.summary(results['li2024'])
        out['LI'] = _table(['Вариант', '90 °C', '65 °C', '45 °C', '25 °C'],
                           [(n, *[_f(v) for v in vals]) for n, vals in rows])
    if 'sandyga2020' in results:
        rows = sandyga2020.summary(results['sandyga2020'])
        out['SD'] = _table(['Вариант', 'СКО lg(∇p/∇p₀)', 'наибольшее расхождение в 5 точках, раз',
                            'проводящая пористость, доли m₀'],
                           [(n, _f(a), _f(b, 1), _f(c, 2)) for n, a, b, c in rows])
        best = results['sandyga2020']['best']
        out['SD_PORES'] = _table(['Диаметр пор, мкм'] + [str(d) for d in sandyga2020.DATA['pores']['d_um']],
                                 [['томография'] + [_f(v, 2) for v in best['pore_loss_exp']],
                                  ['модель'] + [_f(v, 2) for v in best['pore_loss']]])
    if 'he2020' in results:
        rows = he2020.summary(results['he2020'])
        out['HE'] = _table(['Зависимость k(m)', 'СКО lg(k/k₀)', 'точек вне досягаемости'],
                           [(n, _f(e), f'{m} из {tot}') for n, e, m, tot in rows])
    return out


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
    try:
        out = tables(results)
    except KeyError as err:  # results/*.json прежнего формата: пересчитать модуль
        raise SystemExit(f'в results нет поля {err}: пересчитайте модуль без --plot')
    path = RESULTS / 'tables.json'
    old = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    old.update(out)
    path.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding='utf-8')
    for key, table in out.items():
        print(f'\n{key}\n{table}')


if __name__ == '__main__':
    main()
