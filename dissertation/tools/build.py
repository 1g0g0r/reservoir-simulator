"""Сборка literature_review.md из sections/*.md.

    python build.py [--check]

Цитаты в разделах - [@id] или [@id1; @id2] (id - DOI в нижнем регистре или m:...). Номера [n]
ставятся по первому упоминанию, список литературы - по ГОСТ Р 7.0.5-2008 (gost() из build_registry).
Разделы 9x_*.md - приложения: 90 - глоссарий, 93 - карта связей; приложения A (таблица всех источников)
и C (хронология) строятся здесь. --check только проверяет id и печатает статистику.
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_registry import SUBTOPICS, gost, load, record  # noqa: E402

REVIEW = Path(__file__).resolve().parents[1]
CITE = re.compile(r'\[@([^\]]+)\]')


def words(text: str) -> int:
    t = re.sub(r'\$[^$]*\$', ' ', text)                     # формулы не считаем
    t = re.sub(r'^\|.*\|$', ' ', t, flags=re.M)               # таблицы считаем отдельно
    return len(re.findall(r'[A-Za-zА-Яа-яЁё]{2,}', t))


def main():
    meta, manual, *_ = load()
    with (REVIEW / 'sources_registry.csv').open(encoding='utf-8-sig') as fh:
        reg = {r['id']: r for r in csv.DictReader(fh)}
    secs = sorted((REVIEW / 'sections').glob('*.md'))
    body = [p for p in secs if not p.name.startswith('9')]
    apps = [p for p in secs if p.name.startswith('9')]
    order, bad = [], []
    texts = []
    for p in body + apps:
        t = p.read_text(encoding='utf-8')
        for m in CITE.finditer(t):
            for i in [x.strip().lstrip('@').strip() for x in m.group(1).split(';')]:
                if i not in reg:
                    bad.append((p.name, i))
                elif i not in order:
                    order.append(i)
        texts.append((p, t))
    num = {i: n + 1 for n, i in enumerate(order)}
    unverified = [i for i in order if reg[i]['status'] == 'unverified']
    main_words = sum(words(t) for p, t in texts if not p.name.startswith('9'))
    table_words = sum(len(re.findall(r'[A-Za-zА-Яа-яЁё]{2,}', ' '.join(re.findall(r'^\|.*\|$', t, flags=re.M))))
                      for p, t in texts if not p.name.startswith('9'))
    print(f'разделов {len(body)}, приложений {len(apps)}, источников процитировано {len(order)} из {len(reg)}, '
          f'слов текста {main_words} (+{table_words} в таблицах)')
    if bad:
        print('НЕИЗВЕСТНЫЕ id:', bad)
    if unverified:
        print('цитируются непроверенные:', unverified)
    if '--check' in sys.argv:
        return

    def repl(m):
        ns = sorted({num[x.strip().lstrip('@').strip()] for x in m.group(1).split(';')})
        return '[' + ', '.join(map(str, ns)) + ']'

    out = []
    for p, t in texts:
        if p.name.startswith('9'):
            continue
        out.append(CITE.sub(repl, t).rstrip() + '\n')
    out.append('# Список литературы\n\n*Оформление — ГОСТ Р 7.0.5–2008; нумерация по порядку первого упоминания; '
               'BibTeX — `references.bib`.*\n')
    out += [f'{num[i]}. {gost(record(i, meta, manual))}' for i in order]
    # приложения
    out.append('\n# Приложение A. Таблица всех источников реестра\n')
    out.append(f'*{len(reg)} записей `sources_registry.csv`; «№» — номер в списке литературы, если работа цитируется '
               'в тексте. Полные метаданные, аннотации и статусы — в реестре и `sources_annotated.md`.*\n')
    out.append('| Ключ | № | Год | Первый автор | Название | Подтемы | Релевантность | Доступ |')
    out.append('|---|---|---|---|---|---|---|---|')
    for r in sorted(reg.values(), key=lambda r: (str(r['year']), r['key'])):
        out.append(f'| {r["key"]} | {num.get(r["id"], "")} | {r["year"]} | {r["authors"].split(";")[0]} | '
                   f'{r["title"][:70].replace("|", "/")} | {r["subtopics"]} | {r["relevance"]} | {r["status"]} |')
    for p, t in texts:
        if p.name.startswith('90'):
            out.append('\n' + CITE.sub(repl, t).rstrip() + '\n')
    out.append('\n# Приложение C. Хронологическая шкала ключевых работ\n')
    out.append('*Работы высокой релевантности, процитированные в тексте, по годам (из реестра).*\n')
    key = [i for i in order if reg[i]['relevance'] == 'высокая']
    for r in sorted((reg[i] for i in key), key=lambda r: int(r['year'] or 0)):
        out.append(f'- **{r["year"]}** — {r["authors"].split(";")[0]}: {r["contribution"]} [{num[r["id"]]}].')
    for p, t in texts:
        if p.name.startswith('93'):
            out.append('\n' + CITE.sub(repl, t).rstrip() + '\n')
    (REVIEW / 'literature_review.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
    (REVIEW / 'data' / 'citations.json').write_text(json.dumps({'order': order, 'bad': bad, 'words': main_words,
                                                               'table_words': table_words}, ensure_ascii=False),
                                                    encoding='utf-8')
    print('literature_review.md записан')


if __name__ == '__main__':
    main()
