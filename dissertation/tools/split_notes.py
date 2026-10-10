"""Конспекты: data/notes_src.md (блоки '=== id') -> notes/<ключ>.md с шапкой из реестра и notes/README.md.

    python split_notes.py
"""
import csv
import re
from pathlib import Path

REVIEW = Path(__file__).resolve().parents[1]


def main():
    with (REVIEW / 'sources_registry.csv').open(encoding='utf-8-sig') as fh:
        reg = {r['id']: r for r in csv.DictReader(fh)}
    from build_registry import gost, record, load  # noqa: E402  (тот же формат ссылки, что в обзоре)
    meta, manual, *_ = load()
    blocks = re.split(r'^=== (\S+)\s*$', (REVIEW / 'data' / 'notes_src.md').read_text(encoding='utf-8'), flags=re.M)
    out_dir = REVIEW / 'notes'
    out_dir.mkdir(exist_ok=True)
    index = ['# Конспекты ключевых источников', '',
             'Составлены по полным текстам из библиотеки books (туда 10.10.2026 перенесены PDF из `experiments/new` и скачанные открытые копии). '
             'Структура: постановка, метод, результаты, ограничения, связь с проектом `paraphin`.', '',
             '| Ключ | Работа | Полный текст |', '|---|---|---|']
    for rid, body in zip(blocks[1::2], blocks[2::2]):
        r = reg[rid]
        head = [f'# {r["key"]}', '', f'**Источник.** {gost(record(rid, meta, manual))}', '',
                f'**Полный текст.** `{r["file_path"] or "—"}` ({r["pages_pdf"] or "?"} с.); подтемы: {r["subtopics"]}; '
                f'релевантность: {r["relevance"]}.', '']
        (out_dir / f'{r["key"]}.md').write_text('\n'.join(head) + body.strip().replace('**', '\n**', 0) + '\n',
                                                 encoding='utf-8')
        index.append(f'| [{r["key"]}]({r["key"]}.md) | {r["authors"].split(";")[0]} ({r["year"]}). {r["title"][:80]} | '
                     f'{r["status"]} |')
    (out_dir / 'README.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    print(len(blocks[1::2]), 'конспектов')


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
