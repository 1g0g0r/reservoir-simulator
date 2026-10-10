"""Сбор DOI из всех списков литературы проекта с указанием, где каждый DOI встретился.

    python dissertation/tools/harvest.py [папка_с_распакованными_архивами]

Результат - review/data/harvest.csv (doi, файлы-источники). Файлы проекта только читаются.
Распакованные very_big_review.zip и artifacts.zip (resources/claude_reserch) передаются аргументом:
их .bib и .md тоже сканируются, корпус corpus_v2.csv - нет (это пул для поиска, а не цитаты).
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # reservoir-simulator
REVIEW = Path(__file__).resolve().parents[1]
BOOKS = Path.home() / 'books' / 'neft'

DOI_RE = re.compile(r'10\.\d{4,9}/[^\s"\'<>|,;\]}`*]+', re.I)

FILES = [
    'experiments/new/README.md', 'experiments/new/обзор_источников.md',
    'experiments/README.md', 'experiments/сравнение_с_опытами.md',
    'resources/литература/ГДЕ_ФАЙЛЫ.md', 'TODO_list.txt',
]
GLOBS = ['docs/*.md', 'resources/литература/*/README.md', 'resources/claude_reserch/*.md',
         'experiments/*/*.md', 'твт_статья_АСПО/*.md']
BOOK_READMES = ['Парафинистые нефти/README.md', 'Численное моделирование/README.md',
                'Подземная гидромеханика и фильтрация/README.md', 'Физика пласта и PVT/README.md']


def clean(doi: str) -> str:
    doi = doi.rstrip('.').rstrip(':')
    while doi.endswith(')') and doi.count(')') > doi.count('('):   # скобка текста, не DOI
        doi = doi[:-1].rstrip('.')
    # pdf-хвосты и якоря из ссылок
    doi = re.sub(r'(/pdf|/full|/abstract|\.pdf|#.*)$', '', doi, flags=re.I)
    return doi.lower()


def scan(path: Path, label: str, found: dict) -> None:
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return
    for m in DOI_RE.finditer(text):
        found.setdefault(clean(m.group(0)), set()).add(label)


def main() -> None:
    found: dict[str, set] = {}
    for f in FILES:
        scan(ROOT / f, f, found)
    for g in GLOBS:
        for p in sorted(ROOT.glob(g)):
            scan(p, p.relative_to(ROOT).as_posix(), found)
    for b in BOOK_READMES:
        scan(BOOKS / b, 'books/neft/' + b, found)
    if len(sys.argv) > 1:
        extra = Path(sys.argv[1])
        for p in sorted(list(extra.rglob('*.bib')) + list(extra.rglob('*.md')) + list(extra.rglob('*_papers*.csv'))):
            scan(p, 'archive:' + p.name, found)

    out = REVIEW / 'data' / 'harvest.csv'
    out.parent.mkdir(exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['doi', 'n_files', 'files'])
        for doi in sorted(found):
            w.writerow([doi, len(found[doi]), ' | '.join(sorted(found[doi]))])
    print(f'DOI: {len(found)} -> {out}')


if __name__ == '__main__':
    main()
