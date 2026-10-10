"""Отбор источников: подтема, релевантность, включение - по правилам и ручным правкам.

    python curate.py   -> data/curation.csv (id, subtopics, relevance, include, reason)

Кандидаты - DOI из data/harvest.csv, data/seeds*.csv (проверены verify.py) и записи без DOI из
data/manual.csv. Подтема по ключевым словам названия, затем ручные правки OVERRIDE; исключения
EXCLUDE - с причиной (идут в список «исключены, потому что…» обзора).
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data'

RULES = [  # (подтема, регулярное выражение по названию) - первое совпадение
    ('T15', r'machine learning|neural|intelligent|artificial|ensemble|physics-informed|deeponet|fourier neural|surrogate|data-driven|committee machine|deep-learning|gradient boosting'),
    ('T12', r'multigrid|conjugate gradient|preconditioning|iterative method|impes|finite volume|well-block|linear solver|multipoint flux|projection technique|differenzengleichungen|cpr-type|constrained residual|incomplete gaussian'),
    ('T13', r'simulator|toolbox|matlab|numba|numpy|scipy|porepy|dumux|open porous media|comparative solution|verification and validation|geos:'),
    ('T05', r'equation of state|flash|stability|rachford|volume|redlich'),
    ('T09', r'rheolog|viscosit|yield stress|gel|bingham|non-newtonian|thixotrop|restart|yielding'),
    ('T08', r'kozeny|network|percolation|pore-scale|pore scale|capillary pressures|porosity.permeability|lognormal|porosity reviewed|bundle'),
    ('T07', r'filtration|fines|particle|suspension|colloid|adsorption|diffusivit|diffusion coefficient|entrainment|detachment|brownian|bewegung|shear flow of a suspension'),
    ('T06', r'kinetic|crystalliz|aging|ageing|cooling rate|cloud point'),
    ('T14', r'pipeline|wellbore|tubing|flow loop|cold finger|well string|wax deposition'),
    ('T04', r'asphaltene|resin'),
    ('T03', r'wax|paraffin|n-alkane|solid.liquid'),
    ('T02', r'fraction|characteriz|lumped|heavy hydrocarbons|natural gases|petroleum mixtures'),
    ('T01', r'heat|thermal|temperature|displacement|convection|hot fluid'),
]


def _load_overrides():
    """data/overrides.csv: id, subtopics (ручная классификация)."""
    with (DATA / 'overrides.csv').open(encoding='utf-8') as fh:
        return {r['id'].strip().lower(): r['subtopics'].strip() for r in csv.DictReader(fh)}


def _load_list(name):
    p = DATA / name
    return {l.strip().lower() for l in p.read_text(encoding='utf-8').split() if l.strip()} if p.exists() else set()


def _load_excludes():
    with (DATA / 'excludes.csv').open(encoding='utf-8') as fh:
        return {r['id'].strip().lower(): r['reason'] for r in csv.DictReader(fh)}


def subtopic(title: str) -> str:
    t = title.lower()
    for code, rx in RULES:
        if re.search(rx, t):
            return code
    return 'T16'


def main():
    meta = json.loads((DATA / 'meta.json').read_text(encoding='utf-8'))
    over, high, low, excl = _load_overrides(), _load_list('high.txt'), _load_list('low.txt'), _load_excludes()
    dois = []
    for f in ['harvest.csv', 'seeds.csv', 'seeds2.csv', 'seeds3.csv', 'seeds4.csv', 'seeds5.csv']:
        with (DATA / f).open(encoding='utf-8') as fh:
            dois += [r['doi'].strip().lower() for r in csv.DictReader(fh)]
    rows = []
    for d in dict.fromkeys(dois):
        m = meta.get(d, {})
        if d in excl or m.get('source') == 'not_found':
            rows.append([d, '', '', 0, excl.get(d, 'DOI не найден ни в OpenAlex, ни в Crossref')])
            continue
        st = over.get(d) or subtopic(m.get('title', ''))
        rel = 'высокая' if d in high else ('низкая' if d in low else 'средняя')
        rows.append([d, st, rel, 1, ''])
    with (DATA / 'manual.csv').open(encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            rows.append([r['id'], r['subtopics'], r['relevance'], int(r['include']), r.get('reason', '')])
    with (DATA / 'curation.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['id', 'subtopics', 'relevance', 'include', 'reason'])
        w.writerows(rows)
    inc = [r for r in rows if r[3]]
    c = Counter(s for r in inc for s in r[1].split(';'))
    print(f'всего {len(rows)}, включено {len(inc)}, исключено {len(rows) - len(inc)}')
    print(sorted(c.items()))
    print('без ручной подтемы:', [r[0] for r in inc if r[0] not in over and not r[0].startswith('m:')])


if __name__ == '__main__':
    main()
