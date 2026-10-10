"""Поиск свежих работ по подтемам -> data/recent_candidates.csv для ручного отбора.

    python recent.py [N_на_запрос] [--topic] [--crossref | --s2]

По умолчанию OpenAlex и запросы QUERIES (работы с 2022 г.); --topic - запросы QUERIES_TOPIC по теме диссертации
(специальность 1.1.9, с 2015 г., в том числе русскоязычные). У OpenAlex без ключа суточный бюджет на IP: 1000
кредитов, поиск стоит 10, запрос по фильтру - 1 (10.10.2026 бюджет кончился на середине работы). Запасные пути:
--crossref (релевантность хуже) и --s2 (Semantic Scholar, без ключа ~1 запрос в секунду и часто 429). Отобранные
вручную DOI идут в data/seeds*.csv и дальше по обычному пути (verify.py, curate.py). Уже учтенные пропускаются.
"""
import csv
import sys
import time

from verify import DATA, _get, from_openalex

S2 = 'https://api.semanticscholar.org/graph/v1/paper/'
FIELDS = 'title,year,externalIds,venue,citationCount'


def s2(url, **params):
    """Запрос к S2 с паузой; список статей или []."""
    time.sleep(3)
    r = _get(url, fields=FIELDS, **params)
    return r.json().get('data', []) if r is not None and r.ok else []


def row(p):
    """Статья S2 -> (doi, year, cited_by, venue, title); doi пустой, если его нет."""
    return ((p.get('externalIds') or {}).get('DOI', '').lower(), p.get('year') or '', p.get('citationCount') or 0,
            (p.get('venue') or '')[:40], (p.get('title') or '')[:140])

QUERIES = {
    'T01': ['cold water injection wax deposition reservoir', 'non-isothermal waterflooding waxy crude oil reservoir',
            'reservoir cooling formation damage paraffin'],
    'T03': ['wax appearance temperature thermodynamic model crude oil', 'wax precipitation pressure dissolved gas live oil'],
    'T04': ['asphaltene precipitation PC-SAFT modeling', 'asphaltene onset pressure prediction',
            'asphaltene precipitation CO2 injection'],
    'T06': ['wax crystallization kinetics crude oil', 'asphaltene aggregation kinetics'],
    'T07': ['asphaltene deposition porous media permeability impairment', 'fines migration detachment porous media model',
            'colloid deposition erosion porous media'],
    'T08': ['pore network model clogging permeability', 'pore-scale simulation deposition permeability evolution'],
    'T09': ['waxy crude oil gel yield stress thixotropy', 'yield stress fluid flow porous media'],
    'T10': ['wax deposition reservoir simulation', 'asphaltene deposition near-wellbore simulation',
            'organic deposition formation damage modeling'],
    'T11': ['wax deposition core flooding experiment', 'microfluidic asphaltene deposition',
            'high pour point oil reservoir experiment'],
    'T12': ['multigrid preconditioner reservoir simulation pressure', 'sequential implicit reservoir simulation',
            'compositional reservoir simulation solid precipitation'],
    'T13': ['open-source reservoir simulator', 'reservoir simulation benchmark comparison'],
    'T14': ['wax deposition pipeline model', 'asphaltene deposition wellbore model'],
    'T15': ['machine learning wax appearance temperature', 'machine learning asphaltene precipitation',
            'surrogate model reservoir simulation neural operator', 'physics-informed neural network porous media flow'],
}


# Тема диссертации (1.1.9): фильтрация с фазовыми переходами, кольматация, нелинейные законы фильтрации
QUERIES_TOPIC = {
    'T10': ['non-isothermal filtration phase transition paraffin porous medium',
            'mathematical model paraffin deposition porous medium permeability temperature',
            'heavy oil components deposition filtration porous medium model',
            'asphaltene deposition non-isothermal reservoir model'],
    'T07': ['colmatation porous medium mathematical model particle deposition',
            'suspension filtration porous medium deposition kinetics model',
            'pore size distribution particle deposition two-phase filtration'],
    'T01': ['non-isothermal two-phase filtration phase transitions reservoir',
            'heat and mass transfer porous medium crystallization filtration'],
    'T09': ['threshold pressure gradient waxy oil filtration temperature',
            'non-Newtonian oil filtration yield stress reservoir'],
    'RU': ['кольматация пористой среды математическая модель', 'неизотермическая фильтрация парафинистой нефти',
           'фильтрация с фазовыми переходами парафин', 'отложение асфальтенов в пористой среде моделирование',
           'асфальтосмолопарафиновые отложения пласт'],
}


def openalex(q, n, since='2022-01-01'):
    """Поиск OpenAlex (10 кредитов), строки как у crossref()."""
    r = _get('https://api.openalex.org/works', search=q, per_page=n,
             filter=f'from_publication_date:{since},type:article|review|dissertation,has_doi:true')
    recs = [from_openalex(w) for w in (r.json().get('results', []) if r is not None and r.ok else [])]
    return [(x['doi'], x['year'], x['cited_by'], (x['venue'] or '')[:40], (x['title'] or '')[:140]) for x in recs]


def crossref(q, n):
    """Запасной поиск Crossref (статьи с 2022 г.), те же строки, что row()."""
    r = _get('https://api.crossref.org/works', **{'query.bibliographic': q, 'rows': n,
             'filter': 'from-pub-date:2022-01-01,type:journal-article',
             'select': 'DOI,title,issued,container-title,is-referenced-by-count'})
    items = r.json()['message']['items'] if r is not None and r.ok else []
    return [(m['DOI'].lower(), m['issued']['date-parts'][0][0], m.get('is-referenced-by-count', 0),
             (m.get('container-title') or [''])[0][:40], (m.get('title') or [''])[0][:140]) for m in items]


def semantic(q, n):
    return [row(p) for p in s2(S2 + 'search', query=q, year='2022-', limit=n)]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
    topic = '--topic' in sys.argv
    search = (crossref if '--crossref' in sys.argv else semantic if '--s2' in sys.argv
              else (lambda q, k: openalex(q, k, '2015-01-01')) if topic else openalex)
    with (DATA / 'curation.csv').open(encoding='utf-8') as fh:
        known = {r['id'] for r in csv.DictReader(fh)}
    out, seen = [], set()
    for sub, qs in (QUERIES_TOPIC if topic else QUERIES).items():
        for q in qs:
            for doi, *rest in search(q, n):
                if doi and doi not in known and doi not in seen:
                    seen.add(doi)
                    out.append([sub, q, doi, *rest])
    name = 'topic_candidates.csv' if topic else 'recent_candidates.csv'
    with (DATA / name).open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['subtopic', 'q', 'doi', 'year', 'cited_by', 'venue', 'title'])
        w.writerows(out)
    print(len(out))


if __name__ == '__main__':
    main()
