"""Контроль качества обзора и реестра -> data/qa.json и сводка в консоль (отчет - 99_qa_report.md, вручную).

    python qa.py              # проверки по файлам
    python qa.py --sample 20  # плюс перепроверка случайных записей по Crossref и по URL

Проверяется: номера [n] в тексте против списка литературы (каждый номер есть в списке, каждая запись списка
цитируется, нумерация по первому упоминанию); дубли DOI и названий; DOI или URL у каждой записи; уникальность
ключей BibTeX и полнота references.bib; покрытие подтем и свежие работы; ссылки [@id] в chapters/.
"""
import csv
import difflib
import json
import random
import re
import sys
from collections import Counter

from verify import DATA, _get

REVIEW = DATA.parent
SUBTOPICS = [f'T{i:02d}' for i in range(1, 17)] + ['RU']


def numbers(group):
    """'3, 5–7' -> [3, 5, 6, 7]."""
    out = []
    for part in group.split(','):
        a, _, b = part.strip().replace('-', '–').partition('–')
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def norm(s):
    return re.sub(r'[^a-zа-яё0-9]+', ' ', s.lower()).strip()


def main():
    reg = list(csv.DictReader((REVIEW / 'sources_registry.csv').open(encoding='utf-8-sig')))
    text = (REVIEW / 'literature_review.md').read_text(encoding='utf-8')
    body, rest = text.split('\n# Список литературы\n', 1)
    refs, apps = rest.split('\n# Приложение A', 1)
    body = re.sub(r'\$\$.*?\$\$|\$[^$\n]*\$', ' ', body, flags=re.S)          # формулы не ссылки
    order = []
    for g in re.findall(r'\[(\d[\d,\s–-]*)\]', body):
        order += [n for n in numbers(g) if n not in order]
    listed = [int(n) for n in re.findall(r'^(\d+)\. ', refs, flags=re.M)]
    num2key = {int(n): k for k, n in re.findall(r'^\| (\S+) \| (\d+) \|', apps, flags=re.M)}
    by_key = {r['key']: r for r in reg}
    q = {
        'cited_numbers': len(order),
        'listed': len(listed),
        'list_sequential': listed == list(range(1, len(listed) + 1)),
        'cited_not_listed': sorted(set(order) - set(listed)),
        'listed_not_cited': sorted(set(listed) - set(order)),
        'first_mention_order': order == sorted(order),
        'cited_unverified': [num2key[n] for n in order if by_key.get(num2key.get(n), {}).get('status') == 'unverified'],
    }
    dois = Counter(r['doi'].lower() for r in reg if r['doi'])
    titles = Counter((norm(r['title']), r['year']) for r in reg)
    q['dup_doi'] = [d for d, c in dois.items() if c > 1]
    q['dup_title'] = [t for t, c in titles.items() if c > 1]
    q['no_doi_no_url'] = [(r['key'], r['status'], r['verified_by'][:60]) for r in reg if not r['doi'] and not r['url']]
    bib = (REVIEW / 'references.bib').read_text(encoding='utf-8')
    bkeys = re.findall(r'^@\w+\{([^,]+),', bib, flags=re.M)
    verified = [r for r in reg if r['status'] != 'unverified']
    q['bib_entries'] = len(bkeys)
    q['bib_unique'] = len(bkeys) == len(set(bkeys))
    q['bib_missing'] = sorted({r['key'] for r in verified} - set(bkeys))
    cov = {}
    for s in SUBTOPICS:
        rs = [r for r in verified if s in r['subtopics'].split(';')]
        cov[s] = {'all': len(rs), '2023+': sum(1 for r in rs if r['year'] and int(r['year']) >= 2023),
                  '2020+': sum(1 for r in rs if r['year'] and int(r['year']) >= 2020)}
    q['coverage'] = cov
    q['coverage_below_10'] = [s for s, c in cov.items() if c['all'] < 10]
    q['no_recent_2023'] = [s for s, c in cov.items() if c['2023+'] == 0]
    years = [int(r['year']) for r in verified if r['year']]
    q['stats'] = {
        'records': len(reg), 'verified': len(verified), 'status': Counter(r['status'] for r in reg),
        'type': Counter(r['type'] for r in reg), 'language': Counter(r['language'] or '?' for r in reg),
        'decades': Counter(f'{y // 10 * 10}-е' for y in years), 'median_year': sorted(years)[len(years) // 2],
        'share_2020plus': round(sum(y >= 2020 for y in years) / len(years), 3),
    }
    ids = {r['id'] for r in reg}
    q['chapters_unknown_ids'] = {}
    for p in sorted((REVIEW / 'chapters').glob('*.md')):
        used = {x.strip().lstrip('@') for m in re.findall(r'\[@([^\]]+)\]', p.read_text(encoding='utf-8'))
                for x in m.split(';')} - {'id'}
        if used - ids:
            q['chapters_unknown_ids'][p.name] = sorted(used - ids)
    if '--sample' in sys.argv:
        n = int(sys.argv[sys.argv.index('--sample') + 1])
        rnd = random.Random(20261010)
        pool = [r for r in verified if r['doi']]
        sample = []
        for r in rnd.sample(pool, n):
            x = _get('https://api.crossref.org/works/' + r['doi'])
            m = x.json()['message'] if x is not None and x.ok else {}
            t = (m.get('title') or [''])[0]
            ys = {p['date-parts'][0][0] for k in ('published-print', 'published-online', 'issued')
                  if (p := m.get(k)) and p.get('date-parts') and p['date-parts'][0][0]}
            sim = difflib.SequenceMatcher(None, norm(t), norm(r['title'])).ratio()
            yr = {int(v) for v in (r['year'], r['year_online']) if v}
            sample.append({'key': r['key'], 'doi': r['doi'], 'found': bool(m), 'title_similarity': round(sim, 2),
                           'year_ok': bool(yr & ys), 'crossref_years': sorted(ys), 'registry_year': r['year']})
        urls = [r for r in verified if not r['doi'] and r['url'].startswith('http')]
        for r in rnd.sample(urls, min(5, len(urls))):
            x = _get(r['url'])
            sample.append({'key': r['key'], 'url': r['url'], 'http': x.status_code if x is not None else None})
        q['sample'] = sample
    (DATA / 'qa.json').write_text(json.dumps(q, ensure_ascii=False, indent=1, default=dict), encoding='utf-8')
    for k, v in q.items():
        if k not in ('coverage', 'sample', 'stats'):
            print(f'{k}: {v}')
    print('coverage:', {s: c['all'] for s, c in cov.items()})
    print('2023+:', {s: c['2023+'] for s, c in cov.items()})
    print('stats:', {k: v for k, v in q['stats'].items() if k in ('records', 'verified', 'median_year', 'share_2020plus')})
    for s in q.get('sample', []):
        print(s)


if __name__ == '__main__':
    main()
