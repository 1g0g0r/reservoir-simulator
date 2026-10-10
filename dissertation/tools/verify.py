"""Проверка DOI по OpenAlex (пакетами по 50) с запасным Crossref; кэш - review/data/meta.json.

    python verify.py harvest.csv [seeds.csv ...]     # проверить DOI из файлов (колонка doi)
    python verify.py --title "Название" [год]        # найти DOI по названию (OpenAlex search)

Почта в запросы не передается (OpenAlex и Crossref работают без нее). Уже проверенные DOI берутся
из кэша; запись кэша - словарь полей OpenAlex, сведенный к нужному реестру.
"""
import csv
import json
import sys
import time
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parents[1] / 'data'
CACHE = DATA / 'meta.json'
S = requests.Session()
S.headers['User-Agent'] = 'reservoir-simulator literature review (open access only)'


def _get(url, **params):
    for attempt in range(5):
        try:
            r = S.get(url, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt + 1)
            continue
        return r
    return None


def _abstract(inv):
    if not inv:
        return ''
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    return ' '.join(w for _, w in sorted(pos))


def from_openalex(w: dict) -> dict:
    loc = w.get('primary_location') or {}
    src = loc.get('source') or {}
    bib = w.get('biblio') or {}
    oa = w.get('open_access') or {}
    best = w.get('best_oa_location') or {}
    pdfs = [l.get('pdf_url') for l in (w.get('locations') or []) if l.get('pdf_url')]
    return {
        'doi': (w.get('doi') or '').replace('https://doi.org/', '').lower(),
        'openalex': w.get('id', ''),
        'title': w.get('title') or '',
        'authors': [a['author']['display_name'] for a in w.get('authorships', [])],
        'year': w.get('publication_year'),
        'venue': src.get('display_name') or '',
        'volume': bib.get('volume') or '', 'issue': bib.get('issue') or '',
        'pages': '-'.join(p for p in (bib.get('first_page'), bib.get('last_page')) if p) or '',
        'type': w.get('type') or '', 'language': w.get('language') or '',
        'is_oa': oa.get('is_oa', False), 'oa_status': oa.get('oa_status', ''),
        'oa_url': oa.get('oa_url') or '', 'pdf_url': best.get('pdf_url') or '', 'pdf_urls': pdfs,
        'cited_by': w.get('cited_by_count', 0),
        'abstract': _abstract(w.get('abstract_inverted_index')),
        'source': 'openalex',
    }


def from_crossref(doi: str):
    r = _get(f'https://api.crossref.org/works/{doi}')
    if r is None or r.status_code != 200:
        return None
    m = r.json()['message']
    year = None
    for k in ('published-print', 'published-online', 'issued', 'created'):
        if m.get(k, {}).get('date-parts'):
            year = m[k]['date-parts'][0][0]
            break
    return {
        'doi': doi, 'openalex': '', 'title': (m.get('title') or [''])[0],
        'authors': [' '.join(filter(None, (a.get('given'), a.get('family')))) or a.get('name', '')
                    for a in m.get('author', [])],
        'year': year, 'venue': (m.get('container-title') or [''])[0],
        'volume': m.get('volume', ''), 'issue': m.get('issue', ''), 'pages': m.get('page', ''),
        'type': m.get('type', ''), 'language': m.get('language', ''),
        'is_oa': False, 'oa_status': '', 'oa_url': '', 'pdf_url': '', 'pdf_urls': [],
        'cited_by': m.get('is-referenced-by-count', 0), 'abstract': '', 'source': 'crossref',
    }


def load():
    return json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}


def save(cache):
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding='utf-8')


def verify(dois):
    cache = load()
    todo = [d for d in dict.fromkeys(dois) if d and d not in cache]
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        r = _get('https://api.openalex.org/works', filter='doi:' + '|'.join(chunk), per_page=50)
        got = {}
        if r is not None and r.status_code == 200:
            for w in r.json().get('results', []):
                rec = from_openalex(w)
                got[rec['doi']] = rec
        for d in chunk:
            rec = got.get(d) or from_crossref(d)
            cache[d] = rec or {'doi': d, 'source': 'not_found'}
        save(cache)
        print(f'{min(i + 50, len(todo))}/{len(todo)}')
        time.sleep(0.5)
    return cache


def by_title(title, year=None):
    params = {'search': title, 'per_page': 5}
    if year:
        params['filter'] = f'publication_year:{int(year) - 1}-{int(year) + 1}'
    r = _get('https://api.openalex.org/works', **params)
    if r is None or r.status_code != 200:
        return []
    return [from_openalex(w) for w in r.json().get('results', [])]


def main():
    if sys.argv[1] == '--title':
        for rec in by_title(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None):
            print(rec['doi'], rec['year'], rec['title'][:100], '|', ', '.join(rec['authors'][:3]))
        return
    dois = []
    for f in sys.argv[1:]:
        with open(f, encoding='utf-8') as fh:
            dois += [row['doi'].strip().lower() for row in csv.DictReader(fh) if row.get('doi')]
    cache = verify(dois)
    miss = [d for d in dois if cache.get(d, {}).get('source') == 'not_found']
    print(f'всего {len(set(dois))}, не найдено {len(set(miss))}: {sorted(set(miss))}')


if __name__ == '__main__':
    main()
