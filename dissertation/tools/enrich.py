"""Год выпуска тома, том, выпуск и страницы по Crossref для включенных DOI -> data/crossref.json.

    python enrich.py

OpenAlex дает год первой (онлайн) публикации; для ссылки по ГОСТ нужен год выпуска тома
(Crossref published-print, иначе issued). Почта в запросы не передается. Уже собранные DOI пропускаются.
"""
import csv
import json
import time
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parents[1] / 'data'
OUT = DATA / 'crossref.json'


def year(m, key):
    parts = (m.get(key) or {}).get('date-parts') or [[None]]
    return parts[0][0]


def main():
    done = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
    with (DATA / 'curation.csv').open(encoding='utf-8') as fh:
        dois = [r['id'] for r in csv.DictReader(fh) if r['include'] == '1' and r['id'].startswith('10.')]
    s = requests.Session()
    s.headers['User-Agent'] = 'reservoir-simulator literature review'
    for i, d in enumerate(dois):
        if d in done:
            continue
        try:
            r = s.get(f'https://api.crossref.org/works/{d}', timeout=40)
        except requests.RequestException:
            continue
        if r.status_code == 200:
            m = r.json()['message']
            done[d] = dict(year_print=year(m, 'published-print'), year_issued=year(m, 'issued'),
                           volume=m.get('volume', ''), issue=m.get('issue', ''), page=m.get('page', ''),
                           container=(m.get('container-title') or [''])[0])
        else:
            done[d] = {'status': r.status_code}
        if i % 25 == 0:
            OUT.write_text(json.dumps(done, ensure_ascii=False), encoding='utf-8')
        time.sleep(0.3)
    OUT.write_text(json.dumps(done, ensure_ascii=False), encoding='utf-8')
    print(len(done))


if __name__ == '__main__':
    main()
