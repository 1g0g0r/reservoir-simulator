"""Второй проход скачивания: записи со статусом failed (открытая копия есть, но сайт не отдал PDF).

    <python с curl_cffi> fetch2.py

Повторяет открытые ссылки OpenAlex через curl_cffi с имитацией браузера (так Springer, MDPI,
Wiley, ACS, SAGE отдают свои же открытые PDF) и добавляет типовые прямые адреса PDF издателя
по DOI. Берутся только записи, у которых OpenAlex указал открытую копию (is_oa); платные
статьи не трогаются. Обновляет data/files.csv на месте.
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import pymupdf
from curl_cffi import requests

sys.path.insert(0, str(Path(__file__).parent))
from fetch import DATA, FOLDERS, SOURCES, fname  # noqa: E402

PATTERNS = {  # префикс DOI -> шаблоны прямых PDF издателя
    '10.1007/': ['https://link.springer.com/content/pdf/{doi}.pdf'],
    '10.1186/': ['https://link.springer.com/content/pdf/{doi}.pdf'],
    '10.1038/': ['https://www.nature.com/articles/{tail}.pdf'],
    '10.1002/': ['https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}'],
    '10.1155/': ['https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}'],
    '10.1021/': ['https://pubs.acs.org/doi/pdf/{doi}'],
    '10.1177/': ['https://journals.sagepub.com/doi/pdf/{doi}'],
    '10.1080/': ['https://www.tandfonline.com/doi/pdf/{doi}'],
    '10.1063/': ['https://pubs.aip.org/aip/pof/article-pdf/doi/{doi}'],
    '10.2516/': ['https://ogst.ifpenergiesnouvelles.fr/articles/ogst/pdf/{doi}'],
}


def pages(p: Path) -> int:
    try:
        with pymupdf.open(p) as d:
            return d.page_count
    except Exception:
        return 0


def main():
    meta = json.loads((DATA / 'meta.json').read_text(encoding='utf-8'))
    with (DATA / 'curation.csv').open(encoding='utf-8') as fh:
        sub = {r['id']: r['subtopics'] for r in csv.DictReader(fh)}
    with (DATA / 'files.csv').open(encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    s = requests.Session(impersonate='chrome')
    for r in rows:
        if r['status'] != 'failed':
            continue
        rec = meta.get(r['id'], {})
        if not rec.get('is_oa'):
            continue
        doi = rec.get('doi') or r['id']
        urls = [u for u in dict.fromkeys([rec.get('pdf_url')] + rec.get('pdf_urls', []) + [rec.get('oa_url')]) if u]
        for pre, pats in PATTERNS.items():
            if doi.startswith(pre):
                urls += [p.format(doi=doi, tail=doi.split('/', 1)[1]) for p in pats]
        target = SOURCES / FOLDERS.get(sub.get(r['id'], 'T16').split(';')[0], FOLDERS['T16']) / fname(rec)
        last = r['note']
        for u in urls:
            try:
                resp = s.get(u, timeout=60, allow_redirects=True)
            except Exception as e:
                last = f'сеть: {type(e).__name__}; {u}'
                continue
            finally:
                time.sleep(2)
            if resp.status_code == 200 and resp.content[:5] == b'%PDF-':
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(resp.content)
                if pages(target):
                    r.update(status='downloaded', path=str(target), pages=pages(target),
                             sha1=hashlib.sha1(resp.content).hexdigest(), note=f'curl_cffi: {u}')
                    break
                target.unlink()
            else:
                last = f'HTTP {resp.status_code} {resp.headers.get("content-type", "")[:25]}; {u}'
        if r['status'] == 'failed':
            r['note'] = last
        print(r['id'], r['status'], flush=True)
    with (DATA / 'files.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['id', 'status', 'path', 'pages', 'sha1', 'note'])
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
