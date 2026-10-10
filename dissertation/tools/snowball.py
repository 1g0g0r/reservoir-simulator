"""«Снежный ком» по OpenAlex: списки литературы и цитирующие работы (с 2021 г.) для опорных статей
-> data/snowball_candidates.csv для ручного отбора (дальше seeds*.csv, verify.py, curate.py).

    python snowball.py

Опорные статьи - работы папки books/neft/Парафинистые нефти/Асфальтены (образцы структуры и источник
литературы, сами в обзор заново не добавляются) и ключевые пластовые модели по теме диссертации (1.1.9).
Стоимость - запросы по фильтру OpenAlex, ~4 кредита на статью из 1000 суточных.
"""
import csv

from verify import DATA, _get, from_openalex

SEEDS = {
    # папка Асфальтены
    '10.1016/j.fluid.2005.03.029': 'Akbarzadeh 2005', '10.1016/j.petrol.2010.11.017': 'Solaimany-Nazar 2011',
    '10.1016/j.fluid.2015.03.045': 'Fallahnejad 2015', '10.1007/s13202-015-0221-7': 'Kargarpour 2016',
    '10.1007/s13202-016-0269-z': 'Kor 2017', '10.1016/j.fuel.2013.09.038': 'Kord 2014',
    '10.1016/j.petrol.2006.10.007': 'Papadimitriou 2007', '10.1038/s41598-022-23596-w': 'SciRep 2022',
    '10.1080/01932691.2015.1065418': 'Subramanian 2015', '10.2516/ogst/2018039': 'Tabzar 2018',
    # пластовые модели и опыты парафина, теория осаждения в пористой среде
    '10.2118/24069-pa': 'Ring 1994', '10.2118/50746-ms': 'Wang 1999', '10.1115/1.1924466': 'Wang Civan 2005',
    '10.1134/s199508022208025x': 'Nikiforov 2022', '10.1080/12269328.2014.959623': 'Nie Yang 2014',
    '10.3390/app16178546': 'Bekibayev 2026', '10.1023/a:1018832120659': 'Sharafutdinov 2001',
    '10.15372/pmtf20210608': 'Gaidukov 2021', '10.1007/s13202-020-00924-2': 'Sandyga 2020',
    '10.3390/pr12020421': 'Li 2024', '10.1038/s41598-020-71065-z': 'Xie 2020',
    '10.1007/s11242-015-0600-z': 'Civan 2015', '10.1007/s11242-010-9626-4': 'Bedrikovetsky 2011',
}


def works(filter_, n=50):
    r = _get('https://api.openalex.org/works', filter=filter_, per_page=n, sort='cited_by_count:desc')
    return r.json().get('results', []) if r is not None and r.ok else []


def main():
    with (DATA / 'curation.csv').open(encoding='utf-8') as fh:
        known = {r['id'] for r in csv.DictReader(fh)}
    out, seen = [], set()
    for doi, name in SEEDS.items():
        r = _get(f'https://api.openalex.org/works/doi:{doi}')
        if r is None or not r.ok:
            print('нет в OpenAlex:', doi)
            continue
        w = r.json()
        refs = [x.rsplit('/', 1)[-1] for x in w.get('referenced_works', [])]
        found = []
        for i in range(0, len(refs), 50):
            found += [('ссылается', x) for x in works('openalex:' + '|'.join(refs[i:i + 50]))]
        found += [('цитирует', x) for x in works(f'cites:{w["id"].rsplit("/", 1)[-1]},from_publication_date:2021-01-01')]
        for rel, x in found:
            rec = from_openalex(x)
            if rec['doi'] and rec['doi'] not in known and rec['doi'] not in seen:
                seen.add(rec['doi'])
                out.append([name, rel, rec['doi'], rec['year'], rec['cited_by'], (rec['venue'] or '')[:40],
                            (rec['title'] or '')[:140]])
    with (DATA / 'snowball_candidates.csv').open('w', newline='', encoding='utf-8') as fh:
        wr = csv.writer(fh)
        wr.writerow(['seed', 'relation', 'doi', 'year', 'cited_by', 'venue', 'title'])
        wr.writerows(out)
    print(len(out))


if __name__ == '__main__':
    main()
