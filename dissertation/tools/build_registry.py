"""Реестр источников, BibTeX, аннотированная библиография и список нескачанного.

    python build_registry.py

Читает data/meta.json (OpenAlex/Crossref), data/manual.csv (записи без DOI), data/curation.csv
(подтемы, релевантность, включение), data/annotations.txt (аннотации), data/files.csv (полные тексты),
data/excludes.csv. Пишет sources_registry.csv, references.bib, sources_annotated.md,
download_failures.md и data/keys.json (id -> ключ BibTeX, нужен tools/build.py).
Функция gost() форматирует запись по ГОСТ Р 7.0.5-2008 (ее импортирует build.py).
"""
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

REVIEW = Path(__file__).resolve().parents[1]
DATA = REVIEW / 'data'
SUBTOPICS = {
    'T01': 'Неизотермическая многофазная фильтрация и теплоперенос', 'T02': 'Состав нефти и характеризация',
    'T03': 'Термодинамика выпадения парафина', 'T04': 'Асфальтены и смолы: термодинамика и свойства',
    'T05': 'Уравнения состояния, устойчивость, flash', 'T06': 'Кинетика кристаллизации, агрегации и старения',
    'T07': 'Перенос частиц, глубинная фильтрация, вынос, адсорбция', 'T08': 'Модели порового пространства и связь k–φ',
    'T09': 'Реология, гель, течение с пределом текучести', 'T10': 'Модели АСПО в пласте и призабойной зоне',
    'T11': 'Опытные данные и валидация', 'T12': 'Численные методы', 'T13': 'Программные платформы, верификация, бенчмарки',
    'T14': 'Отложения в скважинах и трубопроводах', 'T15': 'Машинное обучение и суррогаты', 'T16': 'Обзоры и монографии',
}
TRANSLIT = dict(zip('абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
                    ['a', 'b', 'v', 'g', 'd', 'e', 'e', 'zh', 'z', 'i', 'y', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't',
                     'u', 'f', 'kh', 'ts', 'ch', 'sh', 'shch', '', 'y', '', 'e', 'yu', 'ya']))
PARTICLES = {'van', 'de', 'da', 'der', 'den', 'von', 'la', 'le', 'dos', 'del', 'di'}
CROSSREF = json.loads((DATA / 'crossref.json').read_text(encoding='utf-8')) if (DATA / 'crossref.json').exists() else {}
# Авторы, которых OpenAlex записал с перепутанным порядком имени или обрезанной двойной фамилией
AUTHOR_FIX = {
    '10.2118/50746-ms': ['Wang S.', 'Civan F.', 'Strycker A.R.'],
    '10.29047/01225383.251': ['Piñerez Torrijos I.D.', 'Mamonov A.', 'Strand S.', 'Puntervold T.'],
    '10.1016/j.petrol.2010.11.017': ['Solaimany-Nazar A.R.', 'Zonnouri A.'],
    '10.1038/s41598-024-54395-0': ['Jalili Darbandi Sofla M.', 'Dermanaki Farahani Z.', 'Ghorbanizadeh S.', 'Namdar H.'],
    '10.1021/ef9006142': ['Mendoza de la Cruz J.L.', 'Argüelles-Vivas F.J.', 'Matías-Pérez V.', 'Durán-Valencia C.A.', 'López-Ramírez S.'],
    '10.1021/acs.energyfuels.6b01289': ['Vilas Bôas Fávero C.', 'Hanpan A.', 'Phichphimok P.', 'Binabdullah K.', 'Fogler H.S.'],
    '10.1016/j.petrol.2017.06.064': ['da Silva V.M.', 'do Carmo R.P.', 'Fleming F.P.', 'Daridon J.-L.', 'Pauly J.', 'Tavares F.W.'],
    '10.1007/s00397-013-0699-1': ['de Souza Mendes P.R.', 'Thompson R.L.'],
    '10.1016/c2013-0-04548-3': ['Elimelech M.', 'Gregory J.', 'Jia X.', 'Williams R.A.'],   # в OpenAlex без авторов
    '10.1016/j.petrol.2005.11.010': ['Altoé Filho J.E.', 'Bedrikovetsky P.', 'Siqueira A.G.', 'de Souza A.L.S.', 'Shecaira F.S.'],
}
TYPE_FIX = {'10.1016/j.petrol.2005.11.010': 'article'}   # статья на 17 с.; OpenAlex считает «Correction…» опечаткой


def ascii_word(s: str) -> str:
    s = ''.join(TRANSLIT.get(c, c) for c in s.lower())
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z]', '', s)


def split_name(name: str):
    """'Jasper N. Ring' -> ('Ring', 'J.N.'); 'Ring J.N.' (уже в форме ГОСТ) -> как есть."""
    name = name.strip().strip(',')
    m = re.match(r'^([^\s,]+(?:\s[^\s,]+)?)\s+((?:[A-ZА-ЯЁ][a-zа-яё]?\.\s?-?)+)$', name)
    if m and '.' in m.group(2):
        return m.group(1), m.group(2).replace(' ', '')
    parts = [p for p in re.split(r'\s+', name) if p]
    if len(parts) == 1:
        return parts[0], ''
    sur = [parts[-1]]
    i = len(parts) - 2
    while i > 0 and parts[i].lower() in PARTICLES:
        sur.insert(0, parts[i])
        i -= 1
    given = parts[:i + 1]
    ini = ''.join(g[0].upper() + '.' for g in given if g and g[0].isalpha() and g.lower() not in PARTICLES)
    return ' '.join(sur), ini


def is_ru(text: str) -> bool:
    return bool(re.search('[а-яё]', text or '', re.I))


def load():
    meta = json.loads((DATA / 'meta.json').read_text(encoding='utf-8'))
    with (DATA / 'manual.csv').open(encoding='utf-8') as fh:
        manual = {r['id']: r for r in csv.DictReader(fh)}
    with (DATA / 'curation.csv').open(encoding='utf-8') as fh:
        cur = list(csv.DictReader(fh))
    with (DATA / 'files.csv').open(encoding='utf-8') as fh:
        files = {r['id']: r for r in csv.DictReader(fh)}
    ann, cid = {}, None
    for line in (DATA / 'annotations.txt').read_text(encoding='utf-8').splitlines():
        if line.startswith('@@ '):
            cid = line[3:].strip()
            ann[cid] = {}
        elif cid and len(line) > 2 and line[1] == ':' and line[0] in 'AKMDL':
            ann[cid][line[0]] = line[2:].strip()
    return meta, manual, cur, files, ann


def record(rid, meta, manual):
    """Единая запись: authors (список 'Фамилия И.О.'), year, title, venue, volume, issue, pages, type, url, doi."""
    if rid.startswith('m:'):
        m = manual[rid]
        authors = [a.strip() for a in m['authors'].split(';') if a.strip()]
        return dict(doi='', url=m['url'], authors=authors, year=m['year'], year_online='', title=m['title'], venue=m['venue'],
                    volume=m['volume'], issue=m['issue'], pages=m['pages'], type=m['type'],
                    language=m['language'], oa='', verified_by=m['verified_by'], cited_by='')
    x = meta[rid]
    authors = []
    for a in x.get('authors', []):
        s, i = split_name(a)
        authors.append(f'{s} {i}'.strip())
    authors = AUTHOR_FIX.get(rid, authors)
    cr = CROSSREF.get(rid, {})
    y_on = x.get('year')
    y = cr.get('year_print') or cr.get('year_issued') or y_on
    venue = x.get('venue') or ''
    if rid.startswith('10.2118/') and rid.endswith('-ms') and (not venue or venue in ('All Days',)):
        venue = f"SPE conference paper {rid.split('/')[1].upper()}"
    lang = 'ru' if is_ru(x.get('title', '')) else (x.get('language') or 'en')
    return dict(doi=rid, url=f'https://doi.org/{rid}', authors=authors, year=y, year_online=y_on if y_on != y else '',
                title=x.get('title', ''), venue=venue or cr.get('container', ''),
                volume=cr.get('volume') or x.get('volume', ''), issue=cr.get('issue') or x.get('issue', ''),
                pages=cr.get('page') or x.get('pages', ''),
                type=TYPE_FIX.get(rid) or {'journal-article': 'article', 'proceedings-article': 'conference-paper'}.get(
                    x.get('type', ''), x.get('type', '')),
                language=lang, oa=x.get('oa_status', ''),
                verified_by=f"{x.get('source', '')}: DOI разрешен, название совпадает", cited_by=x.get('cited_by', ''))


def gost(r) -> str:
    """Библиографическая ссылка по ГОСТ Р 7.0.5-2008 (затекстовая, упрощенная)."""
    ru = r.get('language') == 'ru' or is_ru(r['title'])
    au = r['authors'][:3]
    head = ', '.join(au)
    if len(r['authors']) > 3:
        head += ' [и др.]' if ru else ' [et al.]'
    t = r['type']
    title = r['title'].rstrip('.')
    if t in ('book', 'monograph') or (t == 'book' or r['venue'].startswith(('М.:', 'СПб.', 'London', 'Amsterdam', 'San Diego', 'New York', 'Holte', 'Richardson'))):
        s = f'{head} {title}. {r["venue"]}, {r["year"]}.' if r['venue'] else f'{head} {title}. {r["year"]}.'
        if r['pages']:
            s += f' {r["pages"]} {"с" if ru else "p"}.'
    elif t in ('thesis', 'dissertation'):
        s = f'{head} {title}: {r["venue"]}. {r["year"]}.'
        if r['pages']:
            s += f' {r["pages"]} {"с" if ru else "p"}.'
    elif t == 'software':
        s = f'{head}. {title} [Электронный ресурс]: {r["venue"]}. {r["year"]}.'
    else:
        s = f'{head} {title} // {r["venue"]}. {r["year"]}.'
        if r['volume']:
            s += f' {"Т." if ru else "Vol."} {r["volume"]}'
            s += f', {"№" if ru else "No."} {r["issue"]}.' if r['issue'] else '.'
        elif r['issue']:
            s += f' № {r["issue"]}.'
        if r['pages']:
            s += f' {"С." if ru else "P."} {r["pages"].replace("-", "–")}.'
    if r['doi']:
        s += f' DOI: {r["doi"]}.'
    elif r['url']:
        s += f' URL: {r["url"]}.'
    return re.sub(r'\s+', ' ', s.replace('..', '.')).strip()


def bibkey(r, used):
    first = r['authors'][0] if r['authors'] else 'Anon'
    head, _, last = first.rpartition(' ')
    surname = head if head and '.' in last else first   # 'da Silva V.M.' -> 'da Silva'
    surname = surname.split()[-1] if surname.split() else surname  # частицы в ключ не идут
    sur = ascii_word(surname) or 'anon'
    word = next((ascii_word(w) for w in re.findall(r"[\w'-]+", r['title']) if len(ascii_word(w)) > 3), 'x')
    base = f'{sur.capitalize()}{r["year"] or "nd"}{word.capitalize()}'
    key, n = base, 0
    while key in used:
        n += 1
        key = base + 'abcdefghij'[n - 1]
    used.add(key)
    return key


def bibtex(key, r) -> str:
    t = r['type']
    kind = {'book': 'book', 'thesis': 'phdthesis', 'software': 'misc', 'conference': 'inproceedings',
            'proceedings-article': 'inproceedings', 'book-chapter': 'incollection'}.get(t, 'article')
    if kind == 'article' and 'conference paper' in r['venue']:
        kind = 'inproceedings'
    esc = lambda s: str(s).replace('{', '').replace('}', '').replace('&', r'\&').replace('%', r'\%')
    authors = ' and '.join(esc(a) for a in r['authors']) or 'Anonymous'
    f = [f'  author = {{{authors}}}', f'  title = {{{{{esc(r["title"])}}}}}', f'  year = {{{r["year"]}}}']
    venue_field = {'article': 'journal', 'inproceedings': 'booktitle', 'incollection': 'booktitle',
                   'book': 'publisher', 'phdthesis': 'school', 'misc': 'howpublished'}[kind]
    if r['venue']:
        f.append(f'  {venue_field} = {{{esc(r["venue"])}}}')
    for k in ('volume', 'pages'):
        if r[k]:
            f.append(f'  {k} = {{{esc(r[k]).replace("-", "--")}}}')
    if r['issue']:
        f.append(f'  number = {{{esc(r["issue"])}}}')
    if r['doi']:
        f.append(f'  doi = {{{r["doi"]}}}')
    elif r['url']:
        f.append(f'  url = {{{r["url"]}}}')
    if r['language'] == 'ru':
        f.append('  language = {russian}')
    return f'@{kind}{{{key},\n' + ',\n'.join(f) + '\n}\n'


def main():
    meta, manual, cur, files, ann = load()
    with (DATA / 'excludes.csv').open(encoding='utf-8') as fh:
        excl = list(csv.DictReader(fh))
    used, keys, rows = set(), {}, []
    for c in cur:
        if c['include'] != '1':
            continue
        rid = c['id']
        r = record(rid, meta, manual)
        key = bibkey(r, used)
        keys[rid] = key
        fi = files.get(rid, {})
        status = fi.get('status', '')
        if rid.startswith('m:') and manual[rid].get('status') == 'unverified':
            status = 'unverified'
        a = ann.get(rid, {})
        rows.append(dict(key=key, id=rid, doi=r['doi'], url=r['url'], authors='; '.join(r['authors']), year=r['year'],
                         year_online=r['year_online'],
                         title=r['title'], venue=r['venue'], volume=r['volume'], issue=r['issue'], pages=r['pages'],
                         type=r['type'], language=r['language'], oa_status=r['oa'], status=status,
                         file_path=fi.get('path', ''), pages_pdf=fi.get('pages', ''), sha1=fi.get('sha1', ''),
                         subtopics=c['subtopics'], relevance=c['relevance'], annotation=a.get('A', ''),
                         contribution=a.get('K', ''), method=a.get('M', ''), data_model=a.get('D', ''),
                         limitations=a.get('L', ''), verified_by=r['verified_by'], cited_by=r['cited_by'],
                         file_note=fi.get('note', ''), gost=gost(r)))
    rows.sort(key=lambda x: (str(x['year']), x['key']))
    with (REVIEW / 'sources_registry.csv').open('w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != 'gost'], extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    (REVIEW / 'references.bib').write_text(
        '% BibTeX источников обзора (сгенерировано tools/build_registry.py)\n\n' +
        '\n'.join(bibtex(r['key'], record(r['id'], meta, manual)) for r in rows if r['status'] != 'unverified'),
        encoding='utf-8')
    (DATA / 'keys.json').write_text(json.dumps(keys, ensure_ascii=False, indent=0), encoding='utf-8')

    # аннотированная библиография
    by = defaultdict(list)
    for r in rows:
        by[r['subtopics'].split(';')[0]].append(r)
    st = {'downloaded': 'PDF скачан', 'local_new': 'PDF в experiments/new', 'local_books': 'PDF в библиотеке books',
          'paywalled': 'платный доступ', 'failed': 'открытая копия есть, автоматически не скачана', 'no_pdf': 'без PDF',
          'unverified': '[НЕ ПРОВЕРЕНО]'}
    out = ['# Аннотированная библиография', '',
           f'*{len(rows)} источников по 16 подтемам (коды — `01_scope.md`). Для каждого: ссылка по ГОСТ Р 7.0.5–2008, '
           'аннотация, ключевой вклад, метод, данные или тип модели, ограничения, релевантность, доступ. '
           'Сгенерировано `tools/build_registry.py` из `data/annotations.txt` и метаданных OpenAlex/Crossref.*', '']
    for code in sorted(by):
        out += [f'## {code}. {SUBTOPICS[code]} ({len(by[code])})', '']
        for r in sorted(by[code], key=lambda x: (str(x['year']), x['key'])):
            out += [f'**[{r["key"]}]** {r["gost"]}', '',
                    f'- *Аннотация.* {r["annotation"]}',
                    f'- *Вклад:* {r["contribution"]}. *Метод:* {r["method"]}. *Данные/модель:* {r["data_model"]}. '
                    f'*Ограничения:* {r["limitations"]}.',
                    f'- *Подтемы:* {r["subtopics"]}; *релевантность:* {r["relevance"]}; *доступ:* {st.get(r["status"], r["status"])}'
                    + (f' (`{Path(r["file_path"]).name}`)' if r['file_path'] else '') + '.', '']
    unv = [r for r in rows if r['status'] == 'unverified']
    out += ['## Не проверены (`[НЕ ПРОВЕРЕНО]`)', '',
            'Записи, которые не удалось подтвердить ни по DOI, ни по открытому URL, ни по каталогу. В тексте обзора '
            'они не используются как опора выводов.', '']
    out += [f'- `[НЕ ПРОВЕРЕНО]` {r["gost"]} — {r["verified_by"]}' for r in unv] + ['']
    out += ['## Исключены, потому что…', '',
            'Кандидаты из наработок проекта и поиска, не включенные в реестр, с причиной.', '']
    out += [f'- `{e["id"]}` — {e["reason"]}' for e in excl]
    out += ['- Ring (1991), Balakotaiah & West, Fan et al. (2021), Sarmurzina et al. (2026) — упомянуты внутри разбора '
            'Bekibayev et al. (2026) как источники его корреляций; это вторичные ссылки чужой модели, к теме обзора '
            'прямо не относятся.', '']
    (REVIEW / 'sources_annotated.md').write_text('\n'.join(out), encoding='utf-8')

    # не скачано
    reason = {'paywalled': 'платный доступ: в OpenAlex нет открытой копии',
              'failed': 'открытая копия заявлена, но сайт отдает HTML/капчу/403 автоматическому клиенту — скачать вручную по ссылке',
              'no_pdf': 'книга, ПО или реферат без PDF', 'unverified': 'не проверено'}
    fails = [r for r in rows if r['status'] in reason]
    lines = ['# Что не скачано и почему', '',
             f'*Всего {len(fails)} из {len(rows)}. Пиратские ресурсы не использовались (`assumptions.md`, п. 2). '
             'Для статуса «открытая копия есть» ссылка ведет на легальную открытую версию — ее можно открыть в браузере.*', '']
    for s in ('failed', 'paywalled', 'no_pdf', 'unverified'):
        grp = [r for r in fails if r['status'] == s]
        lines += [f'## {reason[s]} ({len(grp)})', '', '| Ключ | Год | Работа | DOI / URL | Примечание |', '|---|---|---|---|---|']
        for r in grp:
            ident = r['doi'] or r['url']
            note = r['file_note'].replace('|', '/')[:140] if s == 'failed' else (r['verified_by'][:120] if s == 'unverified' else '')
            lines.append(f'| {r["key"]} | {r["year"]} | {r["authors"].split(";")[0]} — {r["title"][:90].replace("|", "/")} | {ident} | {note} |')
        lines.append('')
    (REVIEW / 'download_failures.md').write_text('\n'.join(lines), encoding='utf-8')
    from collections import Counter
    print(len(rows), Counter(r['status'] for r in rows), 'unique keys:', len(set(keys.values())) == len(keys))


if __name__ == '__main__':
    main()
