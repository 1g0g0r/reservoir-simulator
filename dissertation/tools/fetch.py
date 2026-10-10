"""Учет полных текстов: копии в библиотеке books, скачивание открытых PDF (сразу в books).

    python fetch.py [--no-download]   -> data/files.csv (id, status, path, pages, sha1, note)

Статусы: local_books (есть в библиотеке books; сюда же 10.10.2026 перенесены PDF из experiments/new и
experiments/sources), downloaded (открытая копия скачана в books/neft/Парафинистые нефти/<тема>/), paywalled (открытой копии нет),
failed (открытая ссылка есть, но отдает не PDF - капча, проверка браузером), unverified, no_pdf
(книги и ПО без PDF). Пиратские источники не используются. Пауза между запросами 1.5 с, при 429 -
ожидание с ростом.
"""
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pymupdf
import requests

REVIEW = Path(__file__).resolve().parents[1]
DATA = REVIEW / 'data'
BOOKS = Path.home() / 'books' / 'neft'
BOOK_DIRS = ['Парафинистые нефти', 'Численное моделирование', 'Подземная гидромеханика и фильтрация',
             'Физика пласта и PVT']
# открытые копии скачиваются сразу в библиотеку (до 10.10.2026 - в experiments/sources, оттуда перенесены)
SOURCES = BOOKS / 'Парафинистые нефти'

FOLDERS = {'T01': 'Неизотермическая фильтрация и теплоперенос', 'T02': 'Свойства и состав парафинистых нефтей',
           'T03': 'Термодинамика и фазовое равновесие', 'T04': 'Асфальтены', 'T05': 'Термодинамика и фазовое равновесие',
           'T06': 'Кинетика кристаллизации', 'T07': 'Перенос частиц и кольматация', 'T08': 'Модели пористой среды',
           'T09': 'Реология и гель', 'T10': 'Отложения в пласте и моделирование',
           'T11': 'Отложения в пласте и моделирование', 'T12': 'Численные методы и ПО', 'T13': 'Численные методы и ПО',
           'T14': 'АСПО в скважинах и трубопроводах', 'T15': 'Машинное обучение',
           'T16': 'АСПО в скважинах и трубопроводах'}

# Файлы библиотеки books, которые match_book не находит (имя без фамилии и года) -> id реестра
P = 'Парафинистые нефти/'
BOOK_FILES = {
    '10.2118/24069-pa': P + 'Отложения в пласте и моделирование/!!!!_Simulation of Paraffin Deposition in Reservoirs.pdf',
    '10.1201/b18482': P + 'АСПО в скважинах и трубопроводах/Wax deposition  experimental characterizations, theoretical '
                          'modeling, and field practices by Fogler, H. Scott Huang, Zhenyu Zheng, Sheng (z-lib.org).pdf',
    '10.1016/s0376-7361(08)x7008-6': P + 'Свойства и состав парафинистых нефтей/Paraffin Products Properties, Technologies, '
                                         'Applications by M. Freund, R. CsikГіs, S. Keszthelyi and GY. MГіzes (Eds.) (z-lib.org).pdf',
    '10.1016/j.fuel.2013.09.038': P + 'Асфальтены/kord2014.pdf',
    '10.1007/s12182-015-0071-4': P + 'Термодинамика и фазовое равновесие/Jafaribehbahani 2016 - Experimental study and a '
                                     'proposed new approach for thermodynamic modeling of wax precipitation in crude oil using a PC-SAFT.pdf',
    '10.1016/j.petsci.2022.08.008': P + 'Свойства и состав парафинистых нефтей/review2022_crude_oil_wax_petroleum_science.pdf',
    '10.1177/00368504261428354': P + 'Заводнение парафинистых пластов/pmc12949295_high_pour_point_co2_hot_water.xml',
    '10.1016/j.heliyon.2023.e22292': P + 'Модели пористой среды/pmc10724555_review_porous_media_models.xml',
    '10.1021/ef049819e': P + 'Кинетика кристаллизации/Kriz 2005 - Effect of Asphaltenes on Crude Oil Wax Crystallization.pdf',
    '10.1021/acs.cgd.6b00499': P + 'Кинетика кристаллизации/pechook2016.pdf',
    '10.1016/j.advwatres.2020.103524': P + 'Модели пористой среды/ghanbarian2020.pdf',
    '10.1016/j.apm.2020.06.037': P + 'Модели пористой среды/lisitsa2020.pdf',
    '10.15372/pmtf20210608': P + 'Отложения в пласте и моделирование/Техногенное изменение проницаемости пласта при '
                                 'неизотермической фильтрации системы нефть — газ — парафин в условиях фазовых переходов л. а. гайдуков.pdf',
    '10.15372/khur2023462': P + 'Свойства и состав парафинистых нефтей/Физико-химические и реологические свойства вязких '
                                'парафинистых нефтей.pdf',
    '10.37878/2708-0080/2022-4.09': P + 'Термодинамика и фазовое равновесие/4_nomer_itog-128-141.pdf',
    '10.1016/0378-3812(94)02600-6': P + 'Термодинамика и фазовое равновесие/coutinho1995.pdf',
    '10.1016/s0378-3812(98)00366-5': P + 'Термодинамика и фазовое равновесие/Pauly 1998 - Liquid–solid equilibria in a '
                                         'decaneqmulti-paraffins system.pdf',
    '10.6028/jres.066a.024': P + 'Термодинамика и фазовое равновесие/broadhurst1962_nalkanes_melting_nbs.pdf',
    '10.1021/es5013438': P + 'Модели пористой среды/molins2014.pdf',
    '10.1016/j.fluid.2015.03.045': P + 'Асфальтены/fallahnejad2015.pdf',
    '10.3390/eng7040184': P + 'Отложения в пласте и моделирование/togasheva2026_uzen_cooling_WAT.pdf',
    '10.1007/s13202-020-00924-2': P + 'Отложения в пласте и моделирование/sandyga2020_formation_damage_wax_core.pdf',
    # PDF, лежавшие в experiments/new (перенесены 10.10.2026; соответствие имён - experiments/new/ССЫЛКИ_НА_СТАТЬИ.md)
    '10.3390/app16178546': P + 'Отложения в пласте и моделирование/Bekibayev 2026 - Thermochemical Removal of Near-Wellbore Paraffin Deposits Using the Activated Aluminum–Water Reaction A Coupled M.pdf',
    '10.1126/sciadv.abc2530': P + 'Перенос частиц и кольматация/Bizmark 2020 - Multiscale dynamics of colloidal deposition and erosion in porous media.pdf',
    '10.1021/ef700670f': P + 'Асфальтены/Boek 2008 - Deposition of Colloidal Asphaltene in Capillary Flow Experiments and Mesoscopic Simulation.pdf',
    'm:do2021': P + 'Диссертации и ВКР/Do 2021 - Effect of temperature on asphaltene deposition mechanisms in horizontal flow.pdf',
    '10.1103/physreve.95.013110': P + 'Перенос частиц и кольматация/Jäger 2017 - Channelization in porous media driven by erosion and deposition.pdf',
    '10.1038/s41598-024-54395-0': P + 'Асфальтены/Jalili 2024 - Experimental study of asphaltene deposition during CO2 and flue gas injection EOR methods employing a long core.pdf',
    '10.1016/j.fuel.2020.118871': P + 'Асфальтены/Lin 2021 - Pore-scale imaging of asphaltene-induced pore clogging in carbonate rocks.pdf',
    '10.3390/en17102415': P + 'Отложения в пласте и моделирование/Lu 2024 - Nanofluidic Study of Multiscale Phase Transitions and Wax Precipitation in Shale Oil Reservoirs.pdf',
    'm:mahmoudi2022': P + 'Асфальтены/Mahmoudi 2022 - Asphaltene deposition in porous media micromodels experimental studies and comprehensive permeability-reducing mec.pdf',
    'm:maloney2004': P + 'Заводнение парафинистых пластов/Maloney 2004 - Effects of paraffin wax precipitation during cold water injection in a fractured carbonate reservoir.pdf',
    '10.29047/01225383.251': P + 'Перенос частиц и кольматация/Piñerez 2020 - The role of polar organic components in dynamic crude oil adsorption on sandstones and carbonates.pdf',
    '10.1007/s13202-016-0276-0': P + 'Термодинамика и фазовое равновесие/Стручков 2017 - Wax precipitation in multicomponent hydrocarbon system.pdf',
    '10.1007/s13202-018-0539-z': P + 'Асфальтены/Стручков 2019 - Laboratory investigation of asphaltene-induced formation damage.pdf',
    '10.1021/acsomega.2c07247': P + 'Асфальтены/Wang 2023 - Full-Scale Experimental Study on the Effect of CO2 Flooding on Storage-Seepage Capacity of Tight Sandstone Reservoirs.pdf',
    '10.3390/en19071777': P + 'Отложения в пласте и моделирование/Wang 2026 - Enhancing Oil Recovery and CO2 Sequestration Efficiency in Ultra-Deep Heterogeneous Waxy Reservoirs A Comparative Expe.pdf',
    '10.1016/j.petsci.2022.08.026': P + 'Асфальтены/Xiong 2023 - The deposition of asphaltenes under high-temperature and high-pressure (HTHP) conditions.pdf',
    '10.1021/acsomega.2c05480': P + 'Асфальтены/Zhang 2022 - Quantification of Methane-Induced Asphaltene Precipitation in a Multiple Contact Process.pdf',
}

S = requests.Session()
S.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) reservoir-simulator-review/1.0 '
                                '(open-access only)', 'Accept': 'application/pdf,*/*'})
STOP = {'the', 'and', 'for', 'with', 'from', 'using', 'during', 'into', 'their', 'oil', 'of', 'in', 'on', 'a', 'an'}


def norm(s: str) -> list:
    s = re.sub(r'[^a-zа-яё0-9 ]', ' ', s.lower().replace('ﬁ', 'fi').replace('ﬀ', 'ff'))
    return [w for w in s.split() if len(w) > 2 and w not in STOP]


def pdf_pages(path: Path) -> int:
    try:
        with pymupdf.open(path) as d:
            return d.page_count
    except Exception:
        return 0


def fname(rec: dict) -> str:
    a = (rec.get('authors') or ['anon'])[0].split()[-1]
    a = re.sub(r'[^A-Za-zА-Яа-яЁё-]', '', a)[:20] or 'anon'
    short = '_'.join(norm(rec.get('title', ''))[:5])[:60] or 'untitled'
    return f"{rec.get('year') or 'nd'}_{a}_{short}.pdf"


def book_index():
    idx = []
    for d in BOOK_DIRS:
        for p in (BOOKS / d).rglob('*'):
            if p.suffix.lower() in ('.pdf', '.djvu', '.doc', '.docx', '.xml'):
                idx.append((p, set(norm(p.stem))))
    return idx


def match_book(rec, idx, exclude=frozenset()):
    """Файл библиотеки books с тем же названием. Фамилия первого автора обязана быть в имени файла,
    кроме полного совпадения длинного названия: иначе общие слова дают ложные совпадения
    (Coats 1995 и файл Zaydullin 2014, Blunt 2016 и Liefferink 2018). Файлы, явно закреплённые за другими
    записями (exclude), не рассматриваются."""
    tw = set(norm(rec.get('title', '')))
    if len(tw) < 3:
        return None
    first = (rec.get('authors') or [''])[0].split()
    sur = norm(first[-1])[0] if first and norm(first[-1]) else ''
    best, score = None, 0.0
    year = str(rec.get('year') or '')
    for p, ws in idx:
        if str(p) in exclude:
            continue
        s = len(tw & ws) / len(tw) if ws else 0
        # фамилия и год в имени плюс часть слов названия: у одного автора за год бывает несколько работ
        # (Zhang 2022, Wang 2023, Hammami 1999 - ложные совпадения при одном правиле «фамилия+год»)
        if sur and year and s >= 0.15 and re.search(rf'(^|[^a-zа-яё]){re.escape(sur)}[\s_-]*{year}', p.stem.lower()):
            return p                                   # sandyga2020_formation_damage_wax_core.pdf
        if s >= 0.75 and not (sur and sur in ws) and not (s == 1.0 and len(tw) >= 5):
            continue
        if s > score:
            best, score = p, s
    return best if score >= 0.75 else None


def download(urls, target: Path):
    last = 'нет открытой ссылки'
    for url in urls:
        r = None
        for attempt in range(3):
            try:
                r = S.get(url, timeout=60, allow_redirects=True)
            except requests.RequestException as e:
                last, r = f'сеть: {type(e).__name__}', None
                break
            if r.status_code != 429:
                break
            time.sleep(5 * (attempt + 1))
        time.sleep(1.5)
        if r is None:
            continue
        if r.status_code == 200 and r.content[:5] == b'%PDF-':
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(r.content)
            if pdf_pages(target) > 0:
                return True, url
            target.unlink()
            last = 'PDF не открывается'
        else:
            last = f'HTTP {r.status_code}, {r.headers.get("content-type", "")[:30]}'
    return False, last


def main():
    meta = json.loads((DATA / 'meta.json').read_text(encoding='utf-8'))
    with (DATA / 'curation.csv').open(encoding='utf-8') as fh:
        cur = [r for r in csv.DictReader(fh) if r['include'] == '1']
    with (DATA / 'manual.csv').open(encoding='utf-8') as fh:
        manual = {r['id']: r for r in csv.DictReader(fh)}
    old = {}
    if (DATA / 'files.csv').stat().st_size:
        with (DATA / 'files.csv').open(encoding='utf-8') as fh:
            old = {r['id']: r for r in csv.DictReader(fh)}
    idx = book_index()
    assigned = ({str(BOOKS / v) for v in BOOK_FILES.values()}
                | {str(BOOKS / m['local']) for m in manual.values() if m.get('local')}
                | {o['path'] for o in old.values() if o['status'] == 'local_books' and 'перенес' in o['note']})
    out, seen_sha = [], {}
    for r in cur:
        rid = r['id']
        mm = manual.get(rid, {})
        rec = meta.get(rid) or {'title': mm.get('title', ''), 'year': mm.get('year'), 'authors': [mm.get('authors', 'anon')]}
        folder = SOURCES / FOLDERS.get(r['subtopics'].split(';')[0], FOLDERS['T16'])
        status, path, note = '', '', ''
        if mm.get('local') or rid in BOOK_FILES:
            status, path = 'local_books', str(BOOKS / (mm.get('local') or BOOK_FILES[rid]))
        elif (book := match_book(rec, idx, assigned)) is not None:
            status, path = 'local_books', str(book)
        elif rid.startswith('m:'):
            status, note = mm.get('status') or 'no_pdf', mm.get('verified_by', '')
        elif rid in old and (old[rid]['status'] in ('downloaded', 'failed', 'paywalled')
                             or old[rid]['status'] == 'local_books' and 'перенес' in old[rid]['note']
                             and Path(old[rid]['path']).exists()):
            # уже проверено: повторно не скачиваем (второй проход - fetch2.py); перенесённые в books - не трогаем
            status, path, note = old[rid]['status'], old[rid]['path'], old[rid]['note']
        else:
            urls = [u for u in dict.fromkeys([rec.get('pdf_url')] + rec.get('pdf_urls', []) + [rec.get('oa_url')]) if u]
            if not urls:
                status, note = 'paywalled', 'OpenAlex: открытой копии нет'
            elif '--no-download' in sys.argv:
                status, note = 'pending', ' '.join(urls[:2])
            else:
                target = folder / fname(rec)
                ok, info = download(urls, target)
                status, path, note = ('downloaded', str(target), info) if ok else ('failed', '', f'{info}; {urls[0]}')
                print(rid, status, info[:90], flush=True)
        pages, h = '', ''
        if path and path.lower().endswith('.pdf') and Path(path).exists():
            pages, h = pdf_pages(Path(path)), hashlib.sha1(Path(path).read_bytes()).hexdigest()
            if h in seen_sha:
                note = (note + f' дубль файла {seen_sha[h]}').strip()
            seen_sha[h] = rid
        out.append([rid, status, path, pages, h, note])
    with (DATA / 'files.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['id', 'status', 'path', 'pages', 'sha1', 'note'])
        w.writerows(out)
    print(Counter(o[1] for o in out))


if __name__ == '__main__':
    main()
