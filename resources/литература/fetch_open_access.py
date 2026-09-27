"""Скачивание открытых (легальных) копий статей, на которых построен детальный состав нефти.

    python resources/литература/fetch_open_access.py --email you@example.org

Для каждой статьи сначала пробуется известная открытая ссылка (издатель в открытом доступе, NIST,
arXiv, OSTI, Europe PMC), затем - Unpaywall по DOI (api.unpaywall.org требует адрес почты в запросе;
он уходит только туда). Платные статьи не скачиваются: скрипт пишет, что их нет в открытом доступе.
Пиратские источники не используются. Уже скачанные файлы пропускаются.

Статьи разложены по папкам по темам, описание каждой - в README.md папки. В среде, где собиралась модель,
внешняя сеть была закрыта, поэтому PDF в репозитории нет - их кладет этот скрипт.
"""
import argparse
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent

# (папка, файл, DOI, известная открытая ссылка на PDF или None)
ARTICLES = [
    # --- состав нефти и равновесие твердое-жидкость ---
    ('состав_нефти', 'broadhurst1962_nalkanes_melting_nbs.pdf', '10.6028/jres.066A.024',
     'https://nvlpubs.nist.gov/nistpubs/jres/066/3/V66.N03.A05.pdf'),
    ('состав_нефти', 'won1986_wax_solid_solution.pdf', '10.1016/0378-3812(86)80061-9', None),
    ('состав_нефти', 'liragaleana1996_multisolid_wax.pdf', '10.1002/aic.690420120', None),
    ('состав_нефти', 'pedersen1991_north_sea_wax_modeling.pdf', '10.1021/ef00030a021', None),
    ('состав_нефти', 'pedersen1995_cloud_point.pdf', '10.2118/27629-PA', None),
    ('состав_нефти', 'hansen1988_wax_thermodynamic_model.pdf', '10.1002/aic.690341202', None),
    ('состав_нефти', 'coutinho1996_wilson_nalkanes.pdf', '10.1021/ie950447u', None),
    ('состав_нефти', 'katz1978_scn_plus_fraction.pdf', '10.2118/6721-PA', None),
    ('состав_нефти', 'whitson1983_plus_fractions.pdf', '10.2118/12233-PA', None),
    ('состав_нефти', 'riazi1987_petroleum_fractions.pdf', '10.1021/ie00064a023', None),
    # --- давление и WAT ---
    ('давление_WAT', 'pan1997_pressure_composition_wax.pdf', '10.2118/36740-PA', None),
    ('давление_WAT', 'hpudsc_presalt_wat_pressure_gas.pdf', None,
     'https://europepmc.org/articles/PMC12878762?pdf=render'),
    # --- асфальтены и смолы ---
    ('асфальтены', 'hirschberg1984_asphaltene_flocculation.pdf', '10.2118/11202-PA', None),
    ('асфальтены', 'leontaritis1987_colloidal_asphaltene.pdf', '10.2118/16258-MS', None),
    ('асфальтены', 'wang_buckley2001_onset_two_component.pdf', '10.1021/ef010012l', None),
    ('асфальтены', 'buckley1998_asphaltene_solvent_properties.pdf', '10.1080/10916469808949783', None),
    ('асфальтены', 'akbarzadeh2005_regular_solution_asphaltene.pdf', '10.1016/j.fluid.2005.03.029',
     'https://www.ucalgary.ca/ENCH/AER/papers/Akbarzadeh,Alboudwarej,Svrcek,Yarranton,Fluid%20Phase%20Equil,232,159-170,2005.pdf'),
    ('асфальтены', 'yen2001_cii_inhibitors.pdf', '10.2118/65376-MS', None),
    ('асфальтены', 'maqbool2009_asphaltene_kinetics.pdf', '10.1021/ef9002236', None),
    ('асфальтены', 'mullins2012_yen_mullins_model.pdf', '10.1021/ef300185p', None),
    ('асфальтены', 'wang_civan2001_productivity_decline_asphaltene.pdf', '10.2118/64991-MS', None),
    ('асфальтены', 'tabzar2018_asphaltene_multiphase_ogst.pdf', '10.2516/ogst/2018039',
     'https://ogst.ifpenergiesnouvelles.fr/articles/ogst/pdf/2018/01/ogst180048.pdf'),
    ('асфальтены', 'nabzar2008_colloidal_asphaltene_ogst.pdf', '10.2516/ogst:2007083',
     'https://ogst.ifpenergiesnouvelles.fr/articles/ogst/pdf/2008/01/ogst07068.pdf'),
    ('асфальтены', 'darabi2014_asphaltene_wellbore_reservoir.pdf', '10.2118/169121-MS', None),
    ('асфальтены', 'scirep2022_flory_huggins_inhibitors.pdf', '10.1038/s41598-022-23596-w', None),
    ('асфальтены', 'kor2017_asphaltene_models_well.pdf', '10.1007/s13202-016-0269-z', None),
    ('асфальтены', 'kargarpour2016_marrat_asphaltene.pdf', '10.1007/s13202-015-0221-7', None),
    # --- реология и гель ---
    ('реология_гель', 'pedersen_ronningsen2000_wax_viscosity.pdf', '10.1021/ef9901185', None),
    ('реология_гель', 'letoffe1995_wax_dsc_thermomicroscopy.pdf', '10.1016/0016-2361(94)00006-D', None),
    ('реология_гель', 'venkatesan2005_paraffin_gel_strength.pdf', '10.1016/j.ces.2005.02.045', None),
    ('реология_гель', 'visintin2005_waxy_gel_rheology.pdf', '10.1021/la050705k', None),
    ('реология_гель', 'shih1990_colloidal_gel_scaling.pdf', '10.1103/PhysRevA.42.4772', None),
    ('реология_гель', 'dimitriou2014_waxy_crude_thixotropy.pdf', '10.1039/C4SM00578C', None),
    ('реология_гель', 'singh2000_wax_oil_gel_aging.pdf', '10.1002/aic.690460517', None),
    ('реология_гель', 'wu_pruess1992_bingham_porous_media.pdf', '10.2118/20051-PA', None),
    ('реология_гель', 'chevalier2013_darcy_yield_stress.pdf', '10.1016/j.jnnfm.2012.12.005', None),
    ('реология_гель', 'liu2019_darcy_yield_stress_prl_arxiv.pdf', '10.1103/PhysRevLett.122.245502',
     'https://arxiv.org/pdf/1811.09494'),
    ('реология_гель', 'jtac2014_solid_fraction_gel_strength.pdf', '10.1007/s10973-014-3660-3', None),
]


def _download(url: str, target: Path, session: requests.Session) -> bool:
    try:
        resp = session.get(url, timeout=60, allow_redirects=True)
    except requests.RequestException as err:
        print(f'   ошибка сети: {err}')
        return False
    if resp.status_code != 200 or not resp.content.startswith(b'%PDF'):
        print(f'   {url}: ответ {resp.status_code}, не PDF')
        return False
    target.write_bytes(resp.content)
    print(f'   сохранено {target.relative_to(HERE)} ({len(resp.content) // 1024} КБ)')
    return True


def _unpaywall(doi: str, email: str, session: requests.Session):
    try:
        resp = session.get(f'https://api.unpaywall.org/v2/{doi}', params={'email': email}, timeout=30)
        if resp.status_code != 200:
            return None
        best = resp.json().get('best_oa_location') or {}
        return best.get('url_for_pdf')
    except (requests.RequestException, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--email', required=True, help='адрес для запросов к Unpaywall (требование API)')
    args = parser.parse_args()

    session = requests.Session()
    session.headers['User-Agent'] = 'reservoir-simulator literature fetch (open access only)'
    missing = []
    for folder, name, doi, url in ARTICLES:
        target = HERE / folder / name
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f'{folder}/{name} ({doi or url})')
        ok = url is not None and _download(url, target, session)
        if not ok and doi:
            oa = _unpaywall(doi, args.email, session)
            ok = oa is not None and _download(oa, target, session)
        if not ok:
            missing.append((folder, name, doi))
            print('   открытой копии нет - в README статья числится в «найдены, но не скачаны»')

    print(f'\nНе скачано: {len(missing)} из {len(ARTICLES)}')
    sys.exit(0)


if __name__ == '__main__':
    main()
