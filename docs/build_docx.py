"""Сборка docx из markdown-исходников: pandoc переводит формулы TeX в формулы Word (OMML).

    python docs/build_docx.py                       # все документы из DOCS
    python docs/build_docx.py модель_АСПО           # один или несколько по имени
    python docs/build_docx.py путь/к/файлу.md       # любой markdown рядом со своими рисунками

Документы (DOCS): описания моделей лежат в `docs/`, сравнения с опытами - в `experiments/`.
  - `docs/модель_АСПО.md` - математическое описание; вместо {{DEMO}} подставляется `demo_results.md`
    (пишет `make_model_figures.py`);
  - `docs/кинетика_осаждения.md` - кинетика выпадения и осаждения, сопутствующие модели кольматации;
  - `experiments/сравнение_с_опытами.md` - сравнение с опытами всех механизмов; вместо {{КЛЮЧ}} - таблицы из
    `experiments/results/tables.json` (пишет `experiments/run_all.py`);
  - `experiments/состав/валидация_АСПО.md` - первое сравнение детального состава; таблицы из
    `validation_tables.json` рядом (пишет `experiments/состав/validate.py`).
  - `твт_статья_АСПО/article.md` - статья для ТВТ; числа и таблицы - из `твт_статья_АСПО/results/numbers.json`
    (пишет `твт_статья_АСПО/make_article.py`). Две версии из одного текста: `статья_АСПО` - журнальная
    (`article.docx`, без фрагментов между `<!-- полная -->` и `<!-- /полная -->`), `статья_АСПО_полная` -
    без сокращенных формулировок между `<!-- журнальная -->` и `<!-- /журнальная -->` (`article_full.docx`).
    В обеих простые встроенные формулы набираются текстом (`make_docx.tvt_text_math`, правила ТВТ, разд. II.4).
    Журнальная: подписи к рисункам с новой страницы после списка литературы, рисунки и в тексте, и после подписей
    (правила, разд. II.1-3). Полная в журнал не идет: подписи - под рисунками в тексте, раздела в конце нет.

Стили (Times New Roman 12, выравнивание, поля) - те же, что у статьи: reference.docx собирается функциями
`твт_статья/make_docx.py`. pandoc берется из пакета `pypandoc_binary` (requirements.txt), если его нет в PATH.
Пути рисунков в тексте - относительно папки документа.
"""
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEMO = HERE / 'demo_results.md'


def _demo() -> dict:
    demo = DEMO.read_text(encoding='utf-8') if DEMO.is_file() else \
        '*Демонстрационный расчет не выполнен: `python demo_composition.py`, затем `python docs/make_model_figures.py`.*'
    return {'DEMO': demo.strip()}


def _tables(path: Path, how: str):
    def load() -> dict:
        if not path.is_file():
            raise SystemExit(f'нет {path.relative_to(ROOT)}: сначала {how}')
        return json.loads(path.read_text(encoding='utf-8'))
    return load


# имя -> (markdown-исходник, подстановки {{КЛЮЧ}})
DOCS = {
    'модель_АСПО': (HERE / 'модель_АСПО.md', _demo),
    'кинетика_осаждения': (HERE / 'кинетика_осаждения.md', dict),
    'сравнение_с_опытами': (ROOT / 'experiments' / 'сравнение_с_опытами.md',
                            _tables(ROOT / 'experiments' / 'results' / 'tables.json', 'python experiments/run_all.py')),
    'валидация_АСПО': (ROOT / 'experiments' / 'состав' / 'валидация_АСПО.md',
                       _tables(ROOT / 'experiments' / 'состав' / 'validation_tables.json',
                               'python experiments/состав/validate.py')),
}
_ARTICLE = (ROOT / 'твт_статья_АСПО' / 'article.md',
            _tables(ROOT / 'твт_статья_АСПО' / 'results' / 'numbers.json', 'python твт_статья_АСПО/make_article.py'))
DOCS['статья_АСПО'] = _ARTICLE + ('journal',)
DOCS['статья_АСПО_полная'] = _ARTICLE + ('full',)
# фрагменты только полной версии журнальная выбрасывает, а сокращенные формулировки только журнальной - полная
FULL_ONLY = re.compile(r'<!-- полная -->.*?<!-- /полная -->\n?', re.S)
JOURNAL_ONLY = re.compile(r'<!-- журнальная -->.*?<!-- /журнальная -->\n?', re.S)

sys.path.insert(0, str(ROOT / 'твт_статья'))
from make_docx import build_reference, build_docx, tvt_captions, tvt_text_math  # noqa: E402


def _pandoc_on_path() -> None:
    if shutil.which('pandoc'):
        return
    import pypandoc
    os.environ['PATH'] = str(Path(pypandoc.get_pandoc_path()).parent) + os.pathsep + os.environ.get('PATH', '')


def build(source: Path, substitutions=dict, version: str = None) -> None:
    folder = source.parent
    target = source.with_name(source.stem + '_full.docx') if version == 'full' else source.with_suffix('.docx')
    text = source.read_text(encoding='utf-8')
    if version:
        text = (FULL_ONLY if version == 'journal' else JOURNAL_ONLY).sub('', text)
    for key, value in substitutions().items():
        text = text.replace('{{' + key + '}}', value)
    missing = re.findall(r'\{\{[^}]+\}\}', text)
    if missing:
        raise SystemExit(f'{source.name}: не подставлено {missing}')
    if version:  # статья для ТВТ: простые формулы - текстом, подписи к рисункам - по правилам журнала
        text = tvt_captions(tvt_text_math(text), inline=version == 'full')
    work = folder / '_build.md'
    reference = folder / 'reference.docx'
    cwd = Path.cwd()
    try:
        work.write_text(text, encoding='utf-8')
        os.chdir(folder)  # пути рисунков в тексте - относительно папки документа
        build_reference(reference)
        build_docx(work, target, reference)
    finally:
        os.chdir(cwd)
        for path in (work, reference):
            if path.exists():
                path.unlink()

    with zipfile.ZipFile(target) as z:
        xml = z.read('word/document.xml').decode('utf-8')
        media = [n for n in z.namelist() if n.startswith('word/media/')]
    print(f'{target.name}: формул {xml.count("<m:oMath>")}, рисунков {len(media)}')


def main() -> None:
    _pandoc_on_path()
    for name in (sys.argv[1:] or DOCS):
        if name in DOCS:
            build(*DOCS[name])
        elif name.endswith('.md'):
            build(Path(name).resolve())
        else:
            raise SystemExit(f'неизвестный документ {name}: {", ".join(DOCS)} или путь к .md')


if __name__ == '__main__':
    main()
