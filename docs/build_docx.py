"""Сборка docx из исходников в `docs/`: pandoc переводит формулы TeX в формулы Word (OMML).

    python docs/build_docx.py              # оба документа
    python docs/build_docx.py валидация    # один: модель_АСПО или валидация_АСПО

`модель_АСПО.md` - математическое описание; вместо {{DEMO}} подставляется `demo_results.md`
(пишет `make_model_figures.py`). `валидация_АСПО.md` - сравнение с опытами; вместо {{КЛЮЧ}} подставляются
таблицы из `validation_tables.json` (пишет `validate.py`).

Стили (Times New Roman 12, выравнивание, поля) - те же, что у статьи: reference.docx собирается функциями
`твт_статья/make_docx.py`. pandoc берется из пакета `pypandoc_binary` (requirements.txt), если его нет в PATH.
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
TABLES = HERE / 'validation_tables.json'
DOCS = ('модель_АСПО', 'валидация_АСПО')

sys.path.insert(0, str(ROOT / 'твт_статья'))
from make_docx import build_reference, build_docx  # noqa: E402


def _pandoc_on_path() -> None:
    if shutil.which('pandoc'):
        return
    import pypandoc
    os.environ['PATH'] = str(Path(pypandoc.get_pandoc_path()).parent) + os.pathsep + os.environ.get('PATH', '')


def _substitutions(name: str) -> dict:
    if name == 'модель_АСПО':
        demo = DEMO.read_text(encoding='utf-8') if DEMO.is_file() else \
            '*Демонстрационный расчет не выполнен: `python demo_composition.py`, затем `python docs/make_model_figures.py`.*'
        return {'DEMO': demo.strip()}
    if not TABLES.is_file():
        raise SystemExit('нет docs/validation_tables.json: сначала python docs/validate.py')
    return json.loads(TABLES.read_text(encoding='utf-8'))


def build(name: str) -> None:
    source, target = HERE / f'{name}.md', HERE / f'{name}.docx'
    text = source.read_text(encoding='utf-8')
    for key, value in _substitutions(name).items():
        text = text.replace('{{' + key + '}}', value)
    missing = re.findall(r'\{\{[^}]+\}\}', text)
    if missing:
        raise SystemExit(f'{source.name}: не подставлено {missing}')
    work = HERE / '_build.md'
    reference = HERE / 'reference.docx'
    cwd = Path.cwd()
    try:
        work.write_text(text, encoding='utf-8')
        os.chdir(HERE)  # пути рисунков в тексте - относительно docs/
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
        build(name if name.endswith('_АСПО') else f'{name}_АСПО')


if __name__ == '__main__':
    main()
