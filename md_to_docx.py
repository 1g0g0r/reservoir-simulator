"""Сборка docx из markdown-исходников: pandoc переводит формулы TeX в формулы Word (OMML).

    python md_to_docx.py                       # все документы из DOCS
    python md_to_docx.py модель_АСПО           # один или несколько по имени
    python md_to_docx.py путь/к/файлу.md       # любой markdown рядом со своими рисунками
    python md_to_docx.py --lint файл.docx ...  # проверка формул и шрифтов готового docx (без Word)

Документы (DOCS): описания моделей лежат в `docs/`, сравнения с опытами - в `experiments/`, статьи - в `статья_твт/` и `статья_аспо/`.
  - `docs/модель_АСПО.md` - вместо {{DEMO}} подставляется `docs/demo_results.md` (пишет `docs/make_model_figures.py`);
  - `experiments/сравнение_с_опытами.md`, `experiments/состав/валидация_АСПО.md` - вместо {{КЛЮЧ}} таблицы из
    `tables.json` / `validation_tables.json` (пишут `experiments/run_all.py`, `experiments/состав/validate.py`);
  - `статья_твт/article.md` (журнал ТВТ, термические процессы) и `статья_аспо/article.md` (кристаллизация и
    осаждение); числа и таблицы - из `<статья>/results/numbers.json` (пишет `статья_аспо/make_article.py`).
    В обеих простые встроенные формулы набираются текстом (`tvt_text_math`, правила ТВТ, разд. II.4). Журнальная
    (`journal`): подписи к рисункам с новой страницы после списка литературы, рисунки и в тексте, и после подписей
    (правила, разд. II.1-3). Не для журнала (`paper`): подписи под рисунками, рисунки, таблицы и источники
    перенумерованы по первому упоминанию (`renumber`).

Стили (Times New Roman 12, выравнивание, поля) одни на все документы: reference.docx собирается здесь же
(`build_reference`). pandoc берется только из пакета `pypandoc_binary` (requirements.txt): версии pandoc по-разному
размечают формулы (3.9 нарушает порядок элементов m:mcPr, 3.1 пишет номер формулы объектом-матрицей), а сборка должна
быть одинаковой на любой машине. Пути рисунков в тексте - относительно папки документа.

Сборка формул под любую версию Word и другие программы (`_fix_docx`, `lint_docx`): порядок элементов OMML по схеме
ECMA-376 (Word 2007/2010 строги к нему), шрифт Cambria Math в каждом прогоне формулы (иначе Word берет шрифт текста,
и буквы - пустые квадраты), только шрифты Times New Roman вместо шрифтов темы Office (Aptos) в стилях.
"""
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parent

TWIPS_PER_CM = 567

# Поля по правилам ТВТ, п. 4.7: левое 2.5 см, правое 1.5 см, верхнее 1.5 см, нижнее 1.0 см.
MARGINS = {"left": 2.5, "right": 1.5, "top": 1.5, "bottom": 1.0}

FONT = "Times New Roman"

# Размер шрифта в половинных пунктах: 24 = 12 pt.
HALF_POINTS = 24

# Интервал в двадцатых долях пункта (240 = одинарный).
LINE_SPACING = 240  # 276 = 1.15

# Отступ первой строки, «красная строка».
INDENT_CM = 1.25


def pandoc_path() -> str:
    """pandoc из пакета `pypandoc_binary`; без него сборка не идет (системный pandoc размечает формулы иначе)."""
    try:
        import pypandoc
    except ImportError:
        raise SystemExit('нет пакета pypandoc_binary: uv pip install -r requirements.txt')
    return pypandoc.get_pandoc_path()


def _cap_size(match: re.Match) -> str:
    """Ограничивает кегль 12 пунктами.

    В шаблоне pandoc заголовки могут быть крупнее, например 28 pt,
    а по правилам журнала нужен 12 pt.
    """
    tag, value = match.group(1), int(match.group(2))
    return f'<w:{tag} w:val="{min(value, HALF_POINTS)}" />'


def _patch_styles(xml: str) -> str:
    """Times New Roman 12 pt в стилях шаблона.

    Шрифты темы (`w:asciiTheme` и др., у pandoc 3.9 это Aptos) имеют приоритет над `w:ascii`: оставленные в стилях,
    они дают в Word без Aptos текст без засечек. Поэтому каждый `w:rFonts` заменяется целиком.
    """
    xml = re.sub(
        r'<w:rFonts\b[^>]*?/>',
        f'<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:eastAsia="{FONT}" w:cs="{FONT}"/>',
        xml,
    )

    xml = re.sub(
        r'<w:(sz|szCs) w:val="(\d+)"\s*/>',
        _cap_size,
        xml,
    )

    return xml


def _patch_table_borders(xml: str) -> str:
    """Добавляет сплошные границы по всем ячейкам в стиль "Table".

    Pandoc ссылается на этот стиль из каждой таблицы (`w:tblStyle w:val="Table"`),
    а сам стиль по умолчанию задаёт только нижнюю границу первой строки — остальные
    границы рисует лишь Word при живом просмотре, в файле их нет.
    """
    borders = (
        "<w:tblBorders>"
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        "</w:tblBorders>"
    )

    return re.sub(
        r'(<w:style\b[^>]*w:styleId="Table"[^>]*>.*?<w:tblPr>)',
        r"\1" + borders,
        xml,
        count=1,
        flags=re.DOTALL,
    )


def _patch_paragraphs(xml: str) -> str:
    """Добавляет выравнивание по ширине, отступ первой строки и интервал."""
    indent_twips = int(round(INDENT_CM * TWIPS_PER_CM))

    # both = выравнивание по ширине,
    # firstLine = отступ первой строки.
    props = (
        '<w:jc w:val="both"/>'
        f'<w:ind w:firstLine="{indent_twips}"/>'
    )

    def inject_props(match: re.Match) -> str:
        """Вставляет свойства абзаца в нужный стиль."""
        style_block = match.group(0)

        if "<w:pPr/>" in style_block:
            style_block = style_block.replace(
                "<w:pPr/>",
                f"<w:pPr>{props}</w:pPr>",
                1,
            )
        elif "<w:pPr>" in style_block:
            style_block = style_block.replace(
                "<w:pPr>",
                f"<w:pPr>{props}",
                1,
            )

        # Полуторный интервал дописывается в уже существующий <w:spacing/>
        # (BodyText), а не отдельным элементом — второй <w:spacing> в одном
        # pPr неоднозначен для Word. В дефолтном шаблоне pandoc w:line там
        # нет вовсе (единственный w:line во всём styles.xml принадлежит
        # неиспользуемому TOCHeading), поэтому дописываем, а не заменяем.
        style_block = re.sub(
            r'<w:spacing\b([^>]*?)/>',
            lambda m: (
                '<w:spacing'
                + re.sub(r'\s*w:line(Rule)?="[^"]*"', '', m.group(1))
                + f' w:line="{LINE_SPACING}" w:lineRule="auto"/>'
            ),
            style_block,
        )

        return style_block

    # Правим стили, которые pandoc обычно использует для основного текста.
    xml = re.sub(
        r'<w:style\b[^>]*w:styleId="BodyText"[^>]*>.*?</w:style>',
        inject_props,
        xml,
        flags=re.DOTALL,
    )

    xml = re.sub(
        r'<w:style\b[^>]*w:styleId="FirstParagraph"[^>]*>.*?</w:style>',
        inject_props,
        xml,
        flags=re.DOTALL,
    )

    # Ячейки таблиц pandoc оформляет стилем Compact, а он основан на BodyText и унаследовал бы красную строку.
    # В pPr у Compact только <w:spacing/>, а ind по схеме идет после него - дописываем в конец.
    xml = re.sub(
        r'(<w:style\b[^>]*w:styleId="Compact"[^>]*>(?:(?!</w:style>).)*?)(</w:pPr>)',
        r'\1<w:ind w:firstLine="0"/>\2',
        xml,
        count=1,
        flags=re.DOTALL,
    )

    return xml


def _patch_margins(xml: str) -> str:
    """Устанавливает поля страницы и размер A4."""
    values = {
        name: int(round(cm * TWIPS_PER_CM))
        for name, cm in MARGINS.items()
    }

    page = (
        '<w:pgSz w:w="11906" w:h="16838" />'  # A4
        f'<w:pgMar w:top="{values["top"]}" '
        f'w:right="{values["right"]}" '
        f'w:bottom="{values["bottom"]}" '
        f'w:left="{values["left"]}" '
        'w:header="720" w:footer="720" w:gutter="0" />'
    )

    # Тег в дефолтном шаблоне pandoc самозакрывающийся: <w:sectPr />,
    # поэтому его надо не только раскрыть свойствами, но и закрыть самим.
    def inject_page(match: re.Match) -> str:
        return f"<w:sectPr>{page}</w:sectPr>" if match.group(1) else f"<w:sectPr>{page}"

    return re.sub(r"<w:sectPr\s*(/)?>", inject_page, xml, count=1)


def build_reference(reference_path: Path) -> None:
    """Создаёт reference.docx на основе дефолтного шаблона pandoc."""
    default_path = reference_path.with_name("_pandoc_default.docx")

    try:
        # Сохраняем дефолтный reference pandoc.
        with open(default_path, "wb") as f:
            subprocess.run(
                [
                    pandoc_path(),
                    "--print-default-data-file",
                    "reference.docx",
                ],
                stdout=f,
                check=True,
            )

        # Пересобираем DOCX как ZIP и правим нужные XML-файлы.
        with (
            zipfile.ZipFile(default_path) as src,
            zipfile.ZipFile(reference_path, "w", zipfile.ZIP_DEFLATED) as dst,
        ):
            for item in src.infolist():
                content = src.read(item.filename)

                if item.filename == "word/styles.xml":
                    xml_str = content.decode("utf-8")
                    xml_str = _patch_styles(xml_str)
                    xml_str = _patch_paragraphs(xml_str)
                    xml_str = _patch_table_borders(xml_str)
                    content = xml_str.encode("utf-8")

                elif item.filename == "word/document.xml":
                    xml_str = content.decode("utf-8")
                    xml_str = _patch_margins(xml_str)
                    content = xml_str.encode("utf-8")

                elif item.filename == "word/theme/theme1.xml":
                    # шрифты темы (Aptos) - на Times New Roman: на них ссылаются стили, не попавшие под правку
                    xml_str = re.sub(r'(<a:latin typeface=")[^"]*(")', rf'\g<1>{FONT}\g<2>',
                                     content.decode("utf-8"))
                    content = xml_str.encode("utf-8")

                dst.writestr(item, content)

    finally:
        # Временный дефолтный шаблон удаляем всегда.
        if default_path.exists():
            default_path.unlink()


# --- Формулы по правилам ТВТ (разд. II.4) -------------------------------------------------------------------
# Одиночные буквы (латинские курсивом, греческие прямо), обозначения только с верхним или только с нижним индексом
# и простые формулы набираются текстом, а не в редакторе формул. pandoc же переводит в формулу Word всякое $...$,
# поэтому простые встроенные формулы переводятся в markdown-текст до pandoc: *k*~0~ вместо $k_0$. Все, что
# конвертер не понимает (дроби, корни, интегралы, nabla, оба индекса сразу), остается формулой.
GREEK = {'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ϵ', 'varepsilon': 'ε', 'zeta': 'ζ',
         'eta': 'η', 'theta': 'θ', 'vartheta': 'ϑ', 'iota': 'ι', 'kappa': 'κ', 'lambda': 'λ', 'mu': 'μ', 'nu': 'ν',
         'xi': 'ξ', 'pi': 'π', 'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ', 'phi': 'ϕ', 'varphi': 'φ',
         'chi': 'χ', 'psi': 'ψ', 'omega': 'ω', 'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ', 'Xi': 'Ξ',
         'Pi': 'Π', 'Sigma': 'Σ', 'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω'}
# Только знаки, которые есть в Times New Roman: ∝, ∇, ⋅, ∼ в нем нет, и формулы с ними остаются формулами
SYMBOLS = {'ldots': '…', 'cdots': '…', 'le': ' ≤ ', 'leq': ' ≤ ', 'ge': ' ≥ ', 'geq': ' ≥ ', 'approx': ' ≈ ',
           'times': ' × ', 'pm': '±', 'infty': '∞', 'to': ' → ', 'sim': ' \\~ ', ',': ' ', ';': ' ', ' ': ' ',
           'quad': ' ', '%': '%'}
FUNCTIONS = {'ln', 'lg', 'exp', 'max', 'min', 'log'}
_ESCAPE = {'[': '\\[', ']': '\\]', '<': '\\<', '>': '\\>', '*': '\\*'}


class _NotSimple(Exception):
    pass


def _group(tex, i):
    """Содержимое {...} с позиции i (на '{') и позиция за '}'; без скобок - один символ или команда."""
    if tex[i] == '{':
        depth = 0
        for j in range(i, len(tex)):
            depth += {'{': 1, '}': -1}.get(tex[j], 0)
            if depth == 0:
                return tex[i + 1:j], j + 1
        raise _NotSimple
    if tex[i] == '\\':
        m = re.match(r'\\([A-Za-z]+|.)', tex[i:])
        return m.group(0), i + len(m.group(0))
    return tex[i], i + 1


def _md(tex, index=False):
    """Markdown-текст простой формулы; index - внутри индекса (пробелы экранируются, вложенных индексов нет)."""
    out, i = [], 0
    space = '\\ ' if index else ' '
    while i < len(tex):
        c = tex[i]
        if c.isspace():
            i += 1
            continue
        if c in '_^':  # индекс без основы
            raise _NotSimple
        if c.isascii() and c.isalpha():
            j = i
            while j < len(tex) and tex[j].isascii() and tex[j].isalpha():
                j += 1
            word = tex[i:j]
            # в индексе сокращение из двух и более букв - прямо, из одной - курсивом (правила ТВТ, разд. II.4)
            atom = word if index and len(word) > 1 else f'*{word}*'
            i = j
        elif c.isdigit() or c == '.':
            j = i
            while j < len(tex) and (tex[j].isdigit() or tex[j] == '.'):
                j += 1
            atom, i = tex[i:j], j
        elif c == ',':
            atom, i = (',' if index else ', '), i + 1
        elif c == '-':  # бинарный минус - с пробелами, унарный и в индексе - без
            prev = ''.join(out).rstrip()
            atom = ' − ' if prev and not index and prev[-1] not in '(=+−/,' else '−'
            i += 1
        elif c in '+=':
            atom, i = (c if index else f' {c} '), i + 1
        elif c in '/()|!\'':
            atom, i = c, i + 1
        elif c == '~':  # неразрывный пробел TeX
            atom, i = ' ', i + 1
        elif c in _ESCAPE:
            atom, i = _ESCAPE[c], i + 1
        elif c == '{':
            if tex.startswith('{{', i):  # неподставленный шаблон {{КЛЮЧ}}
                raise _NotSimple
            inner, i = _group(tex, i)
            atom = _md(inner, index)
        elif c == '\\':
            cmd, i = _group(tex, i)
            name = cmd[1:]
            if name in GREEK:
                atom = GREEK[name]
            elif name in SYMBOLS:
                atom = SYMBOLS[name].strip() if index else SYMBOLS[name]
            elif name in FUNCTIONS:
                atom = name
            elif name in ('mathrm', 'text', 'operatorname', 'mathbf'):
                inner, i = _group(tex, i)
                if not re.fullmatch(r'[A-Za-zА-Яа-я0-9 .,\-]+', inner):
                    raise _NotSimple
                inner = inner.replace(' ', space)
                atom = f'**{inner}**' if name == 'mathbf' else inner
            else:
                raise _NotSimple
        else:
            raise _NotSimple
        # индекс атома: только верхний или только нижний (правила ТВТ); оба сразу - в редакторе формул
        if i < len(tex) and tex[i] in '_^':
            if index:
                raise _NotSimple
            mark = tex[i]
            arg, i = _group(tex, i + 1)
            if i < len(tex) and tex[i] in '_^':
                raise _NotSimple
            sub = _md(arg, index=True)
            atom += f'~{sub}~' if mark == '_' else f'^{sub}^'
        if out and out[-1].endswith('*') and atom.startswith('*'):
            out.append(' ')  # *a**b* markdown прочтет как полужирный
        out.append(atom)
    return ''.join(out).strip()


def tvt_text_math(text: str) -> str:
    """Простые встроенные формулы $...$ - в текст по правилам ТВТ; выносные $$...$$ не трогаются."""
    inline = re.compile(r'(?<![\\$])\$(?=\S)([^$\n]+?)(?<=\S)\$(?!\d)')

    def convert(m):
        try:
            md = _md(m.group(1))
        except (_NotSimple, IndexError):
            return m.group(0)
        return md or m.group(0)

    parts = text.split('$$')
    return '$$'.join(inline.sub(convert, p) if k % 2 == 0 else p for k, p in enumerate(parts))


CAPTIONS = '# ПОДПИСИ К РИСУНКАМ'
_PAGE_BREAK = '```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n\n'


def tvt_captions(text: str, inline: bool = False) -> str:
    """Подписи к рисункам. Для журнала (правила ТВТ, разд. II.1-3) - на отдельной странице после списка
    литературы, рисунки - и по месту в тексте, и после подписей. inline - для версии не в журнал: подпись
    переезжает под рисунок в тексте, раздела в конце нет."""
    if inline:
        text = renumber(text)
    if CAPTIONS not in text:
        return text
    head, _, tail = text.partition(CAPTIONS)
    if not inline:
        return head + _PAGE_BREAK + CAPTIONS + tail
    caps = re.findall(r'!\[\]\(([^)]+)\)\s*\n\s*\n(Рис\. \d+\..*?)\s*(?=\n\n|\n<!--|\Z)', tail, re.S)
    for path, cap in caps:
        target = f'![]({path})'
        # отдельным абзацем: иначе pandoc не сделал бы из него рисунок с подписью
        figure = f'\n\n![{cap}]({path})\n\n'
        head = head.replace(target, figure) if target in head else head + figure
    return head.rstrip() + '\n'


# --- Сквозная нумерация версии не для журнала ----------------------------------------------------------------
# Блоки только полной версии добавляют рисунки, таблицы и источники, а номера в общем тексте заданы журнальной.
# Полная версия перенумеровывает их по порядку первого упоминания (правила ТВТ, разд. II.7-9 - то же требование).
_FIG = re.compile(r'(?P<label>[Рр]ис\.\s*)(?P<nums>\d+[а-я]?(?:\s*(?:,|–|-)\s*\d+[а-я]?)*)')
_TAB = re.compile(r'(?P<label>[Тт]абл\.\s*|Таблица\s+)(?P<nums>\d+(?:\s*(?:,|–|-)\s*\d+)*)')
_CITE = re.compile(r'(?<![\\!\w])\[(?P<nums>\d+(?:\s*(?:,|–|-)\s*\d+)*)\]')
_SEP = re.compile(r'(\s*(?:,|–|-)\s*)')


def _expand(nums: str):
    """'1–3, 5а' -> [1, 2, 3, 5] (буквы подрисунков отбрасываются)."""
    parts = _SEP.split(nums)
    out = [int(re.match(r'\d+', parts[0]).group())]
    for sep, tok in zip(parts[1::2], parts[2::2]):
        n = int(re.match(r'\d+', tok).group())
        out += list(range(out[-1] + 1, n + 1)) if sep.strip() in '–-' else [n]
    return out


def _order(text: str, rx) -> dict:
    """Старый номер -> новый по порядку первого упоминания."""
    seen = []
    for m in rx.finditer(text):
        seen += [n for n in _expand(m.group('nums')) if n not in seen]
    return {old: new for new, old in enumerate(seen, 1)}


def _remap_simple(m, mapping):
    """Номера с буквами подрисунков и разделителями как есть, только сами числа - по mapping."""
    parts = _SEP.split(m.group('nums'))
    out = [re.sub(r'\d+', lambda d: str(mapping.get(int(d.group()), int(d.group()))), p) if k % 2 == 0 else p
           for k, p in enumerate(parts)]
    return m.group('label') + ''.join(out)


def _compress(nums):
    """[9, 10, 11, 14] -> '9–11, 14'."""
    nums, out, i = sorted(set(nums)), [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f'{nums[i]}–{nums[j]}' if j - i >= 2 else ', '.join(map(str, nums[i:j + 1])))
        i = j + 1
    return ', '.join(out)


def renumber(text: str) -> str:
    """Рисунки, таблицы и источники - по порядку первого упоминания; список литературы - в новом порядке."""
    refs_head = '# СПИСОК ЛИТЕРАТУРЫ'
    body, _, refs = text.partition(refs_head)
    for rx in (_FIG, _TAB):
        mapping = _order(body, rx)
        text = rx.sub(lambda m: _remap_simple(m, mapping), text)
    body, _, refs = text.partition(refs_head)
    mapping = _order(body, _CITE)
    if not refs or not mapping:
        return text
    body = _CITE.sub(lambda m: '[' + _compress(mapping.get(n, n) for n in _expand(m.group('nums'))) + ']', body)
    head, nl, tail = refs.partition('\n#')  # список - до следующего заголовка
    items = re.findall(r'^(\d+)\.\s+(.*)$', head, re.M)
    rest = [n for n, _ in items if int(n) not in mapping]
    mapping.update({int(n): len(mapping) + k for k, n in enumerate(rest, 1)})
    lines = sorted((mapping[int(n)], t) for n, t in items)
    first = re.search(r'^\d+\.\s', head, re.M)
    intro = head[:first.start()] if first else head
    new_head = intro + '\n'.join(f'{n}. {t}' for n, t in lines) + '\n'
    return body + refs_head + new_head + (nl + tail if nl else '')


# --- Формулы Word (OMML): порядок элементов, шрифт, символы ----------------------------------------------------
M_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
MATH_FONT = 'Cambria Math'

# Порядок дочерних элементов по ECMA-376: Word 2007/2010 и часть сборок строги к нему - формула с нарушением
# пустая или файл «нечитаемый». pandoc 3.9, например, пишет в m:mcPr `mcJc` раньше `count`.
OMML_ORDER = {
    'oMathParaPr': ['jc'], 'fPr': ['type', 'ctrlPr'], 'dPr': ['begChr', 'sepChr', 'endChr', 'grow', 'shp', 'ctrlPr'],
    'naryPr': ['chr', 'limLoc', 'grow', 'subHide', 'supHide', 'ctrlPr'], 'accPr': ['chr', 'ctrlPr'],
    'radPr': ['degHide', 'ctrlPr'], 'sSubPr': ['ctrlPr'], 'sSupPr': ['ctrlPr'], 'sSubSupPr': ['alnScr', 'ctrlPr'],
    'eqArrPr': ['baseJc', 'maxDist', 'objDist', 'rSpRule', 'rSp', 'ctrlPr'],
    'mPr': ['baseJc', 'plcHide', 'rSpRule', 'rSp', 'cSp', 'cGpRule', 'cGp', 'mcs', 'ctrlPr'],
    'mcPr': ['count', 'mcJc'], 'rPr': ['lit', 'nor', 'scr', 'sty', 'brk', 'aln'],
    'f': ['fPr', 'num', 'den'], 'rad': ['radPr', 'deg', 'e'], 'nary': ['naryPr', 'sub', 'sup', 'e'],
    'sSub': ['sSubPr', 'e', 'sub'], 'sSup': ['sSupPr', 'e', 'sup'], 'sSubSup': ['sSubSupPr', 'e', 'sub', 'sup'],
    'acc': ['accPr', 'e'], 'd': ['dPr', 'e'], 'eqArr': ['eqArrPr', 'e'], 'm': ['mPr', 'mr'],
    'func': ['funcPr', 'fName', 'e'], 'limLow': ['limLowPr', 'e', 'lim'], 'limUpp': ['limUppPr', 'e', 'lim'],
}
# Замены знаков, которых нет в Cambria Math/Times New Roman, и невидимых символов
MATH_CHARS = {'⋅': '·', '∼': '~', 'ℎ': 'h', '​': ''}
_BAD_CHARS = re.compile('[\U0001d400-\U0001d7ffℎ​⋅∼]')


def _m(name: str) -> str:
    return f'{{{M_NS}}}{name}'


def _w(name: str) -> str:
    return f'{{{W_NS}}}{name}'


def _order_key(el, order):
    name = etree.QName(el).localname if etree.QName(el).namespace == M_NS else None
    return order.index(name) if name in order else len(order)


def _wrong_order(el) -> bool:
    order = OMML_ORDER[etree.QName(el).localname]
    keys = [_order_key(k, order) for k in el]
    return keys != sorted(keys)


def _math_run_props(run) -> None:
    """Шрифт и начертание прогона формулы так, как их пишет сам Word: `<w:rPr><w:rFonts Cambria Math/><w:i/></w:rPr>`.

    Без явного шрифта Word берет шрифт текста, и буквы формулы (символы U+1D4xx «математический курсив») - пустые
    квадраты. Курсив ставят только прогонам с буквами в стиле по умолчанию (цифры и знаки Word не наклоняет);
    `m:nor` (обычный текст, например кириллица) - шрифтом документа."""
    rpr = run.find(_m('rPr'))
    style = rpr.find(_m('sty')).get(_m('val')) if rpr is not None and rpr.find(_m('sty')) is not None else None
    normal = rpr is not None and rpr.find(_m('nor')) is not None
    text = ''.join(t.text or '' for t in run.iter(_m('t')))
    wrpr = run.find(_w('rPr'))
    if wrpr is None:
        wrpr = etree.Element(_w('rPr'))
        run.insert(0 if rpr is None else 1, wrpr)
    for child in list(wrpr):
        if child.tag in (_w('rFonts'), _w('i'), _w('b')):
            wrpr.remove(child)
    new = [etree.Element(_w('rFonts'))]  # по схеме rFonts - первым, затем b, i
    for attr in ('ascii', 'hAnsi'):
        new[0].set(_w(attr), FONT if normal else MATH_FONT)
    if style in ('b', 'bi'):
        new.append(etree.Element(_w('b')))
    if not normal and style in (None, 'i', 'bi') and any(c.isalpha() for c in text):
        new.append(etree.Element(_w('i')))
    for k, el in enumerate(new):
        wrpr.insert(k, el)


def _fix_omml(xml: bytes) -> bytes:
    root = etree.fromstring(xml)
    for el in root.iter(*[_m(n) for n in OMML_ORDER]):
        order = OMML_ORDER[etree.QName(el).localname]
        if _wrong_order(el):
            kids = sorted(el, key=lambda k: _order_key(k, order))  # sorted устойчива: повторы e, mr, mc остаются
            for k in kids:
                el.append(k)
    for t in root.iter(_m('t')):
        if t.text:
            t.text = ''.join(MATH_CHARS.get(c, c) for c in t.text)
    for run in root.iter(_m('r')):
        _math_run_props(run)
    for ctrl in root.iter(_m('ctrlPr')):  # знак-заполнитель: шрифт формул, как у Word
        wrpr = ctrl.find(_w('rPr'))
        if wrpr is not None and wrpr.find(_w('rFonts')) is None:
            font = etree.Element(_w('rFonts'))
            for attr in ('ascii', 'hAnsi'):
                font.set(_w(attr), MATH_FONT)
            wrpr.insert(0, font)
    return etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)


def _fix_docx(path: Path) -> None:
    """Правки готового docx, от которых зависит, как формулы выглядят в других программах и версиях Word.

    - Режим совместимости 15 и шрифт формул Cambria Math в settings.xml: без них Word открывает файл в режиме
      ограниченной функциональности, и разные версии верстают формулы по-разному. Порядок элементов в
      settings.xml задан схемой: compat - перед rsids, mathPr - сразу после.
    - Формулы (`_fix_omml`): порядок дочерних элементов по схеме, шрифт Cambria Math и курсив в прогонах, замена
      знаков, которых нет в шрифтах (⋅, ∼, ℎ, нулевой пробел). ∇ и ∝ есть только в Cambria Math: заменить нечем.
    - Пустой разделитель <m:sepChr m:val=""/>, который pandoc пишет в каждую скобку \\left(...\\right), удаляется:
      просмотрщики на телефонах показывают с ним пустые скобки (Word его понимает). У скобок pandoc один аргумент,
      так что разделитель не нужен.
    - Таблицы - с автоподбором ширины столбцов по содержимому: pandoc ставит фиксированную раскладку с равными
      столбцами. Ширина таблицы остается во всю строку (tblW 100%). Автоподбор выполняет Word при открытии;
      программы, которые его не умеют, покажут прежние равные столбцы.
    """
    tmp = path.with_name(path.stem + '_fix.docx')
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == 'word/settings.xml':
                xml = data.decode('utf-8')
                if 'compatibilityMode' not in xml and '<w:rsids' in xml:
                    xml = xml.replace('<w:rsids', '<w:compat><w:compatSetting w:name="compatibilityMode" '
                                      'w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat>'
                                      '<w:rsids', 1)
                if '<m:mathPr' not in xml and '</w:rsids>' in xml and 'xmlns:m=' in xml:
                    xml = xml.replace('</w:rsids>', '</w:rsids><m:mathPr><m:mathFont m:val="Cambria Math"/>'
                                      '</m:mathPr>', 1)
                data = xml.encode('utf-8')
            elif item.filename == 'word/document.xml':
                xml = data.decode('utf-8')
                xml = re.sub(r'<m:sepChr m:val=""\s*/>', '', xml)
                xml = re.sub(r'<w:tblLayout w:type="fixed"\s*/>', '<w:tblLayout w:type="autofit"/>', xml)
                data = _fix_omml(xml.encode('utf-8'))
            dst.writestr(item, data)
    tmp.replace(path)


def lint_docx(path: Path) -> list:
    """Проблемы готового docx, из-за которых формулы или текст не отобразятся в части версий Word и программ."""
    problems = []
    with zipfile.ZipFile(path) as z:
        document = etree.fromstring(z.read('word/document.xml'))
        styles = z.read('word/styles.xml').decode('utf-8')
    wrong = {}
    for el in document.iter(*[_m(n) for n in OMML_ORDER]):
        if _wrong_order(el):
            key = etree.QName(el).localname + ': ' + ', '.join(etree.QName(k).localname for k in el)
            wrong[key] = wrong.get(key, 0) + 1
    problems += [f'нарушен порядок элементов OMML, {n} раз - {key}' for key, n in wrong.items()]
    runs = list(document.iter(_m('r')))
    no_font = [r for r in runs if r.find(f'{_w("rPr")}/{_w("rFonts")}') is None]
    if no_font:
        problems.append(f'{len(no_font)} из {len(runs)} прогонов формул без шрифта (w:rFonts)')
    bad = {c for t in document.iter(_m('t')) for c in (t.text or '') if _BAD_CHARS.match(c)}
    if bad:
        problems.append('знаки вне шрифтов Word: ' + ', '.join(f'U+{ord(c):04X}' for c in sorted(bad)))
    if re.search(r'w:(ascii|hAnsi|eastAsia|cs)Theme=', styles):
        problems.append('в styles.xml остались шрифты темы Office (Aptos вместо Times New Roman)')
    return problems


# --- Сборка --------------------------------------------------------------------------------------------------
def build_docx(
    source_path: Path,
    output_path: Path,
    reference_path: Path,
) -> None:
    """Создаёт итоговый DOCX с использованием временного reference."""
    subprocess.run(
        [
            pandoc_path(),
            str(source_path),
            "-o",
            str(output_path),
            "--reference-doc",
            str(reference_path),
        ],
        check=True,
    )
    _fix_docx(output_path)


def _demo() -> dict:
    path = ROOT / 'docs' / 'demo_results.md'
    demo = path.read_text(encoding='utf-8') if path.is_file() else \
        '*Демонстрационный расчет не выполнен: `python demo_composition.py`, затем `python docs/make_model_figures.py`.*'
    return {'DEMO': demo.strip()}


def _tables(path: Path, how: str):
    def load() -> dict:
        if not path.is_file():
            raise SystemExit(f'нет {path.relative_to(ROOT)}: сначала {how}')
        return json.loads(path.read_text(encoding='utf-8'))
    return load


def _numbers(article: str):
    return _tables(ROOT / article / 'results' / 'numbers.json', 'python статья_аспо/make_article.py')


# имя -> (markdown-исходник, подстановки {{КЛЮЧ}}[, версия: 'journal' - подписи к рисункам после литературы, правила
# ТВТ; 'paper' - подписи под рисунками, сквозная нумерация])
DOCS = {
    'физическая_модель': (ROOT / 'docs' / 'физическая_модель.md', dict),
    'метод_решения': (ROOT / 'docs' / 'метод_решения.md', dict),
    'модель_АСПО': (ROOT / 'docs' / 'модель_АСПО.md', _demo),
    'кинетика_осаждения': (ROOT / 'docs' / 'кинетика_осаждения.md', dict),
    'сравнение_с_опытами': (ROOT / 'experiments' / 'сравнение_с_опытами.md',
                            _tables(ROOT / 'experiments' / 'results' / 'tables.json', 'python experiments/run_all.py')),
    'валидация_АСПО': (ROOT / 'experiments' / 'состав' / 'валидация_АСПО.md',
                       _tables(ROOT / 'experiments' / 'состав' / 'validation_tables.json',
                               'python experiments/состав/validate.py')),
    'статья_твт': (ROOT / 'статья_твт' / 'article.md', _numbers('статья_твт'), 'journal'),
    'статья_аспо': (ROOT / 'статья_аспо' / 'article.md', _numbers('статья_аспо'), 'paper'),
}


def build(source: Path, substitutions=dict, version: str = None) -> None:
    folder = source.parent
    target = source.with_suffix('.docx')
    text = source.read_text(encoding='utf-8')
    for key, value in substitutions().items():
        text = text.replace('{{' + key + '}}', value)
    missing = re.findall(r'\{\{[^}]+\}\}', text)
    if missing:
        raise SystemExit(f'{source.name}: не подставлено {missing}')
    if version:  # статья для ТВТ: простые формулы - текстом, подписи к рисункам - по правилам журнала
        text = tvt_captions(tvt_text_math(text), inline=version == 'paper')
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
    for problem in lint_docx(target):
        print(f'  ВНИМАНИЕ: {problem}')


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ['--lint']:
        for name in args[1:]:
            problems = lint_docx(Path(name))
            print(f'{name}: ' + ('замечаний нет' if not problems else ''))
            for problem in problems:
                print(f'  - {problem}')
        return
    for name in args or DOCS:
        if name in DOCS:
            build(*DOCS[name])
        elif name.endswith('.md'):
            build(Path(name).resolve())
        else:
            raise SystemExit(f'неизвестный документ {name}: {", ".join(DOCS)} или путь к .md')


if __name__ == '__main__':
    main()
