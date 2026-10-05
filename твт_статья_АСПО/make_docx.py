"""Сборка DOCX для ТВТ: временный reference -> pandoc -> удаление reference.

Скрипт:
1. Берёт дефолтный reference-шаблон pandoc.
2. Правит в нём шрифты, интервалы, поля и отступы под требования ТВТ.
3. Использует полученный reference для сборки итогового DOCX.
4. Удаляет временный reference.docx.
"""

import re
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

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


def _cap_size(match: re.Match) -> str:
    """Ограничивает кегль 12 пунктами.

    В шаблоне pandoc заголовки могут быть крупнее, например 28 pt,
    а по правилам журнала нужен 12 pt.
    """
    tag, value = match.group(1), int(match.group(2))
    return f'<w:{tag} w:val="{min(value, HALF_POINTS)}" />'


def _patch_styles(xml: str) -> str:
    """Times New Roman 12 pt в стилях шаблона."""
    xml = re.sub(
        r'w:ascii="[^"]*" w:hAnsi="[^"]*"',
        f'w:ascii="{FONT}" w:hAnsi="{FONT}"',
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
                    "pandoc",
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
    if inline:  # маркеры версий в полной версии - пустые комментарии, но мешают переставлять строки списков
        text = renumber(re.sub(r'<!-- /?полная -->\n?', '', text))
    if CAPTIONS not in text:
        return text
    head, _, tail = text.partition(CAPTIONS)
    if not inline:
        return head + _PAGE_BREAK + CAPTIONS + tail
    caps = re.findall(r'!\[\]\(([^)]+)\)\s*\n\s*\n(Рис\. \d+\..*?)\s*(?=\n\n|\n<!--|\Z)', tail, re.S)
    for path, cap in caps:
        target = f'![]({path})'
        # отдельным абзацем: рядом с маркерами <!-- полная --> pandoc не сделал бы из него рисунок с подписью
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


def _fix_docx(path: Path) -> None:
    """Правки готового docx, от которых зависит, как формулы выглядят в других программах и версиях Word.

    - Режим совместимости 15 и шрифт формул Cambria Math в settings.xml: без них Word открывает файл в режиме
      ограниченной функциональности, и разные версии верстают формулы по-разному. Порядок элементов в
      settings.xml задан схемой: compat - перед rsids, mathPr - сразу после.
    - В формулах оператор-точка U+22C5 и тильда U+223C заменяются на U+00B7 и '~': если Cambria Math нет
      (LibreOffice, WPS, просмотрщики), подставляется Times New Roman, а в нем этих знаков нет - они пропадают.
      ∇ и ∝ заменить нечем; если нужна полная переносимость - писать grad, div и «пропорционально».
    - Пустой разделитель <m:sepChr m:val=""/>, который pandoc пишет в каждую скобку \\left(...\\right), удаляется:
      просмотрщики на телефонах показывают с ним пустые скобки (Word его понимает). У скобок pandoc один аргумент,
      так что разделитель не нужен; заодно уходит нарушение порядка элементов m:dPr у pandoc 3.1 (sepChr после endChr).
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
                xml = re.sub(r'(<m:t(?: [^>]*)?>)([^<]*)(</m:t>)',
                             lambda m: m.group(1) + m.group(2).replace('⋅', '·').replace('∼', '~')
                             + m.group(3), xml)
                xml = re.sub(r'<m:sepChr m:val=""\s*/>', '', xml)
                xml = re.sub(r'<w:tblLayout w:type="fixed"\s*/>', '<w:tblLayout w:type="autofit"/>', xml)
                data = xml.encode('utf-8')
            dst.writestr(item, data)
    tmp.replace(path)


def build_docx(
    source_path: Path,
    output_path: Path,
    reference_path: Path,
) -> None:
    """Создаёт итоговый DOCX с использованием временного reference."""
    subprocess.run(
        [
            "pandoc",
            str(source_path),
            "-o",
            str(output_path),
            "--reference-doc",
            str(reference_path),
        ],
        check=True,
    )
    _fix_docx(output_path)


def main(short_docx: bool = False) -> None:
    if short_docx:
        source = 'article_short'
    else:
        source = 'article'
    source_path = Path.cwd() / f'{source}.md'
    output_path = Path.cwd() / f'{source}.docx'
    reference = Path.cwd() / 'reference.docx'
    work = Path.cwd() / f'_{source}_tvt.md'

    try:
        # 1. Генерируем временный reference.
        build_reference(reference)

        # 2. Простые формулы - текстом, подписи к рисункам - по правилам ТВТ; создаём итоговый DOCX.
        text = source_path.read_text(encoding='utf-8')
        work.write_text(tvt_captions(tvt_text_math(text)), encoding='utf-8')
        build_docx(work, output_path, reference)

    finally:
        # 3. Удаляем временные файлы даже при ошибке.
        for path in (reference, work):
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    main()
