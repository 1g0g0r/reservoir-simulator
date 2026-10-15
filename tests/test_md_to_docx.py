"""Сборка docx (`md_to_docx.py`): простые формулы - текстом, сквозная нумерация, формулы и шрифты для любой версии Word."""
import re
import subprocess
import zipfile
from pathlib import Path

import pytest
from lxml import etree

from md_to_docx import (_CITE, _expand, _fix_docx, _patch_paragraphs, _patch_styles, lint_docx, M_NS,
                        pandoc_path, renumber, tvt_captions, tvt_text_math, W_NS)

ROOT = Path(__file__).resolve().parent.parent


def test_simple_math_becomes_text():
    assert tvt_text_math('$k/k_0$') == '*k*/*k*~0~'
    assert tvt_text_math('$q_{ad}$, $\\eta$') == '*q*~ad~, η'  # сокращение из двух букв - прямо, греческие - прямо
    assert tvt_text_math('$\\Delta T_b = T - T_0$') == 'Δ*T*~*b*~ = *T* − *T*~0~'
    assert tvt_text_math('$a b$') == '*a* *b*'  # не *a**b*, что markdown прочел бы как полужирный


def test_complex_math_stays_formula():
    for tex in ('$w_k^s$', '$z_n \\propto e^{-sn}$', '$\\frac{a}{b}$', '$$x_1$$', 'цена $5 и $10'):
        assert tvt_text_math(tex) == tex


def test_renumber_by_first_mention():
    text = ('Сначала рис. 3 и табл. 2 [5], затем рис. 1–2 и табл. 1 [2, 3, 4].\n\n'
            'Таблица 2. Б\n\nТаблица 1. А\n\n# СПИСОК ЛИТЕРАТУРЫ\n\n'
            '1. Один.\n2. Два.\n3. Три.\n4. Четыре.\n5. Пять.\n')
    out = renumber(text)
    assert out.startswith('Сначала рис. 1 и табл. 1 [1], затем рис. 2–3 и табл. 2 [2–4].')
    assert 'Таблица 1. Б' in out and 'Таблица 2. А' in out
    assert out.endswith('1. Пять.\n2. Два.\n3. Три.\n4. Четыре.\n5. Один.\n')  # не цитированный - в конец


def test_full_version_captions_inline():
    text = ('Текст (рис. 2).\n\n![](f2.png)\n\nЕще (рис. 1).\n\n![](f1.png)\n\n'
            '# ПОДПИСИ К РИСУНКАМ\n\n![](f1.png)\n\nРис. 1. Первый.\n\n![](f2.png)\n\nРис. 2. Второй.\n')
    full = tvt_captions(text, inline=True)
    assert 'ПОДПИСИ' not in full
    assert full.index('![Рис. 1. Второй.](f2.png)') < full.index('![Рис. 2. Первый.](f1.png)')
    journal = tvt_captions(text)
    assert 'w:type="page"' in journal and journal.count('![](f1.png)') == 2


NS = f'xmlns:m="{M_NS}" xmlns:w="{W_NS}"'


def _docx(tmp_path, body: str, styles: str = '<w:styles/>') -> Path:
    path = tmp_path / 'a.docx'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('word/document.xml', f'<w:document {NS}><w:body>{body}</w:body></w:document>')
        z.writestr('word/styles.xml', f'<w:styles {NS}>{styles}</w:styles>')
    return path


def _document(path: Path) -> str:
    return zipfile.ZipFile(path).read('word/document.xml').decode('utf-8')


def test_fix_docx(tmp_path):
    """Пустой sepChr у скобок \\left(...\\right) телефоны показывают пустыми скобками - его не должно остаться;
    таблицы - с автоподбором ширины столбцов вместо фиксированной раскладки pandoc."""
    path = _docx(tmp_path, '<m:oMath><m:d><m:dPr><m:begChr m:val="(" /><m:endChr m:val=")" /><m:sepChr m:val="" />'
                 '<m:grow /></m:dPr><m:e><m:r><m:t>m</m:t></m:r></m:e></m:d></m:oMath>'
                 '<w:tbl><w:tblPr><w:tblW w:type="pct" w:w="5000" /><w:tblLayout w:type="fixed" /></w:tblPr></w:tbl>')
    _fix_docx(path)
    out = _document(path)
    assert 'sepChr' not in out
    assert re.search(r'<m:dPr><m:begChr m:val="\("/><m:endChr m:val="\)"/><m:grow/></m:dPr>', out)
    assert '<w:tblLayout w:type="autofit"/>' in out and 'fixed' not in out


def test_omml_schema_order_fixed(tmp_path):
    """pandoc 3.9 пишет в m:mcPr `mcJc` раньше `count`; Word 2007/2010 такую формулу не показывает."""
    path = _docx(tmp_path, '<m:oMath><m:m><m:mPr><m:mcs><m:mc><m:mcPr><m:mcJc m:val="left"/><m:count m:val="1"/>'
                 '</m:mcPr></m:mc></m:mcs></m:mPr><m:mr><m:e><m:r><m:t>x</m:t></m:r></m:e></m:mr></m:m></m:oMath>')
    assert any('порядок' in p for p in lint_docx(path))
    _fix_docx(path)
    assert '<m:mcPr><m:count m:val="1"/><m:mcJc m:val="left"/></m:mcPr>' in _document(path)
    assert not any('порядок' in p for p in lint_docx(path))


def test_math_runs_get_font_and_italic(tmp_path):
    """Каждый прогон формулы - со шрифтом Cambria Math (без него Word берет шрифт текста, буквы - квадраты);
    курсив - буквам в стиле по умолчанию; цифры, обычный текст (m:nor) и знаки - без курсива; запрещенные знаки заменены."""
    path = _docx(tmp_path, '<m:oMath><m:r><m:t>ρ</m:t></m:r><m:r><m:t>2</m:t></m:r>'
                 '<m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>∇⋅ℎ\u200b</m:t></m:r>'
                 '<m:r><m:rPr><m:nor/></m:rPr><m:t>вода</m:t></m:r>'
                 '<m:r><m:rPr><m:sty m:val="bi"/></m:rPr><m:t>U</m:t></m:r></m:oMath>')
    _fix_docx(path)
    runs = etree.fromstring(zipfile.ZipFile(path).read('word/document.xml')).findall(f'.//{{{M_NS}}}r')
    props = [(r.find(f'{{{W_NS}}}rPr/{{{W_NS}}}rFonts').get(f'{{{W_NS}}}ascii'),
              [etree.QName(e).localname for e in r.find(f'{{{W_NS}}}rPr')]) for r in runs]
    assert props == [('Cambria Math', ['rFonts', 'i']), ('Cambria Math', ['rFonts']), ('Cambria Math', ['rFonts']),
                     ('Times New Roman', ['rFonts']), ('Cambria Math', ['rFonts', 'b', 'i'])]
    assert ''.join(runs[2].itertext()) == '∇·h'
    assert lint_docx(path) == []


def test_styles_without_office_theme_fonts():
    """Шрифты темы Office (Aptos) перекрывают w:ascii: без них Word показывает текст без засечек."""
    out = _patch_styles('<w:rFonts w:asciiTheme="minorHAnsi" w:hAnsiTheme="minorHAnsi" w:cstheme="minorBidi" />'
                        '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>')
    assert 'Theme' not in out and 'cstheme' not in out and out.count('Times New Roman') == 8


def test_table_cells_without_first_line_indent():
    """Ячейки таблиц (стиль Compact, основан на BodyText) - без красной строки, обычный текст - с ней."""
    styles = ('<w:style w:type="paragraph" w:styleId="BodyText"><w:basedOn w:val="Normal" /><w:pPr>'
              '<w:spacing w:before="180" w:after="180" /></w:pPr></w:style>'
              '<w:style w:type="paragraph" w:customStyle="1" w:styleId="Compact"><w:basedOn w:val="BodyText" />'
              '<w:pPr><w:spacing w:before="36" w:after="36" /></w:pPr></w:style>')
    out = _patch_paragraphs(styles)
    body, compact = out.split('<w:style w:type="paragraph" w:customStyle="1" w:styleId="Compact">')
    assert 'w:firstLine="709"' in body
    assert compact.endswith('<w:spacing w:before="36" w:after="36" /><w:ind w:firstLine="0"/></w:pPr></w:style>')


def test_thermal_article_rules():
    """Статья для ТВТ (`статья_твт/article.md`, без перенумерации при сборке): источники - по порядку первого
    упоминания и все процитированы, каждая нумерованная формула упомянута в тексте (правила, разд. II.4, II.9)."""
    text = (ROOT / 'статья_твт' / 'article.md').read_text(encoding='utf-8')
    body, _, refs = text.partition('# СПИСОК ЛИТЕРАТУРЫ')
    seen = []
    for m in _CITE.finditer(body):
        seen += [n for n in _expand(m.group('nums')) if n not in seen]
    assert seen == list(range(1, len(seen) + 1)), f'порядок первого упоминания: {seen}'
    listed = [int(n) for n in re.findall(r'^(\d+)\.\s', refs.partition('\n# ')[0], re.M)]
    assert listed == seen, f'в списке {listed}, процитированы {seen}'
    labels = [int(n) for n in re.findall(r'\\qquad \((\d+)\)\$\$', body)]
    prose = re.sub(r'\$\$.*?\$\$', '', body, flags=re.S)
    for n in labels:
        assert re.search(rf'\({n}\)', prose), f'формула ({n}) не упомянута в тексте'


FORMULAS = r"""Дробь $\frac{\partial (m S_i)}{\partial t} + \nabla \cdot \mathbf{U}_i = q_i$, сумма и интеграл:

$$\sum_{i} \int_0^1 \frac{a_i}{b} \, dr \qquad (1)$$

$$\begin{aligned} \sqrt{x} &= \hat{y} \\ z &= \rho_w \end{aligned}$$
"""


def test_built_docx_passes_lint(tmp_path):
    """Сборка pandoc -> `_fix_docx`: дробь, сумма, интеграл, корень, акцент, aligned - без замечаний линтера."""
    pytest.importorskip('pypandoc')
    source = tmp_path / 'a.md'
    source.write_text(FORMULAS, encoding='utf-8')
    out = tmp_path / 'a.docx'
    subprocess.run([pandoc_path(), str(source), '-o', str(out)], check=True)
    _fix_docx(out)
    assert lint_docx(out) == [] or all('Aptos' in p for p in lint_docx(out))  # стили - не по шаблону ТВТ
    assert zipfile.ZipFile(out).read('word/document.xml').count(b'<m:oMath>') >= 3
