"""Сборка статей для ТВТ (`твт_статья_АСПО/make_docx.py`): простые формулы - текстом, сквозная нумерация полной версии."""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'твт_статья_АСПО'))
from make_docx import _fix_docx, _patch_paragraphs, renumber, tvt_captions, tvt_text_math  # noqa: E402


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
    text = ('Текст (рис. 2).\n\n<!-- полная -->\n![](f2.png)\n<!-- /полная -->\n\nЕще (рис. 1).\n\n![](f1.png)\n\n'
            '# ПОДПИСИ К РИСУНКАМ\n\n![](f1.png)\n\nРис. 1. Первый.\n\n![](f2.png)\n\nРис. 2. Второй.\n')
    full = tvt_captions(text, inline=True)
    assert 'ПОДПИСИ' not in full
    assert full.index('![Рис. 1. Второй.](f2.png)') < full.index('![Рис. 2. Первый.](f1.png)')
    journal = tvt_captions(text)
    assert 'w:type="page"' in journal and journal.count('![](f1.png)') == 2


def test_fix_docx(tmp_path):
    """Пустой sepChr у скобок \\left(...\\right) телефоны показывают пустыми скобками - его не должно остаться;
    таблицы - с автоподбором ширины столбцов вместо фиксированной раскладки pandoc."""
    doc = ('<m:oMath><m:d><m:dPr><m:begChr m:val="(" /><m:endChr m:val=")" /><m:sepChr m:val="" /><m:grow />'
           '</m:dPr><m:e><m:r><m:t>m</m:t></m:r></m:e></m:d></m:oMath>'
           '<w:tbl><w:tblPr><w:tblW w:type="pct" w:w="5000" /><w:tblLayout w:type="fixed" /></w:tblPr></w:tbl>')
    path = tmp_path / 'a.docx'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('word/document.xml', doc)
    _fix_docx(path)
    out = zipfile.ZipFile(path).read('word/document.xml').decode('utf-8')
    assert 'sepChr' not in out
    assert '<m:dPr><m:begChr m:val="(" /><m:endChr m:val=")" /><m:grow /></m:dPr><m:e><m:r><m:t>m</m:t>' in out
    assert '<w:tblLayout w:type="autofit"/>' in out and 'fixed' not in out


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
