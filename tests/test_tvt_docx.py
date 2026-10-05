"""Сборка статей для ТВТ (`твт_статья/make_docx.py`): простые формулы - текстом, сквозная нумерация полной версии."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'твт_статья'))
from make_docx import renumber, tvt_captions, tvt_text_math  # noqa: E402


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
