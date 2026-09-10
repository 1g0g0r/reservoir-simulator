"""Шаблон оформления docx под требования ТВТ для pandoc.

Pandoc берет шрифты, интервалы и поля из reference-документа. Скрипт берет дефолтный шаблон
pandoc и правит в нем ровно то, что требуют правила журнала (`PravilaTVT.pdf`):
Times New Roman 12 пт, полуторный интервал, поля 2.5 / 1.5 / 1.5 / 1.0 см.

Запускается один раз:  python твт_статья/make_reference_docx.py
"""
import re
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / 'reference.docx'

TWIPS_PER_CM = 567
MARGINS = {'left': 2.5, 'right': 1.5, 'top': 1.5, 'bottom': 1.0}  # см, правила ТВТ п. 4.7
FONT = 'Times New Roman'
HALF_POINTS = 24   # 12 пт
LINE_SPACING = 360  # полуторный интервал, в двадцатых долях пункта


def _cap_size(match: re.Match) -> str:
    """Кегль не крупнее 12 пт: правила требуют 12 пт, а в шаблоне заголовки до 28 пт."""
    tag, value = match.group(1), int(match.group(2))
    return f'<w:{tag} w:val="{min(value, HALF_POINTS)}" />'


def _patch_styles(xml: str) -> str:
    """Times New Roman 12 пт и полуторный интервал в стилях шаблона."""
    xml = re.sub(r'w:ascii="[^"]*" w:hAnsi="[^"]*"',
                 f'w:ascii="{FONT}" w:hAnsi="{FONT}"', xml)
    xml = re.sub(r'<w:(sz|szCs) w:val="(\d+)"\s*/>', _cap_size, xml)
    xml = re.sub(r'w:line="\d+"', f'w:line="{LINE_SPACING}"', xml)
    return xml


def _patch_margins(xml: str) -> str:
    """Поля страницы по правилам журнала.

    В шаблоне pandoc секция `w:sectPr` есть, а размеров страницы и полей в ней нет - Word
    подставляет свои умолчания. Дописываем их в начало секции.
    """
    values = {name: int(round(cm * TWIPS_PER_CM)) for name, cm in MARGINS.items()}
    page = ('<w:pgSz w:w="11906" w:h="16838" />'  # A4
            f'<w:pgMar w:top="{values["top"]}" w:right="{values["right"]}" '
            f'w:bottom="{values["bottom"]}" w:left="{values["left"]}" '
            'w:header="720" w:footer="720" w:gutter="0" />')
    return xml.replace('<w:sectPr>', f'<w:sectPr>{page}', 1)


def main() -> None:
    default = HERE / '_pandoc_default.docx'
    with open(default, 'wb') as f:
        subprocess.run(['pandoc', '--print-default-data-file', 'reference.docx'],
                       stdout=f, check=True)

    with zipfile.ZipFile(default) as src, zipfile.ZipFile(REFERENCE, 'w', zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            content = src.read(item.filename)
            if item.filename == 'word/styles.xml':
                content = _patch_styles(content.decode('utf-8')).encode('utf-8')
            elif item.filename == 'word/document.xml':
                content = _patch_margins(content.decode('utf-8')).encode('utf-8')
            dst.writestr(item, content)

    default.unlink()
    print(f'Шаблон записан: {REFERENCE}')


if __name__ == '__main__':
    main()
