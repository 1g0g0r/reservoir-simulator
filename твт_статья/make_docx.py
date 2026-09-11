"""Сборка DOCX для ТВТ: временный reference -> pandoc -> удаление reference.

Скрипт:
1. Берёт дефолтный reference-шаблон pandoc.
2. Правит в нём шрифты, интервалы, поля и отступы под требования ТВТ.
3. Использует полученный reference для сборки итогового DOCX.
4. Удаляет временный reference.docx.
"""

import argparse
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

# Интервал 1.15 в двадцатых долях пункта (240 = одинарный).
LINE_SPACING = 276

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


def main(source = 'article') -> None:
    source_path = Path.cwd() / f'{source}.md'
    output_path = Path.cwd() / f'{source}.docx'
    reference = Path.cwd() / 'reference.docx'

    try:
        # 1. Генерируем временный reference.
        build_reference(reference)

        # 2. Создаём итоговый DOCX.
        build_docx(source_path, output_path, reference)

    finally:
        # 3. Удаляем reference даже при ошибке.
        if reference.exists():
            reference.unlink()


if __name__ == "__main__":
    main()