"""Сборка literature_review.docx оформлением статьи (твт_статья_АСПО/make_docx.py: Times New Roman 12, поля,
выравнивание по ширине, формулы Word, таблицы с границами).

    python tools/to_docx.py

Схему приложения D (блок mermaid) Word не показывает, поэтому она рисуется здесь matplotlib из того же текста
и встает в docx рисунком. Раскладка ручная: координаты узлов - POS (см), обходы линий - VIA, подписи пунктирных
связей - LABEL. Узел без позиции в POS - ошибка с его именем.
"""
import math
import os
import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

REVIEW = Path(__file__).resolve().parents[1]
ROOT = REVIEW.parent
sys.path.insert(0, str(ROOT / 'твт_статья_АСПО'))
from make_docx import build_docx, build_reference  # noqa: E402

W, H = 17.0, 22.0      # см: ширина рисунка - ширина текста страницы статьи
NW, NH = 2.4, 1.25     # узел
PAD, TITLE = .35, .4   # поля группы и место под ее название
POS = {
    'C': (2.6, 18.6), 'E': (6.4, 18.6), 'P': (10.2, 18.6), 'W': (3.4, 16.7), 'A': (9.4, 16.7),
    'K1': (5.8, 13.4), 'K2': (8.4, 13.4), 'K3': (11.0, 13.4),
    'G': (2.6, 10.0), 'D1': (5.8, 10.0), 'D2': (8.6, 10.9), 'D3': (8.6, 9.1), 'D4': (11.4, 10.0),
    'F2': (2.8, 6.4), 'F1': (8.6, 6.4), 'F5': (11.6, 6.4), 'F4': (2.8, 4.8), 'F3': (5.6, 4.8),
    'N1': (3.4, 1.9), 'N2': (7.0, 1.9), 'N3': (10.6, 1.9),
    'X1': (15.4, 17.9), 'X2': (15.4, 10.9), 'X3': (15.4, 9.1), 'X4': (15.4, 5.6),
    'ML': (6.4, 21.2), 'TR': (15.4, 21.2),
}
VIA = {  # обратные связи и дальние ссылки - в обход блоков
    ('F2', 'W'): [(0.45, 6.4), (0.45, 16.7)],
    ('F4', 'K1'): [(0.8, 4.8), (0.8, 13.4)],
    ('ML', 'NM'): [(0.15, 21.2), (0.15, 2.1)],
    ('TR', 'KN'): [(13.3, 21.2), (13.3, 13.6)],
    ('X3', 'D1'): [(13.4, 8.35), (5.8, 8.35)],
}
LABEL = {('TR', 'KN'): (13.3, 19.6, 90), ('ML', 'TD'): (6.6, 20.27, 0), ('ML', 'NM'): (0.15, 11.0, 90)}

ARROW = re.compile(r'\s*(-->|-\..*?\.->)\s*')
NODE = re.compile(r'(\w+)\s*\[([^\]]+)\]')


def parse(src: str):
    """Узлы, группы и связи из flowchart mermaid: A[метка], subgraph G[название] ... end, A --> B, A -. метка .-> B."""
    labels, groups, edges, cur = {}, {}, [], None
    for line in src.splitlines():
        line = line.strip()
        if m := re.match(r'subgraph (\w+)\[(.+)\]$', line):
            cur = m[1]
            groups[cur] = (m[2], [])
            continue
        if line == 'end':
            cur = None
            continue
        for m in NODE.finditer(line):
            labels[m[1]] = m[2].strip('"')
            if cur:
                groups[cur][1].append(m[1])
        parts = ARROW.split(line)
        for a, arrow, b in zip(parts[0::2], parts[1::2], parts[2::2]):
            edges.append((re.match(r'\w+', a)[0], re.match(r'\w+', b)[0], arrow != '-->' and arrow[2:-3].strip()))
    return labels, groups, edges


def clip(r, tx, ty):
    """Точка на границе прямоугольника r = (x, y, полуширина, полувысота) в сторону (tx, ty)."""
    x, y, hw, hh = r
    dx, dy = tx - x, ty - y
    t = min(hw / abs(dx) if dx else math.inf, hh / abs(dy) if dy else math.inf)
    return x + t * dx, y + t * dy


def draw_map(src: str, path: Path) -> None:
    labels, groups, edges = parse(src)
    missing = sorted(set(labels) - set(POS))
    if missing:
        raise SystemExit(f'нет позиции узлов {missing}: добавь в POS ({Path(__file__).name})')
    plt.rcParams['font.family'] = 'Times New Roman'
    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    box = {n: (*POS[n], NW / 2, NH / 2) for n in labels}
    for g, (title, members) in groups.items():
        x0 = min(POS[n][0] for n in members) - NW / 2 - PAD
        x1 = max(POS[n][0] for n in members) + NW / 2 + PAD
        y0 = min(POS[n][1] for n in members) - NH / 2 - PAD
        y1 = max(POS[n][1] for n in members) + NH / 2 + PAD + TITLE
        box[g] = ((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2)
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0,rounding_size=0.2',
                                    fc='#efefef', ec='#888888', lw=0.6, zorder=0))
        ax.text(x0 + .2, y1 - .27, title, fontsize=7.5, weight='bold', va='center', zorder=4,
                bbox=dict(fc='#efefef', ec='none', pad=1))  # линии уходят под название
    for n, label in labels.items():
        x, y = POS[n]
        ax.add_patch(FancyBboxPatch((x - NW / 2, y - NH / 2), NW, NH, boxstyle='round,pad=0,rounding_size=0.12',
                                    fc='white', ec='black', lw=0.6, zorder=2))
        ax.text(x, y, textwrap.fill(label, 18, break_on_hyphens=False), fontsize=6.6, ha='center', va='center',
                linespacing=1.1, zorder=3)
    for a, b, label in edges:
        pts = [box[a][:2], *VIA.get((a, b), ()), box[b][:2]]
        pts[0], pts[-1] = clip(box[a], *pts[1]), clip(box[b], *pts[-2])
        style = dict(color='#444444', lw=0.7, ls=(0, (3, 2))) if label else dict(color='black', lw=0.7, ls='-')
        for p, q in zip(pts[:-2], pts[1:-1]):
            ax.plot(*zip(p, q), **style, zorder=1)
        ax.annotate('', pts[-1], pts[-2], arrowprops=dict(arrowstyle='-|>', mutation_scale=7, shrinkA=0, shrinkB=0,
                                                          color=style['color'], lw=style['lw'], ls=style['ls']),
                    zorder=1)
        if label:
            x, y, rot = LABEL[a, b]
            ax.text(x, y, label, fontsize=6.5, style='italic', rotation=rot, ha='left' if rot == 0 else 'center',
                    va='center', bbox=dict(fc='white', ec='none', pad=0.6), zorder=4)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    try:  # pandoc из pypandoc_binary впереди системного, как в docs/build_docx.py
        import pypandoc
        os.environ['PATH'] = str(Path(pypandoc.get_pandoc_path()).parent) + os.pathsep + os.environ['PATH']
    except ImportError:
        pass
    source = REVIEW / 'literature_review.md'
    text = source.read_text(encoding='utf-8')
    block = re.search(r'```mermaid\n(.*?)```\n', text, re.S)
    figure, work, reference = REVIEW / '_map.png', REVIEW / '_build.md', REVIEW / 'reference.docx'
    cwd = Path.cwd()
    try:
        if block:
            draw_map(block[1], figure)
            text = text.replace(block[0], f'![Рисунок D.1 — Карта связей направлений]({figure.name}){{width=100%}}\n')
        work.write_text(text, encoding='utf-8')
        os.chdir(REVIEW)  # пути рисунков - относительно папки обзора
        build_reference(reference)
        build_docx(work, source.with_suffix('.docx'), reference)
    finally:
        os.chdir(cwd)
        for p in (figure, work, reference):
            p.unlink(missing_ok=True)
    print(source.with_suffix('.docx').name, 'записан')


if __name__ == '__main__':
    main()
