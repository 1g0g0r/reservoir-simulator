"""Прогон решателя с поправленными константами - в копии пакета, а не правкой `paraphin/constants.py` на месте.

Единственный патчер проекта: им пользуются тесты, опыты (`experiments/`), демо и статья. Флаги и размеры сетки -
константы уровня модуля, numba вшивает их в машинный код. Править constants.py в репозитории на время прогона
опасно: индекс дискового кеша numba привязан к mtime файла с функцией, и процесс, лениво компилирующий функции
после отката constants.py, мог бы подхватить код, собранный с чужими флагами; жесткое прерывание оставило бы
файл поправленным, а каждая правка сбрасывала бы кеш репозитория. Копия живет в `outputs/.flags_on/<имя>/paraphin` со своим кешем:
репозиторий и его кеш не трогаются, кеш копии переживает перезапуски (файлы копируются с сохранением
mtime и только если изменились), а за устаревание отвечает та же проверка хеша исходников, что и в пакете.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COPIES = ROOT / 'outputs' / '.flags_on'


def patch_constants(text: str, values: dict) -> str:
    """Подстановка значений в текст `constants.py`: каждое имя присваивается в начале строки, ровно один раз
    и одной строкой (`Nx, Ny` - одно имя). Строка заменяется целиком, вместе с комментарием."""
    for name, value in values.items():
        pattern = re.compile(rf'^{re.escape(name)}(\s*:\s*\w+)?\s*=.*$', re.M)
        if len(pattern.findall(text)) != 1:
            raise KeyError(f'В constants.py нет ровно одной строки `{name} = ...`')
        text = pattern.sub(f'{name} = {value}', text)
    return text


def make_copy(name: str, values: dict) -> Path:
    """Копия пакета `paraphin` с поправленными константами. Возвращает корень копии (его - в sys.path)."""
    root = COPIES / name
    src, dst = ROOT / 'paraphin', root / 'paraphin'
    for path in src.rglob('*'):
        if '__pycache__' in path.parts or path.is_dir():
            continue
        target = dst / path.relative_to(src)
        if path.name == 'constants.py':
            continue
        if not target.is_file() or target.read_bytes() != path.read_bytes():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    constants = patch_constants((src / 'constants.py').read_text(encoding='utf-8'), values)
    target = dst / 'constants.py'
    if not target.is_file() or target.read_text(encoding='utf-8') != constants:
        target.write_text(constants, encoding='utf-8')
    return root


def run_worker(name: str, values: dict, script: Path, args=(), timeout: float = 3600.0) -> dict:
    """Запуск `script` в отдельном процессе против копии пакета. Скрипт пишет JSON в путь из argv[1]."""
    root = make_copy(name, values)
    out = root / 'result.json'
    env = dict(os.environ, PYTHONPATH=str(root))
    subprocess.run([sys.executable, str(script), str(out), *map(str, args)], cwd=root, env=env, check=True,
                   timeout=timeout)
    return json.loads(out.read_text(encoding='utf-8'))
