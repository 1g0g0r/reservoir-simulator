#!/usr/bin/env bash
# Сборка UMFPACK-расширения sparse_numba под Windows.
#
# Колесо sparse-numba для Windows содержит только SuperLU: в sparse_umfpack лежит один
# umfpack_numba_interface.py, а самого расширения cy_umfpack_wrapper там нет. Поэтому
# предупреждение "UMFPACK libraries were not found" одной установкой SuiteSparse не лечится -
# расширение надо собрать. Скрипт делает это целиком: качает SuiteSparse/OpenBLAS из репозитория
# MSYS2, забирает исходники обёртки из upstream, собирает СТАТИЧЕСКИ и кладёт .pyd в site-packages.
#
# Статически - принципиально: в колесе лежат mingw-DLL SuperLU, собранные против msvcrt, а пакеты
# MSYS2 - против ucrt. Динамическая линковка привела бы к тому, что libopenblas.dll, libgfortran-5.dll
# и libgcc_s_seh-1.dll существуют в двух вариантах под одним именем, а Windows загрузит только первый.
# Статический .pyd зависит только от ucrt и python312.dll и не пересекается с SuperLU.
#
# Запуск из Git Bash при активированном venv:  bash build_umfpack_windows.sh
set -euo pipefail

GCC=${GCC:-"C:/Program Files/mingw64/bin/gcc.exe"}
BIN=$(dirname "$GCC")
WORK=${WORK:-"${TMPDIR:-/tmp}/umfpack-build"}
REPO=https://repo.msys2.org/mingw/ucrt64
SRC=https://raw.githubusercontent.com/th1275/sparse_numba/main/sparse_numba/sparse_umfpack

# --- 0. Проверки окружения -------------------------------------------------------------------
[ -x "$GCC" ] || { echo "Нет mingw-w64 gcc: $GCC (переопредели переменной GCC)"; exit 1; }
[ "$("$GCC" -dumpmachine)" = x86_64-w64-mingw32 ] || { echo "gcc не x86_64-w64-mingw32"; exit 1; }
printf '#include <stdio.h>\nint main(void){return 0;}\n' > "${TMPDIR:-/tmp}/crt.c"
"$GCC" "${TMPDIR:-/tmp}/crt.c" -o "${TMPDIR:-/tmp}/crt.exe"
"$BIN/objdump.exe" -p "${TMPDIR:-/tmp}/crt.exe" | grep -q api-ms-win-crt \
  || { echo "gcc собран против msvcrt, а пакеты берутся из ucrt64. Нужен UCRT-вариант MinGW-Builds."; exit 1; }

IFS='|' read -r PYINC PYLIBS PYTAG PKGDIR <<EOF
$(python -c "
import sys, sysconfig, os, sparse_numba
print('|'.join((os.path.join(sys.base_prefix, 'include'),
                os.path.join(sys.base_prefix, 'libs'),
                f'cp{sys.version_info.major}{sys.version_info.minor}',
                os.path.dirname(sparse_numba.__file__))).replace(chr(92), '/'))")
EOF
echo "python: $PYTAG, sparse_numba: $PKGDIR"

mkdir -p "$WORK/dl" "$WORK/src" "$WORK/ss"

# --- 1. zstd: пакеты MSYS2 лежат в .tar.zst, GNU tar зовёт внешний zstd ------------------------
ZSTD="$WORK/dl/zstd-v1.5.6-win64/zstd.exe"
if [ ! -x "$ZSTD" ]; then
  curl -sSL -o "$WORK/dl/zstd.zip" https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-v1.5.6-win64.zip
  python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$WORK/dl/zstd.zip" "$WORK/dl"
fi

# --- 2. SuiteSparse + OpenBLAS из MSYS2 (ucrt64) ----------------------------------------------
# Имя берём из индекса репозитория: старые версии оттуда вычищаются, жёсткая ссылка протухнет.
for pkg in suitesparse openblas; do
  file=$(curl -sSL "$REPO/" | grep -oE "mingw-w64-ucrt-x86_64-$pkg-[0-9][^\"]*\.pkg\.tar\.zst" | sort -V | tail -1)
  [ -n "$file" ] || { echo "не нашёл пакет $pkg в $REPO"; exit 1; }
  echo "MSYS2: $file"
  [ -f "$WORK/dl/$file" ] || curl -sSL -o "$WORK/dl/$file" "$REPO/$file"
  "$ZSTD" -d -c "$WORK/dl/$file" | tar -C "$WORK/ss" -xf -
done
SS="$WORK/ss/ucrt64"

# --- 3. Исходники обёртки из upstream (в колесо они не попали) --------------------------------
# .c уже сгенерирован Cython'ом, ставить Cython не нужно.
for f in cy_umfpack_wrapper.c cy_umfpack_wrapper_api.h umfpack_wrapper.c umfpack_wrapper.h; do
  [ -f "$WORK/src/$f" ] || curl -sSL -o "$WORK/src/$f" "$SRC/$f"
done

# CHOLMOD/METIS из MSYS2 собран против их версии mingw-w64 crt, где ещё экспортировался _setjmp.
# В свежем MinGW-Builds этого имени нет - есть __intrinsic_setjmp с той же сигнатурой.
cat > "$WORK/src/setjmp_shim.c" <<'EOF'
int __intrinsic_setjmp(void *buf, void *frame);
void *__imp__setjmp = (void *) &__intrinsic_setjmp;
EOF

# --- 4. Сборка ---------------------------------------------------------------------------------
OUT="cy_umfpack_wrapper.$PYTAG-win_amd64.pyd"
(cd "$WORK/src" && "$GCC" -O2 -shared \
  -I"$PYINC" -I"$SS/include/suitesparse" -I. \
  cy_umfpack_wrapper.c umfpack_wrapper.c setjmp_shim.c -o "$OUT" \
  "$SS/lib/libumfpack.a" "$SS/lib/libcholmod.a" "$SS/lib/libamd.a" "$SS/lib/libcamd.a" \
  "$SS/lib/libcolamd.a" "$SS/lib/libccolamd.a" "$SS/lib/libsuitesparseconfig.a" \
  "$SS/lib/libopenblas.a" \
  -L"$PYLIBS" -l"python${PYTAG#cp}" \
  -static-libgcc -Wl,-Bstatic -lgomp -lgfortran -lquadmath -lpthread -Wl,-Bdynamic -lm)
"$BIN/strip.exe" --strip-unneeded "$WORK/src/$OUT"

# Внешних зависимостей быть не должно, кроме ucrt/kernel32/python
if "$BIN/objdump.exe" -p "$WORK/src/$OUT" | grep -i "DLL Name" | grep -viE "api-ms-win-crt|KERNEL32|python"; then
  echo "ОШИБКА: .pyd тянет посторонние DLL - линковка не статическая"; exit 1
fi

# --- 5. Установка + починка детекта ------------------------------------------------------------
cp "$WORK/src/$OUT" "$PKGDIR/sparse_umfpack/"

# is_umf_available() ищет *umfpack*.dll рядом с пакетом; при статической сборке её нет.
# Дописываем проверку по самому модулю - так же, как это уже сделано для SuperLU.
python - "$PKGDIR/__init__.py" <<'EOF'
import io, sys
path = sys.argv[1]
src = io.open(path, encoding='utf-8').read()
anchor = "    # Log the availability status\n"
patch = '''    # Расширение, слинкованное с SuiteSparse статически, не кладет рядом ни одной
    # *umfpack*.dll, поэтому glob выше его не находит. Проверяем сам модуль - ровно так же,
    # как initialize_superlu проверяет cy_superlu_wrapper.
    if not _HAS_UMFPACK:
        try:
            from .sparse_umfpack import cy_umfpack_wrapper  # noqa: F401
            _HAS_UMFPACK = True
        except ImportError:
            pass

'''
assert anchor in src, 'якорь не найден: sparse_numba обновился, патч надо пересмотреть'
if patch not in src:
    io.open(path, 'w', encoding='utf-8', newline='\n').write(src.replace(anchor, patch + anchor, 1))
EOF

# --- 6. Проверка -------------------------------------------------------------------------------
python - <<'EOF'
import numpy as np
from scipy.sparse import random as sprandom, identity
import sparse_numba
from sparse_numba.sparse_superlu.superlu_numba_interface import superlu_solve_csc
from sparse_numba.sparse_umfpack.umfpack_numba_interface import (
    umfpack_solve_csc, umfpack_factorize_csc, umfpack_solve_factored, umfpack_free_factors)

assert sparse_numba.is_umf_available(), 'UMFPACK не определился'
A = (sprandom(500, 500, density=0.01, random_state=1) + 5.0 * identity(500)).tocsc()
A.sort_indices()
x_true = np.random.default_rng(0).standard_normal(500)
b = A @ x_true
args = (np.ascontiguousarray(A.data), A.indices.astype(np.int32), A.indptr.astype(np.int32))

x_u, info_u = umfpack_solve_csc(*args, b)
x_s, _ = superlu_solve_csc(*args, b)          # оба солвера в одном процессе
h, _ = umfpack_factorize_csc(*args)
x_f, info_f = umfpack_solve_factored(h, b)
umfpack_free_factors(h)

assert info_u == 0 and np.allclose(x_u, x_true), 'UMFPACK решил неверно'
assert info_f == 0 and np.allclose(x_f, x_true), 'UMFPACK (префакторизация) решил неверно'
assert np.allclose(x_u, x_s), 'UMFPACK и SuperLU разошлись'
print('UMFPACK работает: невязка %.2e, SuperLU рядом живой' % np.linalg.norm(x_u - x_true))
EOF
echo "Готово. Промежуточные файлы: $WORK (можно удалить)."
