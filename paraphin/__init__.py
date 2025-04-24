import taichi as ti

from .constants import data_type, Nr, r

# массивы радиусов пор в необходимых степенях
r1 = ti.field(dtype=data_type, shape=Nr)
r2 = ti.field(dtype=data_type, shape=Nr)
r3 = ti.field(dtype=data_type, shape=Nr)
r4 = ti.field(dtype=data_type, shape=Nr)
r5 = ti.field(dtype=data_type, shape=Nr)
r6 = ti.field(dtype=data_type, shape=Nr)
r2_np = r * r
r3_np = r2_np * r
r4_np = r3_np * r
r5_np = r4_np * r
r6_np = r5_np * r
r1.from_numpy(r)
r2.from_numpy(r2_np)
r3.from_numpy(r3_np)
r4.from_numpy(r4_np)
r5.from_numpy(r5_np)
r6.from_numpy(r6_np)
