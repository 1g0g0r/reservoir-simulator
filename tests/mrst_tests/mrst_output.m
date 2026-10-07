function mrst_output(r, out_file)
% Результат расчета tests/mrst_tests/mrst_*.m в JSON для tests/mrst_tests/common.py (jsonencode). Графики строит
% только Python-тест; без out_file (ручной запуск) ничего не пишется - результат возвращает сама функция теста.
if isempty(out_file)
    return
end
fid = fopen(out_file, 'w');
fprintf(fid, '%s', jsonencode(r));
fclose(fid);
end
