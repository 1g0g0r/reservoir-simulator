function c = mrst_start(case_file)
% Начало расчетов tests/mrst_tests/mrst_*.m: постановка и запуск MRST.
% С case_file постановка читается из JSON - его пишет tests/mrst_tests/common.py из constants.py, все числа в СИ.
% Без него - постановка start.py по умолчанию (defaults ниже, числа constants.py на 2026-10; тест передает
% актуальные). MRST запускается из c.mrst_root или переменной окружения MRST_ROOT, если еще не запущен в этом
% сеансе, подключаются модули incomp ad-core ad-blackoil ad-props compositional geothermal; в Octave подменяется
% один метод geothermal (octave_fix_geothermal).
if nargin >= 1 && ~isempty(case_file)
    c = jsondecode(fileread(case_file));
else
    c = defaults();
end
if isempty(which('mrstModule'))
    root = getenv('MRST_ROOT');
    if isfield(c, 'mrst_root')
        root = c.mrst_root;
    end
    assert(~isempty(root), 'MRST не запущен: выполните startup.m из каталога MRST или задайте MRST_ROOT');
    run(fullfile(root, 'startup.m'));
end
mrstModule add incomp ad-core ad-blackoil ad-props compositional geothermal
gravity off
if exist('OCTAVE_VERSION', 'builtin')
    octave_fix_geothermal(ROOTDIR);
end
end


function c = defaults()
% Постановка start.py без парафина и теплообмена: пласт 200x200x10 м, нагнетательная (1, 1) на 150 бар,
% добывающая (nx, ny) на 50 бар, четверть скважины по Писману; 40x40, 2000 сут с шагом 1 сут.
c = struct('Lx', 200, 'Ly', 200, 'h', 10, 'k', 0.2 * 9.869233e-13, 'm', 0.3, ...
           'mu_o', 5.769e-3, 'ro_w', 1000, 'ro_o', 860, 'n', 2, 'S_min', 0.18, 'S_max', 0.7, 'rw', 0.1, ...
           'p_inj', 150e5, 'p_prod', 50e5, 'wi_mult', 0.25, 'p0', 100e5, ...
           'T0', 273.15 + 70, 'T_inj', 273.15 + 20, 'Cp_w', 4200, 'lambda_w', 0.6, 'lambda_R', 1.8, ...
           'rho_R', 2700, 'Cp_R', 1000, 'nx', 40, 'ny', 40, 'flooded_injector', false, 'solver', 'ad', ...
           'inj_control', 'bhp', 'inj_val', 150e5, 'dt', 86400, 'n_steps', 2000, 'perm', []);
c.field_steps = [250; 500; 1000; 2000];
c.mu_w = 2.414e-5 * 10^(247.8 / (c.T0 - 140));  % Андраде при пластовой температуре, как calc_mu_w
end


function octave_fix_geothermal(mrst_root)
% GeothermalGenericFacilityModel.getProductionWellTemperature (MRST 2024a) умножает вектор на разреженную матрицу
% поэлементно (`~ok.*map.perforationSum*Tperf./nc`): MATLAB расширяет размерности, Octave для разреженных - нет.
% Исправленная копия класса (то же значение: средняя температура вскрытых ячеек) кладется во временный каталог
% перед MRST в пути поиска; установленный MRST не меняется.
src = fullfile(mrst_root, 'modules', 'geothermal', 'models', 'GeothermalGenericFacilityModel.m');
bad = '~ok.*map.perforationSum*Tperf./nc';
text = fileread(src);
if isempty(strfind(text, bad))
    return
end
dst = fullfile(tempdir, 'mrst_octave_fix');
if ~exist(dst, 'dir')
    mkdir(dst);
end
fid = fopen(fullfile(dst, 'GeothermalGenericFacilityModel.m'), 'w');
fprintf(fid, '%s', strrep(text, bad, '~ok.*((map.perforationSum*Tperf)./nc)'));
fclose(fid);
addpath(dst);
end
