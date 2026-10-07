function r = mrst_heterogeneous(case_file, out_file)
% MRST для tests/mrst_tests/test_heterogeneous.py: двухфазная задача start.py на логнормальном поле проницаемости.
%
% Постановка start.py без парафина и теплообмена (скважины на забойном давлении), но k - поле c.perm, [м^2],
% в порядке ячеек MRST (элемент i + (j-1)*nx). Тест передает поле test_heterogeneous.perm_field (numpy, зерно 1):
% ln k нормален со средним ln(init_k) и СКО 1, радиус корреляции 20 м; то же поле получает paraphin (solver.k).
% Однородная k прячет ошибки ориентации осей и осреднения на гранях, поле - нет. Проводимость грани у обоих -
% гармоническое среднее k (computeTrans), продуктивность скважин - по k своей ячейки. Решатель - c.solver
% ('incomp' или 'ad', тест считает оба).
%
% Тест (common.run_mrst):  octave-cli --no-gui --eval "addpath('<репозиторий>/tests/mrst_tests');
%                            mrst_heterogeneous('<имя>.case.json', '<имя>.json')"
% Вручную - из tests/mrst_tests в MATLAB или Octave без аргументов (incomp, 40x40, свое поле k (rng(1))): r = mrst_heterogeneous;
if nargin < 1, case_file = ''; end
if nargin < 2, out_file = ''; end
c = mrst_start(case_file);
if isempty(case_file)
    c.solver = 'incomp';
    c.dt = 0.25 * day;
    c.n_steps = 8000;
    c.field_steps = [1000; 2000; 4000; 8000];
    c.perm = lognormal_field(c, 1.0, 20.0);
end

G = computeGeometry(cartGrid([c.nx, c.ny, 1], [c.Lx, c.Ly, c.h]));
rock = makeRock(G, c.perm(:), c.m);
model = [];
if strcmp(c.solver, 'incomp')
    fluid = initCoreyFluid('mu', [c.mu_w, c.mu_o], 'rho', [c.ro_w, c.ro_o], 'n', [c.n, c.n], ...
                           'sr', [c.S_min, 1 - c.S_max], 'kwm', [1, 1]);
else
    fluid = initSimpleADIFluid('phases', 'WO', 'mu', [c.mu_w, c.mu_o], 'rho', [c.ro_w, c.ro_o], ...
                               'n', [c.n, c.n], 'smin', [c.S_min, 1 - c.S_max]);
    model = TwoPhaseOilWaterModel(G, rock, fluid);
end

% Писман по k ячейки скважины; четверть скважины в углу
W = verticalWell([], G, rock, 1, 1, [], 'Type', 'bhp', 'Val', c.p_inj, 'Radius', c.rw, ...
                 'Comp_i', [1, 0], 'Sign', 1, 'Name', 'inj');
W = verticalWell(W, G, rock, c.nx, c.ny, [], 'Type', 'bhp', 'Val', c.p_prod, 'Radius', c.rw, ...
                 'Comp_i', [0, 1], 'Sign', -1, 'Name', 'prod');
for w = 1:numel(W)
    W(w).WI = W(w).WI * c.wi_mult;
end

state0 = initResSol(G, c.p0, [c.S_min, 1 - c.S_min]);
r = mrst_run(c.solver, G, rock, fluid, W, state0, c, model);
mrst_output(r, out_file);
end


function perm = lognormal_field(c, sigma, corr)
% Поле для ручного запуска (тест передает свое): гауссов шум, сглаженный ядром exp(-r^2/corr^2),
% СКО ln k = sigma вокруг ln k.
rng(1);
hx = c.Lx / c.nx;
[X, Y] = meshgrid(-ceil(2 * corr / hx):ceil(2 * corr / hx));
g = conv2(randn(c.nx, c.ny), exp(-(X.^2 + Y.^2) * (hx / corr)^2), 'same');
perm = c.k * exp(sigma * (g(:) - mean(g(:))) / std(g(:)));
end
