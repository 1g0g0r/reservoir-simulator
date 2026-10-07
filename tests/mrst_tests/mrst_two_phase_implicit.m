function r = mrst_two_phase_implicit(case_file, out_file)
% MRST для tests/mrst_tests/test_two_phase_implicit.py: двухфазная задача start.py полностью неявной схемой
% ad-blackoil (как tests/mrst_tests/mrst_test.m).
%
% Постановка - как в mrst_two_phase_impes.m (start.py без парафина и теплообмена, ОФП Кори на [S_min, S_max],
% четверть скважины по Писману, flooded_injector). Схема другая: TwoPhaseOilWaterModel решает давление и
% насыщенность совместно методом Ньютона, фазовые подвижности на грани берутся против потока (у paraphin суммарная
% подвижность - гармоническое среднее, доли фаз - против потока), неявный Эйлер с шагом c.dt (1 сут).
% Фазы названы как есть: W - вода, O - нефть (в mrst_test.m фаза W несла свойства нефти).
%
% Тест (common.run_mrst):  octave-cli --no-gui --eval "addpath('<репозиторий>/tests/mrst_tests');
%                            mrst_two_phase_implicit('<имя>.case.json', '<имя>.json')"
% Вручную - из tests/mrst_tests в MATLAB или Octave без аргументов (40x40, 2000 сут): r = mrst_two_phase_implicit;
if nargin < 1, case_file = ''; end
if nargin < 2, out_file = ''; end
c = mrst_start(case_file);

G = computeGeometry(cartGrid([c.nx, c.ny, 1], [c.Lx, c.Ly, c.h]));
rock = makeRock(G, c.k, c.m);
% smin = [S_min, 1 - S_max]: k_rw = ((S - S_min)/(1 - sum(smin)))^n, k_ro = ((1 - S - (1 - S_max))/(...))^n
fluid = initSimpleADIFluid('phases', 'WO', 'mu', [c.mu_w, c.mu_o], 'rho', [c.ro_w, c.ro_o], ...
                           'n', [c.n, c.n], 'smin', [c.S_min, 1 - c.S_max]);
model = TwoPhaseOilWaterModel(G, rock, fluid);

% Писман: r_e = 0.14*sqrt(hx^2 + hy^2), WI = 2*pi*k*h/ln(r_e/r_w); wi_mult = 0.25 - четверть скважины в углу
W = verticalWell([], G, rock, 1, 1, [], 'Type', 'bhp', 'Val', c.p_inj, 'Radius', c.rw, ...
                 'Comp_i', [1, 0], 'Sign', 1, 'Name', 'inj');
W = verticalWell(W, G, rock, c.nx, c.ny, [], 'Type', 'bhp', 'Val', c.p_prod, 'Radius', c.rw, ...
                 'Comp_i', [0, 1], 'Sign', -1, 'Name', 'prod');
for w = 1:numel(W)
    W(w).WI = W(w).WI * c.wi_mult;
end

state0 = initResSol(G, c.p0, [c.S_min, 1 - c.S_min]);
if c.flooded_injector
    state0.s(1, :) = [c.S_max, 1 - c.S_max];
end

r = mrst_run('ad', G, rock, fluid, W, state0, c, model);
mrst_output(r, out_file);
end
