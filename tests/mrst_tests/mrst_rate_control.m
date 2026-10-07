function r = mrst_rate_control(case_file, out_file)
% MRST для tests/mrst_tests/test_rate_control.py: нагнетательная на заданном дебите, добывающая на забойном давлении.
%
% Постановка start.py без парафина и теплообмена с закомментированным в нем вариантом закачки по дебиту:
% нагнетательная закачивает c.inj_val (100 м^3/сут) воды, ее забойное давление - неизвестное (в paraphin -
% upd_q_and_eta: P_заб = P_ячейки + q/J). Решатель - c.solver: 'incomp' (IMPES, как у paraphin, шаг давления 0.25
% сут) или 'ad' (ad-blackoil, неявный Эйлер, 1 сут); тест считает оба. Пока ячейка нагнетательной не промыта, MRST
% берет в ней суммарную подвижность, а paraphin - k/mu_w, поэтому забойное давление MRST в первые сутки выше.
%
% Тест (common.run_mrst):  octave-cli --no-gui --eval "addpath('<репозиторий>/tests/mrst_tests');
%                            mrst_rate_control('<имя>.case.json', '<имя>.json')"
% Вручную - из tests/mrst_tests в MATLAB или Octave без аргументов (incomp, 40x40, 2000 сут): r = mrst_rate_control;
if nargin < 1, case_file = ''; end
if nargin < 2, out_file = ''; end
c = mrst_start(case_file);
if isempty(case_file)
    c.solver = 'incomp';
    c.inj_val = 100 / day;
    c.dt = 0.25 * day;
    c.n_steps = 8000;
    c.field_steps = [1000; 2000; 4000; 8000];
end

G = computeGeometry(cartGrid([c.nx, c.ny, 1], [c.Lx, c.Ly, c.h]));
rock = makeRock(G, c.k, c.m);
model = [];
if strcmp(c.solver, 'incomp')
    fluid = initCoreyFluid('mu', [c.mu_w, c.mu_o], 'rho', [c.ro_w, c.ro_o], 'n', [c.n, c.n], ...
                           'sr', [c.S_min, 1 - c.S_max], 'kwm', [1, 1]);
else
    fluid = initSimpleADIFluid('phases', 'WO', 'mu', [c.mu_w, c.mu_o], 'rho', [c.ro_w, c.ro_o], ...
                               'n', [c.n, c.n], 'smin', [c.S_min, 1 - c.S_max]);
    model = TwoPhaseOilWaterModel(G, rock, fluid);
end

% Нагнетательная - 'rate' (дебит воды, [м^3/с]), добывающая - 'bhp'; Писман и четверть скважины - как в
% mrst_two_phase_impes.m
W = verticalWell([], G, rock, 1, 1, [], 'Type', 'rate', 'Val', c.inj_val, 'Radius', c.rw, ...
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
