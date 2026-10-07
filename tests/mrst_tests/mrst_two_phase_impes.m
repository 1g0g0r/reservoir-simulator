function r = mrst_two_phase_impes(case_file, out_file)
% MRST для tests/mrst_tests/test_two_phase_impes.py: двухфазная задача start.py схемой IMPES модуля incomp.
%
% Постановка start.py без парафина и теплообмена: пласт Lx x Ly x h, однородные k и m, вода и нефть с постоянными
% вязкостями (пластовая температура), ОФП Кори k_rw = ((S - S_min)/(S_max - S_min))^n,
% k_ro = ((S_max - S)/(S_max - S_min))^n - ровно pf_w, pf_o paraphin; нагнетательная в ячейке (1, 1) и добывающая
% в (nx, ny) на забойном давлении, четверть скважины по Писману. flooded_injector - ячейка нагнетательной заранее
% промыта (S = S_max): тогда подвижность закачки одинакова у paraphin (k/mu_w) и MRST (суммарная подвижность ячейки).
% Схема та же, что у paraphin: давление неявно, суммарная подвижность на грани - гармоническое среднее, насыщенность
% явно против потока (mrst_run, 'incomp'); давление здесь пересчитывается раз в c.dt (0.25 сут).
%
% Тест (common.run_mrst):  octave-cli --no-gui --eval "addpath('<репозиторий>/tests/mrst_tests');
%                            mrst_two_phase_impes('<имя>.case.json', '<имя>.json')"
% Вручную - из tests/mrst_tests в MATLAB или Octave без аргументов (40x40, 2000 сут): r = mrst_two_phase_impes;
if nargin < 1, case_file = ''; end
if nargin < 2, out_file = ''; end
c = mrst_start(case_file);
if isempty(case_file)
    c.dt = 0.25 * day;
    c.n_steps = 8000;
    c.field_steps = [1000; 2000; 4000; 8000];
end

G = computeGeometry(cartGrid([c.nx, c.ny, 1], [c.Lx, c.Ly, c.h]));
rock = makeRock(G, c.k, c.m);
fluid = initCoreyFluid('mu', [c.mu_w, c.mu_o], 'rho', [c.ro_w, c.ro_o], 'n', [c.n, c.n], ...
                       'sr', [c.S_min, 1 - c.S_max], 'kwm', [1, 1]);

% Писман: r_e = 0.14*sqrt(hx^2 + hy^2), WI = 2*pi*k*h/ln(r_e/r_w), как _re и _prod_mult в paraphin;
% wi_mult = 0.25 - четверть скважины в углу (mult в start.py)
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

r = mrst_run('incomp', G, rock, fluid, W, state0, c);
mrst_output(r, out_file);
end
