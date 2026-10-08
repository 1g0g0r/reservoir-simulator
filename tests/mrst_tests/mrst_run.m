function r = mrst_run(solver, G, rock, fluid, W, state0, c, model, bc)
% Расчет MRST по времени и результат для сравнения с paraphin (общий для tests/mrst_tests/mrst_*.m).
%   solver = 'incomp' - IMPES модуля incomp, fluid - initCoreyFluid: на каждом шаге c.dt давление incompTPFA
%            (суммарная подвижность на грани - гармоническое среднее, как mid() в paraphin), затем
%            explicitTransport (доли фаз против потока, подшаги по Куранту). Дебиты - по давлению с S начала шага,
%            как в paraphin;
%   solver = 'ad'     - simulateScheduleAD для model (TwoPhaseOilWaterModel, GeothermalModel), неявный Эйлер с c.dt.
% bc - граничные условия (pside, fluxside); без скважин (W = []) «закачка» - приток через границы, «отбор» - отток
% по фазам, а «добывающая» для температуры - последняя ячейка сетки.
% r: t, [с]; q_inj (> 0 - закачка), qw_prod, qo_prod (отбор), [м^3/с]; bhp_inj, [Па]; T_prod - температура ячейки
% добывающей, [K] (нули без температуры) - на каждом шаге; поля p, [Па], sw, T, [K] в шагах c.field_steps (ячейка
% (i, j) - элемент i + (j-1)*nx); elapsed - время расчета, [с].
if nargin < 8, model = []; end
if nargin < 9, bc = []; end
tic;
nph = size(state0.s, 2);
if strcmp(solver, 'incomp')
    T = computeTrans(G, rock);
    state = state0;
    state.wellSol = initWellSol(W, c.p0);
    [q_inj, qw_prod, qo_prod, bhp_inj] = deal(zeros(c.n_steps, 1));
    states = cell(c.n_steps, 1);
    for n = 1:c.n_steps
        state = incompTPFA(state, G, T, fluid, 'wells', W, 'bc', bc);
        mob = fluid.relperm(state.s) ./ [c.mu_w, c.mu_o];
        if isempty(W)
            % Фазовые потоки граничных граней: приток - по составу bc.sat, отток - по подвижностям ячейки
            cells = sum(G.faces.neighbors(bc.face, :), 2);
            q = state.flux(bc.face) .* sign_into(G, bc.face);
            fw = mob(cells, 1) ./ sum(mob(cells, :), 2);
            fw(q > 0) = bc.sat(q > 0, 1);
            [q_inj(n), qw_prod(n), qo_prod(n)] = bc_rates([q .* fw, q .* (1 - fw)]);
        else
            wc = W(2).cells;
            q_inj(n) = state.wellSol(1).flux;
            bhp_inj(n) = state.wellSol(1).pressure;
            qw_prod(n) = -state.wellSol(2).flux * mob(wc, 1) / sum(mob(wc, :));
            qo_prod(n) = -state.wellSol(2).flux * mob(wc, 2) / sum(mob(wc, :));
        end
        state = explicitTransport(state, G, c.dt, rock, fluid, 'wells', W, 'bc', bc);
        if any(n == c.field_steps)  % все состояния 75x75 за 8000 шагов - гигабайты
            states{n} = state;
        end
    end
    T_prod = zeros(c.n_steps, 1);
else
    schedule = simpleSchedule(repmat(c.dt, c.n_steps, 1), 'W', W, 'bc', bc);
    [ws, states] = simulateScheduleAD(state0, model, schedule);
    n = numel(states);
    if isempty(W)
        % Фазовые потоки граничных граней (ReservoirModel.storeBoundaryFluxes), ориентированные внутрь области
        [q_inj, qw_prod, qo_prod, bhp_inj] = deal(zeros(n, 1));
        for k = 1:n
            q = states{k}.flux(bc.face, :) .* sign_into(G, bc.face);
            if nph == 1
                q = [q, zeros(size(q))];
            end
            [q_inj(k), qw_prod(k), qo_prod(k)] = bc_rates(q);
        end
        prod_cell = G.cells.num;
    else
        q_inj = cellfun(@(x) x(1).qWs, ws);
        bhp_inj = cellfun(@(x) x(1).bhp, ws);
        qw_prod = -cellfun(@(x) x(2).qWs, ws);
        qo_prod = zeros(size(qw_prod));
        if isfield(ws{1}, 'qOs') && ~isempty(ws{1}(2).qOs)  % у одной воды поле есть, но пустое
            qo_prod = -cellfun(@(x) x(2).qOs, ws);
        end
        prod_cell = W(2).cells;
    end
    T_prod = zeros(size(qw_prod));
    if isfield(states{1}, 'T')
        T_prod = cellfun(@(s) s.T(prod_cell), states);
    end
end
n = numel(q_inj);
r = struct('t', c.dt * (1:n)', 'elapsed', toc, 'q_inj', q_inj, 'qw_prod', qw_prod, 'qo_prod', qo_prod, ...
           'bhp_inj', bhp_inj, 'T_prod', T_prod);
steps = c.field_steps(c.field_steps <= n);
r.field_t = r.t(steps);
r.p = cellfun(@(s) s.pressure, states(steps), 'UniformOutput', false);
r.sw = cellfun(@(s) s.s(:, 1), states(steps), 'UniformOutput', false);
if isfield(states{steps(1)}, 'T')
    r.T = cellfun(@(s) s.T, states(steps), 'UniformOutput', false);
end
end


function s = sign_into(G, faces)
% +1, если положительный поток грани (от neighbors(:, 1) к neighbors(:, 2)) направлен внутрь области
s = 1 - 2 * (G.faces.neighbors(faces, 2) == 0);
end


function [q_in, q_w, q_o] = bc_rates(q)
% q - фазовые потоки граничных граней внутрь области (вода, нефть): отток - сумма отрицательных по фазам, приток
% равен оттоку (несжимаемость). Записанный приток не берется: в geothermal поток граней притока в state.flux
% пересчитан после шага не с той подвижностью, что в уравнениях (6521 против оттока 4177 м^3/сут в первые сутки).
q_w = -sum(min(q(:, 1), 0));
q_o = -sum(min(q(:, 2), 0));
q_in = q_w + q_o;
end
