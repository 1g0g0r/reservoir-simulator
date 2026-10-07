function r = mrst_run(solver, G, rock, fluid, W, state0, c, model)
% Расчет MRST по времени и результат для сравнения с paraphin (общий для tests/mrst_tests/mrst_*.m).
%   solver = 'incomp' - IMPES модуля incomp, fluid - initCoreyFluid: на каждом шаге c.dt давление incompTPFA
%            (суммарная подвижность на грани - гармоническое среднее, как mid() в paraphin), затем
%            explicitTransport (доли фаз против потока, подшаги по Куранту). Дебиты - по давлению с S начала шага,
%            как в paraphin;
%   solver = 'ad'     - simulateScheduleAD для model (TwoPhaseOilWaterModel, GeothermalModel), неявный Эйлер с c.dt.
% r: t, [с]; q_inj (> 0 - закачка), qw_prod, qo_prod (отбор), [м^3/с]; bhp_inj, [Па]; T_prod - температура ячейки
% добывающей, [K] (нули без температуры) - на каждом шаге; поля p, [Па], sw, T, [K] в шагах c.field_steps (ячейка
% (i, j) - элемент i + (j-1)*nx); elapsed - время расчета, [с].
tic;
if strcmp(solver, 'incomp')
    T = computeTrans(G, rock);
    state = state0;
    state.wellSol = initWellSol(W, c.p0);
    [q_inj, qw_prod, qo_prod, bhp_inj] = deal(zeros(c.n_steps, 1));
    states = cell(c.n_steps, 1);
    wc = W(2).cells;
    for n = 1:c.n_steps
        state = incompTPFA(state, G, T, fluid, 'wells', W);
        mob = fluid.relperm(state.s(wc, :)) ./ [c.mu_w, c.mu_o];
        q_inj(n) = state.wellSol(1).flux;
        bhp_inj(n) = state.wellSol(1).pressure;
        qw_prod(n) = -state.wellSol(2).flux * mob(1) / sum(mob);
        qo_prod(n) = -state.wellSol(2).flux * mob(2) / sum(mob);
        state = explicitTransport(state, G, c.dt, rock, fluid, 'wells', W);
        if any(n == c.field_steps)  % все состояния 75x75 за 8000 шагов - гигабайты
            states{n} = state;
        end
    end
    T_prod = zeros(c.n_steps, 1);
else
    schedule = simpleSchedule(repmat(c.dt, c.n_steps, 1), 'W', W);
    [ws, states] = simulateScheduleAD(state0, model, schedule);
    q_inj = cellfun(@(x) x(1).qWs, ws);
    bhp_inj = cellfun(@(x) x(1).bhp, ws);
    qw_prod = -cellfun(@(x) x(2).qWs, ws);
    qo_prod = zeros(size(qw_prod));
    if isfield(ws{1}, 'qOs') && ~isempty(ws{1}(2).qOs)  % у одной воды поле есть, но пустое
        qo_prod = -cellfun(@(x) x(2).qOs, ws);
    end
    T_prod = zeros(size(qw_prod));
    if isfield(states{1}, 'T')
        T_prod = cellfun(@(s) s.T(W(2).cells), states);
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
