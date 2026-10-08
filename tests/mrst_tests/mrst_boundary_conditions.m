function r = mrst_boundary_conditions(case_file, out_file)
% MRST для tests/mrst_tests/test_boundary_conditions.py: граничные условия на одномерной полосе.
%
% Полоса Lx x Ly x h из nx = 1 и ny ячеек вдоль y, без скважин: течение задают границы ymin (вход; у paraphin -
% Bound.Left, j < 0) и ymax (выход; Bound.Right). Варианты c.variant:
%   'dirichlet' - давление p_inj на входе и p_prod на выходе (pside), на входе втекает вода (sat = [1, 0]),
%                 начальная S = S_min. Проводимость граничной грани на притоке: incomp - подвижность ячейки,
%                 ad - против потока (втекающей воды); у paraphin - гармоническое среднее подвижностей ячейки и
%                 фиктивной ячейки при S из ГУ (Pressure.py, Flows_in_cells.py);
%   'neumann'   - у paraphin градиент давления g на входе (поток g*A*lambda), здесь - тот же расход
%                 c.q_bc = g*A*k/mu_w (fluxside, > 0 - внутрь области); полоса водонасыщена (S = S_max), подвижность
%                 постоянна, поэтому и у paraphin расход постоянен; на выходе - p_prod;
%   'thermal'   - одна вода (GeothermalModel): давление и температура Дирихле на обеих границах - на входе p_inj и
%                 T_inj, на выходе p_prod и T0 (у paraphin - те же Дирихле, иначе теплопроводность через выходную
%                 грань разная).
% Решатель - c.solver: 'incomp' или 'ad' (dirichlet, neumann); в варианте thermal - GeothermalModel.
%
% Тест (common.run_mrst):  octave-cli --no-gui --eval "addpath('<репозиторий>/tests/mrst_tests');
%                            mrst_boundary_conditions('<имя>.case.json', '<имя>.json')"
% Вручную - из tests/mrst_tests в MATLAB или Octave без аргументов (dirichlet, incomp, 1x100, 200 сут):
%   r = mrst_boundary_conditions;
if nargin < 1, case_file = ''; end
if nargin < 2, out_file = ''; end
c = mrst_start(case_file);
if isempty(case_file)
    c.variant = 'dirichlet';
    c.solver = 'incomp';
    c.nx = 1;
    c.ny = 100;
    c.dt = 0.05 * day;
    c.n_steps = 4000;
    c.field_steps = [500; 1000; 2000; 4000];
end

G = computeGeometry(cartGrid([c.nx, c.ny, 1], [c.Lx, c.Ly, c.h]));
rock = makeRock(G, c.k, c.m);
model = [];
solver = c.solver;
if strcmp(c.variant, 'thermal')
    rock = addThermalRockProps(rock, 'lambdaR', c.lambda_R, 'rhoR', c.rho_R, 'CpR', c.Cp_R);
    fluid = initSimpleADIFluid('phases', 'W', 'mu', c.mu_w, 'rho', c.ro_w);
    fluid = addThermalFluidProps(fluid, 'Cp', c.Cp_w, 'lambdaF', c.lambda_w, 'useEOS', false);
    fluid.muW = @(p, T, varargin) 2.414e-5 * 10.^(247.8 ./ (T - 140.0));  % Андраде, T в K
    model = GeothermalModel(G, rock, fluid);
    bc = pside([], G, 'ymin', c.p_inj, 'sat', 1);
    n_in = numel(bc.face);
    bc = pside(bc, G, 'ymax', c.p_prod, 'sat', 1);
    T_bc = repmat(c.T0, numel(bc.face), 1);
    T_bc(1:n_in) = c.T_inj;
    bc = addThermalBCProps(bc, 'T', T_bc);
    state0 = initResSol(G, c.p0, 1);
    state0.T = repmat(c.T0, G.cells.num, 1);
    solver = 'ad';
else
    if strcmp(c.solver, 'incomp')
        fluid = initCoreyFluid('mu', [c.mu_w, c.mu_o], 'rho', [c.ro_w, c.ro_o], 'n', [c.n, c.n], ...
                               'sr', [c.S_min, 1 - c.S_max], 'kwm', [1, 1]);
    else
        fluid = initSimpleADIFluid('phases', 'WO', 'mu', [c.mu_w, c.mu_o], 'rho', [c.ro_w, c.ro_o], ...
                                   'n', [c.n, c.n], 'smin', [c.S_min, 1 - c.S_max]);
        model = TwoPhaseOilWaterModel(G, rock, fluid);
    end
    if strcmp(c.variant, 'dirichlet')
        bc = pside([], G, 'ymin', c.p_inj, 'sat', [1, 0]);
        s0 = c.S_min;
    else
        bc = fluxside([], G, 'ymin', c.q_bc, 'sat', [1, 0]);
        s0 = c.S_max;
    end
    bc = pside(bc, G, 'ymax', c.p_prod, 'sat', [1, 0]);  % состав на выходе не нужен: там отток из ячейки
    state0 = initResSol(G, c.p0, [s0, 1 - s0]);
end

r = mrst_run(solver, G, rock, fluid, [], state0, c, model, bc);
mrst_output(r, out_file);
end
