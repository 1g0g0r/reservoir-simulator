function r = mrst_thermal(case_file, out_file)
% MRST для tests/mrst_tests/test_thermal.py: закачка холодной воды, неизотермическая фильтрация, модуль geothermal.
%
% GeothermalModel считает одну фазу - воду (его двухфазный режим - вода и пар одного компонента, без ОФП
% нефть-вода), поэтому и paraphin в тесте однофазный (S_max = 1, нефти нет). Вода c.T_inj (20 C) закачивается
% в пласт c.T0 (70 C) по схеме start.py (скважины на забойном давлении), давление и температура - неявно, шаг
% c.dt (1 сут). Как в paraphin: плотность воды постоянна, вязкость - Андраде (calc_mu_w), теплоемкость
% m*ro_w*Cp_w + (1-m)*rho_R*Cp_R, теплопроводность m*lambda_w + (1-m)*lambda_R, теплообмена с кровлей и подошвой
% нет. Отличие: энтальпия MRST h = u + p/ro, то есть в его уравнении энергии есть нагрев трением -v*grad p
% (computeFlashGeothermal, источники скважин), не больше (p_inj - p_prod)/(ro_w*Cp_w) ~ 2.4 K; в paraphin его нет.
%
% Тест (common.run_mrst):  octave-cli --no-gui --eval "addpath('<репозиторий>/tests/mrst_tests');
%                            mrst_thermal('<имя>.case.json', '<имя>.json')"
% Вручную - из tests/mrst_tests в MATLAB или Octave без аргументов (20x20, 1000 сут): r = mrst_thermal;
if nargin < 1, case_file = ''; end
if nargin < 2, out_file = ''; end
c = mrst_start(case_file);
if isempty(case_file)
    c.nx = 20;
    c.ny = 20;
    c.n_steps = 1000;
    c.field_steps = [100; 250; 500; 1000];
end

G = computeGeometry(cartGrid([c.nx, c.ny, 1], [c.Lx, c.Ly, c.h]));
rock = makeRock(G, c.k, c.m);
rock = addThermalRockProps(rock, 'lambdaR', c.lambda_R, 'rhoR', c.rho_R, 'CpR', c.Cp_R);
fluid = initSimpleADIFluid('phases', 'W', 'mu', c.mu_w, 'rho', c.ro_w);
fluid = addThermalFluidProps(fluid, 'Cp', c.Cp_w, 'lambdaF', c.lambda_w, 'useEOS', false);
fluid.muW = @(p, T, varargin) 2.414e-5 * 10.^(247.8 ./ (T - 140.0));  % Андраде, T в K
model = GeothermalModel(G, rock, fluid);

% Писман и четверть скважины - как в mrst_two_phase_impes.m; температура закачки - T_inj
W = verticalWell([], G, rock, 1, 1, [], 'Type', 'bhp', 'Val', c.p_inj, 'Radius', c.rw, ...
                 'Comp_i', 1, 'Sign', 1, 'Name', 'inj');
W = verticalWell(W, G, rock, c.nx, c.ny, [], 'Type', 'bhp', 'Val', c.p_prod, 'Radius', c.rw, ...
                 'Comp_i', 1, 'Sign', -1, 'Name', 'prod');
for w = 1:numel(W)
    W(w).WI = W(w).WI * c.wi_mult;
end
W = addThermalWellProps(W, G, rock, fluid, 'T', c.T_inj);

state0 = initResSol(G, c.p0, 1);
state0.T = repmat(c.T0, G.cells.num, 1);
r = mrst_run('ad', G, rock, fluid, W, state0, c, model);
mrst_output(r, out_file);
end
