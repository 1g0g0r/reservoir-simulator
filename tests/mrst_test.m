run startup.m
mrstModule add incomp ad-core ad-blackoil 


%% Geometry
Lx = 250; Ly = 250; Lz = 1;
nx = 10; ny = 10; nz = 1;
hx = Lx / nx; hy = Ly / ny;
G = cartGrid([nx, ny, nz], [Lx, Ly, Lz]);
G = computeGeometry(G);
gravity off

%% Additional constants
x_coord = linspace(0, Lx, nx);
y_coord = linspace(0, Ly, ny);

n_isolines = 11;
color_steps = 128;
sat_colormap = [linspace(0.6, 0.0  , color_steps)', ... 
                linspace(0.4, 0.447, color_steps)', ... 
                linspace(0.2, 0.741, color_steps)'];

%% Set permeability and porosity
k = 0.2*darcy;
m = 0.2;
rock = makeRock(G, k, m);

%% Define constant properties for viscosity and density
mu_w = 1 * centi*poise;
mu_o = 5 * centi*poise;
rho_w = 1000 * kilogram/meter^3;
rho_o = 860 * kilogram/meter^3;
                            % [вода, нефть]
fluid = initSimpleADIFluid('phases', 'WO', ...
                           'mu', [mu_w, mu_o], ...
                           'rho',[rho_w, rho_o]);
relperm = struct('type', 'corey', ...
                 'params', struct('nw', 2, 'no', 2, 'swr', 0.2, 'sor', 0.3));

% Создание модели с явным указанием relperm
model = TwoPhaseOilWaterModel(G, rock, fluid, 'relperm', relperm);

%% Add wells
rw = 0.1;
p_inj  = 150 * barsa();
p_prod = 50 * barsa();

W = verticalWell([], G, rock, 1, 1, [],...
                 'Type', 'bhp', 'Val', p_inj, ...
                 'Radius', rw,  'Comp_i', [0, 1]);

W = verticalWell(W, G, rock, nx, ny, [],...
                 'Type', 'bhp' , 'Val', p_prod, ...
                 'Radius', rw, 'Comp_i', [1, 0]);

%% Create a initialized state and set initial saturation to phase 1.
init_sol = initResSol(G, p_prod, [0.8, 0.2]);

%% Start calculation
dt = 1 * day;
t_end = 100 * day;
dt_pict = 2500 * dt;

%% Start simulation
schedule = simpleSchedule(repmat(dt, [t_end / day, 1]), 'W', W);
[wellsData, fieldData] = simulateScheduleAD(init_sol, model, schedule);

times = cumsum(schedule.step.val) / day;
q_inj = cellfun(@(ws) ws(1).qTs, wellsData);
Q_inj = cumsum(q_inj * dt);
q_prod = -cellfun(@(ws) ws(2).qTs, wellsData);
Qo_prod = cumsum(q_prod * dt);
eta    = cellfun(@(ws) ws(2).ocut, wellsData);

step = int32(dt_pict / dt);
end_idx = int32(length(times));
for i = unique([1:step:end_idx, end_idx])
    t = times(i);
    p_arr = reshape(fieldData{i, 1}.pressure/barsa(), [nx, ny, nz]);
    s_arr = reshape(fieldData{i, 1}.s(:, 2), [nx, ny, nz]);

    figure    
    % двумерные p(x,y) и s(x,y)
    subplot(1, 2, 1);
    [C, h] = contourf(x_coord, y_coord, p_arr, n_isolines);
    clabel(C, h, 'FontSize', 5, 'Color', 'k');  
    title(['P(x, y) в  ', num2str(t),  ' сут'])  
    colorbar; 
    xlabel('X, метры');
    ylabel('Y, метры');
    sp = subplot(1, 2, 2);
    contourf(x_coord, y_coord, s_arr.', n_isolines)
    title(['S(x, y) в  ', num2str(t),  ' сут'])  
    colorbar; 
    colormap(sp, sat_colormap)
    xlabel('X, метры');
    ylabel('Y, метры');
end

q_prod = q_prod * day();
q_inj = q_inj * day();

figure;
plot(times, q_prod, 'LineWidth', 2); 
plot(times, q_inj, '--', 'LineWidth', 2); 
hold on;
xlabel('t, дни');
ylabel('q, м^3/сут');
title('График дебета скважины');
grid on;

