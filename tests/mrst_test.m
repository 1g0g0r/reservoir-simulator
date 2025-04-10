run startup.m
mrstModule add incomp ad-core ad-blackoil


%% Geometry
Lx = 400; Ly = 400; Lz = 1;
nx = 80; ny = 80; nz = 1;
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
k = 0.3*darcy;
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
                           'rho',[rho_w, rho_o], ...
                           'n',  [2, 2]);
model = TwoPhaseOilWaterModel(G, rock, fluid);

%% Add wells
rw = 0.1;
p_inj  = 60 * barsa();
p_prod = 40 * barsa();

W = verticalWell([], G, rock, 1, 1, [],...
                 'Type', 'bhp', 'Val', p_inj, ...
                 'Radius', rw,  'Comp_i', [0, 1]);

W = verticalWell(W, G, rock, nx, ny, [],...
                 'Type', 'bhp' , 'Val', p_prod, ...
                 'Radius', rw, 'Comp_i', [1, 0]);

%% Create a initialized state and set initial saturation to phase 1.
init_sol = initResSol(G, p_prod, [1, 0]);

%% Start calculation
dt = 1 * day;
t_end = 100 * day;
dt_pict = 250 * dt;

%% Start simulation
schedule = simpleSchedule(repmat(dt, [t_end / day, 1]), 'W', W);
[wellsData, fieldData] = simulateScheduleAD(init_sol, model, schedule);

times = cumsum(schedule.step.val) / day;
q_inj = cellfun(@(ws) ws(1).qTs, wellsData);
Q_inj = cumsum(q_inj * dt);
q_prod = -cellfun(@(ws) ws(2).qTs, wellsData);
Qo_prod = cumsum(q_prod * dt);
eta    = cellfun(@(ws) ws(2).ocut, wellsData);

re = 0.14 * sqrt(hx*hx + hy*hy);
q_mult = 2.0 * pi * k * Lz / log(re/rw);  
p_array_w = cellfun(@(ws) (ws.pressure(1,1)-p_inj), fieldData);
p_array_o = cellfun(@(ws) (ws.pressure(nx * ny, 1)-p_prod), fieldData);
peaceman_q_w = -p_array_w .* (q_mult / mu_o);
peaceman_q_o = p_array_o .* (q_mult / mu_w);


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
peaceman_q_o = peaceman_q_o * day();
peaceman_q_w = peaceman_q_w * day();

figure;
plot(times, q_prod, 'LineWidth', 2); 
hold on;
plot(times, peaceman_q_o, '--', 'LineWidth', 2); 
xlabel('t, дни');
ylabel('q, м^3/сут');
legend('reference', 'peaceman');
title('График дебита добывающей скважины');
grid on;

figure;
plot(times, q_inj, 'LineWidth', 2); 
hold on;
plot(times, peaceman_q_w, '--', 'LineWidth', 2); 
xlabel('t, дни');
ylabel('q, м^3/сут');
legend('reference', 'peaceman');
title('График приемистости нагнетательной скважины');
grid on;


