run startup.m
mrstModule add incomp ad-core ad-blackoil


%% Geometry
Lx = 300; Ly = 300; Lz = 1;
nx = 128; ny = 128; nz = 1;
G = cartGrid([nx, ny, nz], [Lx, Ly, Lz]);
G = computeGeometry(G);
gravity off

%% Set permeability and porosity
k = 0.3*darcy;
m = 0.2;
rock.perm = repmat(k, [G.cells.num, 1]);
rock.poro = repmat(m, [G.cells.num, 1]);

%% Relative phase permeability
alpha = 2;
x = linspace(0, 1, 100) .';
y = linspace(1, 0, 100) .';
kr  = tabulatedSatFunc([x, x.^alpha, y.^alpha]);

%% Define constant properties for viscosity and density
kmu = 0.5;
props = constantProperties([ 1.0, 5.0] .* centi*poise, ...
                           [1000, 860] .* kilogram/meter^3);
fluid = struct('properties', props                  , ...
               'saturation', @(x, varargin)    x.s  , ...
               'relperm'   , kr);

%% Add wells
rw = 0.1;
p_inj  = 17 * barsa();
p_prod = 10 * barsa();

W = verticalWell([], G, rock, 1, 1, [],...
                 'Type', 'bhp', 'Val', p_inj, ...
                 'Radius', rw, 'InnerProduct', 'ip_tpf', ...
                 'Comp_i', [0, 1]);

W = verticalWell(W, G, rock, nx, ny, [],...
                 'Type', 'bhp' , 'Val', p_prod, ...
                 'Radius', rw, 'InnerProduct', 'ip_tpf', ...
                 'Comp_i', [1, 0]);

%% Create a initialized state and set initial saturation to phase 1.
sol = initState(G, [], 0, [1, 0]);

%% Find transmissibility.
T = computeTrans(G, rock);

%% Reference TPFA
psolve = @(state) incompTPFA(state, G, T, fluid, 'wells', W);

%% Implicit transport solver
tsolve = @(state, dT) implicitTransport(state, G, dT, rock, ...
                                        fluid, 'wells', W);

%% Start calculation
dT = 5 * day;
t  = dT;
times = [0];
t_end = 50 * day;

dT_pict = 2 * dT;
i_pict = 0;

x_coord = linspace(0, Lx, nx);
y_coord = linspace(0, Ly, ny);
z_coord = linspace(0, Lz, nz);

Q_inj  = [0];
q_inj  = [0];
Q_prod = [0];
q_prod = [0];
sol = psolve(sol);
while t < t_end
    sol = tsolve(sol, dT);
    sol = psolve(sol);
    p_arr = reshape(sol.pressure/barsa(), [nx, ny, nz]);
    s_arr = reshape(sol.s(:, 2), [nx, ny, nz]);
    
    q_inj  = [q_inj, sum(sol.wellSol(1).flux)];
    q_prod = [q_prod, -sum(sol.wellSol(2).flux)];
    Q_inj  = [Q_inj, Q_inj(end) + q_inj(end) * dT];
    Q_prod = [Q_prod, Q_prod(end) + q_prod(end) * dT];

    if t_end - t <= dT  %t >= i_pict * dT_pict
        figure    
        % трехмерные p(x,y,z)
        subplot(1, 2, 1);
        plotCellData(G, sol.pressure/barsa(), 'EdgeColor', 'none');  
        title(['P(x, y, z) в  ', num2str(convertTo(t,day)),  ' день'])  
        colorbar; 
        view(0, 90);
        
        % трехмерные s(x,y,z)
        subplot(1, 2, 2);
        plotCellData(G, sol.s(:, 2), 'EdgeColor', 'none');  
        title(['S(x, y, z) в  ', num2str(convertTo(t,day)),  ' день'])  
        colorbar; 
        view(0, 90);

        i_pict = i_pict + 1;
    end

    t = t + dT;
    times = [times, convertTo(t,day)];
end

%{
figure;
plot(times, Q_prod); 
title('График накопленного дебита нефти добывающей скважины');
xlabel('t, дни');
ylabel('Q, м^3');
grid on;

figure;
plot(times, Q_inj); 
title('График накопленной приемистости нагнетательной скважины');
xlabel('t, дни');
ylabel('Q, м^3');
grid on;

figure;
plot(times, q_prod); 
title('График дебита добывающей скважины');
xlabel('t, дни');
ylabel('q, м^3/день');
grid on;

figure;
plot(times, q_inj); 
title('График приемистости нагнетательной скважины');
xlabel('t, дни');
ylabel('q, м^3/день');
grid on;
%}

