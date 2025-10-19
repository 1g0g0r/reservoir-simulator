run startup.m

%% calculate
G = cartGrid([20, 20, 1], [200, 200, 2]);
G = computeGeometry(G);
% Disable gravity
gravity off
% Set up uniform permeability and constant porosity
rock.perm = repmat(100*milli*darcy, [G.cells.num, 1]);
rock.poro = repmat(0.5            , [G.cells.num, 1]);
% A simple two phase system.
fluid = initSimpleFluid('mu' , [   1,  10]*centi*poise     , ...
                        'rho', [1014, 859]*kilogram/meter^3, ...
                        'n'  , [   2,   2]);
% Two wells, an injector at 1 bar and a producer at 0 bar.
W = verticalWell([], G, rock, 1, 1, [], ...
            'Type', 'bhp' , 'Val', 1*barsa(), ...
            'Radius', 0.1, 'InnerProduct', 'ip_tpf', ...
            'Comp_i', [0, 1], 'Name', 'Wi');

W = verticalWell(W, G, rock, 20, 20, [], ...
            'Type', 'bhp' , 'Val', 0*barsa(), ...
            'Radius', 0.1, 'InnerProduct', 'ip_tpf', ...
            'Comp_i', [1, 0], 'Name', 'Wp');

% Create a initialized state and set initial saturation to phase 1.
sol = initState(G, [], 0, [1, 0]);

% Find transmissibility.
T = computeTrans(G, rock);

% Reference TPFA
psolve = @(state) incompTPFA(state, G, T, fluid, 'wells', W);

% Implicit transport solver
tsolve   = @(state, dT) implicitTransport(state, G, dT, rock, ...
                                                fluid, 'wells', W);


%% PlotGrid
clf;
plotGrid(G)  % plotGrid(G, 'EdgeAlpha', 0.1, 'FaceColor', 'blue')
view(30,50)


%% plotGrid and subsets
clf;
equal_index = mod(1:G.cells.num,2) == 0;
plotGrid(G,  equal_index, 'FaceColor', 'red')
plotGrid(G, ~equal_index, 'FaceColor', 'blue')
view(30,50)


%% plotCellData
sol= psolve(sol);
clf;
plotCellData(G, sol.pressure)
colorbar
view(30,50)


%% plotCellData with subsets and plotWell
[i j k] = ind2sub(G.cartDims, 1:G.cells.num);
clf;
plotGrid(G, 'FaceAlpha', 0, 'EdgeAlpha', .1)
plotCellData(G, sol.pressure, j == round(G.cartDims(2)/2))
% Plot the wells
plotWell(G, W);
view(30,50)


%% Plot Faces with positive normals in z direction
clf;
plotFaces(G, find(G.faces.normals(:,3)>0));
view(30,50);
plotGrid(G, 'FaceAlpha', 0, 'EdgeAlpha', .1)


clf;
plotGrid(G, 'FaceAlpha', 0.1, 'EdgeAlpha', .5, 'FaceColor', 'black')
plotWell(G, W);
view(30,50);


%% Animated example
dT = 10*day;
for i = 1:60
    sol = tsolve(sol, dT);
    sol = psolve(sol);
    clf;
    plotCellData(G, sol.s(:,2), sol.s(:,2)>0.05)
    plotGrid(G, 'FaceAlpha', 0, 'EdgeAlpha', .1)
    plotWell(G, W);
    view(30,50);
    pause(.3)
end
