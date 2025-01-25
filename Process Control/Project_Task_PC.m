% Parameters (define these before running the simulation)
A1 = 3;  % ft^2 (area of tank 1)
A2 = 5;  % ft^2 (area of tank 2)
C1 = 8;  % cfm/ft^0.5 (flow coefficient between tanks)
C2 = 12; % cfm/ft^0.5 (outflow coefficient from tank 2)
qi = 16; % cfm (steady inflow rate)

% Initial conditions
h1_initial = 0; % Initial height in tank 1 (ft)
h2_initial = 0; % Initial height in tank 2 (ft)
y0 = [h1_initial; h2_initial]; % Initial state vector

% Time span for simulation (0 to 200 minutes)
tspan = [0 200];

% Solve ODE using ode45 and pass additional parameters
[t, y] = ode45(@(t, y) two_tank_system(t, y, A1, A2, C1, C2, qi), tspan, y0);

% Extract results for h2 only (before adding noise)
h2 = y(:, 2);

% Introduce noise: Noise is 1% of the steady-state value of h2
noise_variance = 0.01 * h2(end);  % 1% of the final value of h2
noise = noise_variance * randn(size(h2)); % Add random noise with zero mean

% Add noise to the output variable (h2)
h2_noisy = h2 + noise;

% Plotting results for steady-state identification (with noise)
figure;
plot(t, h2_noisy, 'DisplayName', 'h_2(t) with Noise');
xlabel('Time (minutes)');
ylabel('Height in Tank 2 (ft)');
title('Steady-State Identification for Two-Tank System with Noise');
legend show;
grid on;

% Parameters for step changes
step_change_10 = 1.1;  % 10% step increase in inflow rate
step_change_20 = 1.2;  % 20% step increase in inflow rate

% Create figure for plotting
figure;
hold on;

% Step change for +10%
qi_step_10 = qi * step_change_10;
[t_10, y_10] = ode45(@(t, y) two_tank_step(t, y, A1, A2, C1, C2, qi, qi_step_10), tspan, y0);
h2_10 = y_10(:, 2);

% Add noise
noise_10 = noise_variance * randn(size(h2_10));  % Add random noise
h2_noisy_10 = h2_10 + noise_10;  % Noisy output for +10%

% Plot results for +10% step change
plot(t_10, h2_noisy_10, 'DisplayName', 'Step Change: +10% with Noise');

% Step change for +20%
qi_step_20 = qi * step_change_20;
[t_20, y_20] = ode45(@(t, y) two_tank_step(t, y, A1, A2, C1, C2, qi, qi_step_20), tspan, y0);
h2_20 = y_20(:, 2);

% Add noise
noise_20 = noise_variance * randn(size(h2_20));  % Add random noise
h2_noisy_20 = h2_20 + noise_20;  % Noisy output for +20%

% Plot results for +20% step change
plot(t_20, h2_noisy_20, 'DisplayName', 'Step Change: +20% with Noise');

% Customize the plot
xlabel('Time (minutes)');
ylabel('Height in Tank 2 (ft)');
title('Open-Loop Simulation for Different Step Changes in Inflow (with Noise)');
legend show;
grid on;
hold off;


% Function Definitions (already defined before)
function dydt = two_tank_system(t, y, A1, A2, C1, C2, qi)
    h1 = y(1); % Height in tank 1
    h2 = y(2); % Height in tank 2
    
    % Flow calculations (ensuring no negative square roots)
    q1 = C1 * sqrt(max(h1 - h2, 0)); % Flow from tank 1 to tank 2
    q2 = C2 * sqrt(max(h2, 0));      % Outflow from tank 2

    % Differential equations
    dh1_dt = (qi - q1) / A1;
    dh2_dt = (q1 - q2) / A2;
    
    dydt = [dh1_dt; dh2_dt];
end

function dydt = two_tank_step(t, y, A1, A2, C1, C2, qi_initial, qi_step)
    h1 = y(1); % Height in tank 1
    h2 = y(2); % Height in tank 2

    % Apply step change at t = 100 minutes
    if t >= 100
        qi = qi_step; % New inflow rate after step
    else
        qi = qi_initial; % Initial inflow rate
    end

    % Flow calculations
    q1 = C1 * sqrt(max(h1 - h2, 0)); % Flow from tank 1 to tank 2
    q2 = C2 * sqrt(max(h2, 0));      % Outflow from tank 2

    % Differential equations
    dh1_dt = (qi - q1) / A1;
    dh2_dt = (q1 - q2) / A2;

    dydt = [dh1_dt; dh2_dt];
end

% ------------------------------
% TRAINING AND VALIDATION SPLIT
% ------------------------------

% Combine data for 10% step change
t_combined_10 = t_10; % Time vector for 10% step change
noisy_y_combined_10 = h2_noisy_10; % Noisy response for 10% step change

% Combine data for 20% step change
t_combined_20 = t_20; % Time vector for 20% step change
noisy_y_combined_20 = h2_noisy_20; % Noisy response for 20% step change


% Split the data into training (70%) and validation (30%)
train_ratio = 0.7; % 70% for training
train_size_10 = round(train_ratio * length(t_combined_10));
train_size_20 = round(train_ratio * length(t_combined_20));

% Training data (first 70% of the data)
time_train_10 = t_combined_10(1:train_size_10);
response_train_10 = noisy_y_combined_10(1:train_size_10);

time_train_20 = t_combined_20(1:train_size_20);
response_train_20 = noisy_y_combined_20(1:train_size_20);

% Validation data (last 30% of the data)
time_val_10 = t_combined_10(train_size_10+1:end);
response_val_10 = noisy_y_combined_10(train_size_10+1:end);

time_val_20 = t_combined_20(train_size_20+1:end);
response_val_20 = noisy_y_combined_20(train_size_20+1:end);


% Define the FOPTD models for 10% and 20% step changes
foptd_model_10 = @(params, t) steady_state_value + M_10 * params(1) * (1 - exp(-(t - params(3)) / params(2))) .* (t >= params(3));
foptd_model_20 = @(params, t) steady_state_value + M_20 * params(1) * (1 - exp(-(t - params(3)) / params(2))) .* (t >= params(3));

% Cost function for nonlinear regression (only on training data)
cost_function_10 = @(params) sum((response_train_10 - foptd_model_10(params, time_train_10)).^2);
cost_function_20 = @(params) sum((response_train_20 - foptd_model_20(params, time_train_20)).^2);

% Initial guesses and bounds for Kp, Tau, and Theta
initial_guess = [1, 5, 5]; % Adjust based on expected behavior
lower_bounds = [0, 0, 0];    % Non-negative constraints for Kp, Tau, and Theta
upper_bounds = [5, 50, 100]; % Upper limits to guide the optimization

% Define optimization options
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp');

% Perform nonlinear regression for 10% step change
params_10 = fmincon(cost_function_10, initial_guess, [], [], [], [], lower_bounds, upper_bounds, [], options);

% Perform nonlinear regression for 20% step change
params_20 = fmincon(cost_function_20, initial_guess, [], [], [], [], lower_bounds, upper_bounds, [], options);

% Generate predictions using the optimized parameters on validation data
y_pred_val_10 = foptd_model_10(params_10, time_val_10);
y_pred_val_20 = foptd_model_20(params_20, time_val_20);

% Calculate validation errors (MSE)
mse_val_10 = mean((response_val_10 - y_pred_val_10).^2);
mse_val_20 = mean((response_val_20 - y_pred_val_20).^2);

% Display results
fprintf('Optimized parameters for 10%% step change: Kp = %.2f, Tau = %.2f, Delay = %.2f\n', params_10(1), params_10(2), 100-params_10(3));
fprintf('Optimized parameters for 20%% step change: Kp = %.2f, Tau = %.2f, Delay = %.2f\n', params_20(1), params_20(2), 100-params_20(3));
mean_K = (params_10(1) + params_20(1))/2;
mean_Tau = (params_10(2) + params_20(2))/2;
mean_Theta = (200-(params_10(3) + params_20(3)))/2
fprintf('Final parameters for +M step change: Kp = %.2f, Tau = %.2f, Delay = %.2f\n', mean_K, mean_Tau, mean_Theta);

% Display the validation MSE
fprintf('Validation MSE for 10%% step change: %.4f\n', mse_val_10);
fprintf('Validation MSE for 20%% step change: %.4f\n', mse_val_20);

% Generate predictions using the optimized parameters
y_pred_10 = foptd_model_10(params_10, t_combined_10);
y_pred_20 = foptd_model_20(params_20, t_combined_20);


% Plot actual vs fitted model for 10% step change
figure;
subplot(2,1,1);
plot(t_10, h2_noisy_10, 'b', 'LineWidth', 1);
hold on;
plot(t_combined_10, y_pred_10, 'r--', 'LineWidth', 1);
xlabel('Time (minutes)');
ylabel('Tank 2 Level (h_2)');
title('FOPTD Model Fit for 10% Step Change');
legend('Actual Data', 'FOPTD Model');
grid on;

% Plot actual vs fitted model for 20% step change
subplot(2,1,2);
plot(t_20,  h2_noisy_20, 'b', 'LineWidth', 1);
hold on;
plot(t_combined_20, y_pred_20, 'r--', 'LineWidth', 1);
xlabel('Time (minutes)');
ylabel('Tank 2 Level (h_2)');
title('FOPTD Model Fit for 20% Step Change');
legend('Actual Data', 'FOPTD Model');
grid on;

% Optionally, plot validation results
figure;
subplot(2,1,1);
plot(time_val_10, response_val_10, 'b', 'LineWidth', 1);
hold on;
plot(time_val_10, y_pred_val_10, 'r--', 'LineWidth', 1);
xlabel('Time (minutes)');
ylabel('Tank 2 Level (h_2)');
title('FOPTD Model Validation for 10% Step Change');
legend('Validation Data', 'FOPTD Model');
grid on;

subplot(2,1,2);
plot(time_val_20, response_val_20, 'b', 'LineWidth', 1);
hold on;
plot(time_val_20, y_pred_val_20, 'r--', 'LineWidth', 1);
xlabel('Time (minutes)');
ylabel('Tank 2 Level (h_2)');
title('FOPTD Model Validation for 20% Step Change');
legend('Validation Data', 'FOPTD Model');
grid on;



% Define the transfer function G(s) = Kp / (Tau*s + 1)
num = [mean_K];
den = [mean_Tau, 1];
G = tf(num, den);

% Find the range of Kc for stability using Routh-Hurwitz Criterion
Kc_stable_min = -1 / mean_K;  % Lower limit for stability
fprintf('The system is stable for Kc > %.2f\n', Kc_stable_min);

% Ask the user to input Kc value
Kc = input('Enter the value of proportional gain Kc: ');

% Root Locus Plot
figure;
rlocus(G);
title('Root Locus of the Closed-Loop System');
xlabel('Real Axis');
ylabel('Imaginary Axis');
grid on;


% Closed-loop transfer function: T(s) = (Kc*Kp) / (Tau*s + 1 + Kc*Kp)
T = feedback(Kc*G, 1);

% Simulation time vector: total 3000 samples
t = linspace(0, 3000, 3000);  

% Setpoint for servo problem
setpoint = [ones(1, 500), 1.05*ones(1, 1500), 1.05*ones(1, 1000)];
setpoint1 = [ones(1, 500), ones(1, 1500), ones(1, 1000)];  % Constant setpoint for comparison

% Disturbance: Introduced at sample 2000
disturbance = zeros(1, length(t));
disturbance(2000:end) = 0.05;  

% Simulate the system's response to setpoint and disturbance
[y, t_out] = lsim(T, setpoint, t);
[y_disturbed, ~] = lsim(T, disturbance, t);

% Total response (setpoint + disturbance)
y_total = y + y_disturbed;


figure;
% Subplot 1: Plot the setpoint and controlled variable (no disturbance)
subplot(2, 1, 1); 
plot(t_out, setpoint, 'r--', 'LineWidth', 1.5); hold on;
plot(t_out, y, 'b', 'LineWidth', 1.5);
legend('Setpoint', 'Controlled Variable (CV)');
xlabel('Time (samples)');
ylabel('Response');
title('Closed-Loop Response with Proportional Control');
grid on;
% Subplot 2: Response to disturbance alone
subplot(2, 1, 2);  
plot(t_out, setpoint1, 'r--', 'LineWidth', 1.5); hold on;
plot(t_out, y_disturbed, 'b', 'LineWidth', 1.5);
legend('Setpoint', 'Controlled Variable (CV) with Disturbance');
xlabel('Time (samples)');
ylabel('Response');
title('Response with Disturbance Only');
grid on;

% Plot of Total response (setpoint + disturbance)
figure;
plot(t_out, setpoint, 'r--', 'LineWidth', 1.5); hold on;
plot(t_out, y_total, 'b', 'LineWidth', 1.5);
legend('Setpoint', 'Total Controlled Variable (CV) with Disturbance');
xlabel('Time (samples)');
ylabel('Total Response');
title('Total Response: Setpoint + Disturbance');
grid on;
