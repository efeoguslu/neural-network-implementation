clc, clear, close all;

%% Data

train_data = [
    0, 0, 0;
    1, 0, 1;
    0, 1, 1;
    1, 1, 0;
];

%% Main

m = rand_xor();

eps = 1e-1;
rate = 1e-1;

for i = 1:100*1000
    g = finite_diff(m, eps, train_data);
    m = learn(m, g, rate);
    % Uncomment the line below to display the cost during iterations
    % fprintf('Iteration %d: cost = %f\n', i, cost(m, train_data));
end

fprintf('Final cost = %f\n', cost(m, train_data));

fprintf('------------------------------\n');
for i = 0:1
    for j = 0:1
        fprintf('%d ^ %d = %f\n', i, j, forward(m, i, j));
    end
end

fprintf('------------------------------\n');
fprintf('"OR" neuron:\n');
for i = 0:1
    for j = 0:1
        fprintf('%d | %d = %f\n', i, j, sigmoidf(m.or_w1*i + m.or_w2*j + m.or_b));
    end
end

fprintf('------------------------------\n');
fprintf('"NAND" neuron:\n');
for i = 0:1
    for j = 0:1
        fprintf('~(%d & %d) = %f\n', i, j, sigmoidf(m.nand_w1*i + m.nand_w2*j + m.nand_b));
    end
end

fprintf('------------------------------\n');
fprintf('"AND" neuron:\n');
for i = 0:1
    for j = 0:1
        fprintf('%d & %d = %f\n', i, j, sigmoidf(m.and_w1*i + m.and_w2*j + m.and_b));
    end
end

%% Functions

function result = cost(m, train)
    result = 0.0;
    n = size(train, 1);

    for i = 1:n
        x1 = train(i, 1);
        x2 = train(i, 2);
        y = forward(m, x1, x2);
        d = y - train(i, 3);
        result = result + d * d;
    end

    result = result / n;
end

function result = sigmoidf(x)
    result = 1 ./ (1 + exp(-x));
end

function output = forward(m, x1, x2)
    a = sigmoidf(m.or_w1*x1 + m.or_w2*x2 + m.or_b);
    b = sigmoidf(m.nand_w1*x1 + m.nand_w2*x2 + m.nand_b);
    output = sigmoidf(a*m.and_w1 + b*m.and_w2 + m.and_b);
end

function m = rand_xor()
    m.or_w1 = rand();
    m.or_w2 = rand();
    m.or_b = rand();
    m.nand_w1 = rand();
    m.nand_w2 = rand();
    m.nand_b = rand();
    m.and_w1 = rand();
    m.and_w2 = rand();
    m.and_b = rand();
end

function m = learn(m, g, rate)
    m.or_w1 = m.or_w1 - rate * g.or_w1;
    m.or_w2 = m.or_w2 - rate * g.or_w2;
    m.or_b = m.or_b - rate * g.or_b;
    m.nand_w1 = m.nand_w1 - rate * g.nand_w1;
    m.nand_w2 = m.nand_w2 - rate * g.nand_w2;
    m.nand_b = m.nand_b - rate * g.nand_b;
    m.and_w1 = m.and_w1 - rate * g.and_w1;
    m.and_w2 = m.and_w2 - rate * g.and_w2;
    m.and_b = m.and_b - rate * g.and_b;
end

function g = finite_diff(m, eps, train)
    g = struct(); % Initialize the gradient struct

    c = cost(m, train);

    % Compute partial derivatives using finite differences
    fields = fieldnames(m);
    for i = 1:numel(fields)
        field = fields{i};

        % Save the original value
        saved = m.(field);

        % Perturb the parameter
        m.(field) = m.(field) + eps;

        % Compute the perturbed cost
        perturbed_cost = cost(m, train);

        % Compute the finite difference gradient
        g.(field) = (perturbed_cost - c) / eps;

        % Restore the original value
        m.(field) = saved;
    end
end
