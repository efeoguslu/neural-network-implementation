% Load trained model weights
weights = load('weights.txt');

% Create a meshgrid for the input space
[x1, x2] = meshgrid(0:0.01:1, 0:0.01:1);

% Initialize array to store predictions
predictions = zeros(size(x1));

% Compute predictions for each point in the meshgrid
for i = 1:size(x1, 1)
    for j = 1:size(x1, 2)
        predictions(i, j) = forward(weights, x1(i, j), x2(i, j));
    end
end

% Plot the decision boundaries
figure;
contour(x1, x2, predictions > 0.5, 1, 'LineColor', 'r', 'LineWidth', 2);
hold on;

% Plot training data points
scatter(train(:, 1), train(:, 2), 50, train(:, 3), 'filled', 'MarkerEdgeColor', 'k');

% Set plot labels
title('XOR Decision Boundaries');
xlabel('Input 1');
ylabel('Input 2');

% Show the plot
hold off;

function output = forward(weights, x1, x2)
    % Extract weights
    or_w1 = weights(1);
    or_w2 = weights(2);
    or_b = weights(3);
    
    nand_w1 = weights(4);
    nand_w2 = weights(5);
    nand_b = weights(6);
    
    and_w1 = weights(7);
    and_w2 = weights(8);
    and_b = weights(9);

    % Compute outputs of the first layer
    a = sigmoid(or_w1 * x1 + or_w2 * x2 + or_b);
    b = sigmoid(nand_w1 * x1 + nand_w2 * x2 + nand_b);
    
    % Return the output of the last layer
    output = sigmoid(a * and_w1 + b * and_w2 + and_b);
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

