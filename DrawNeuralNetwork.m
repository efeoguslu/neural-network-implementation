fileID = fopen('params.txt','r');
num_layers = fscanf(fileID, '%d', 1);
neurons_per_layer = fscanf(fileID, '%d', num_layers);
fclose(fileID);

DrawTheNeuralNetwork();

function DrawTheNeuralNetwork()
    params = dlmread('params.txt');
    num_layers = params(1);
    neurons_per_layer = params(2:end);
    
    max_neurons = max(neurons_per_layer);
    figure;
    set(gcf, 'Color', [0.08 0.08 0.08]);
    hold on;
    
    vertical_spacing = 1 / (max_neurons + 2);
    padding = vertical_spacing;
    
    neuron_positions = cell(num_layers, 1);
    
    for l = 1:num_layers
        num_neurons = neurons_per_layer(l);
        startY = (max_neurons - num_neurons + 1) * vertical_spacing / 2 + padding;
        neuron_positions{l} = zeros(num_neurons, 2);
        
        for n = 1:num_neurons
            x = l * 2;  % Multiply by 2 for spacing between layers
            y = startY + (n-1) * vertical_spacing;
            neuron_positions{l}(n, :) = [x, y];
            plot(x, y, 'o', 'MarkerSize', 30, 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerEdgeColor', 'k'); % Adjust MarkerSize and colors
            
            % Label neurons
            text(x, y, sprintf('a_{%d%d}', l, n), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 8);
        end
    end
    
    % Drawing lines and labeling weights
    for l = 1:(num_layers-1)
        for n = 1:size(neuron_positions{l}, 1)
            for m = 1:size(neuron_positions{l+1}, 1)
                pos1 = neuron_positions{l}(n, :);
                pos2 = neuron_positions{l+1}(m, :);
                line([pos1(1), pos2(1)], [pos1(2), pos2(2)], 'Color', 'y'); 
                
                % Label weights
                midpoint = (pos1 + pos2) / 2;
                text(midpoint(1), midpoint(2), sprintf('w_{%d%d}', n, m), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 8);
            end
        end
    end
    
   % Label layers and adjust their position
layer_spacing = 2; % Assuming layers are spaced by a factor of 2
text(layer_spacing, max_neurons * vertical_spacing + 2 * padding, 'Input Layer', 'HorizontalAlignment', 'center', 'FontSize', 10);
text(layer_spacing * num_layers, max_neurons * vertical_spacing + 2 * padding, 'Output Layer', 'HorizontalAlignment', 'center', 'FontSize', 10);

% Adjust for hidden layers if more than one hidden layer
if num_layers > 2
    for l = 2:(num_layers-1)
        text(layer_spacing * l, max_neurons * vertical_spacing + 2 * padding, sprintf('Hidden Layer %d', l-1), 'HorizontalAlignment', 'center', 'FontSize', 10);
    end
end
    
    % Formatting the plot
    xlim([0, num_layers * 2 + 1]);
    ylim([0, max_neurons * vertical_spacing + (2 * padding)]);
    axis off;
    hold off;
end


