clc; clear; close all;
%% 

canvas_side = 1000;

% Ask user for number of layers and number of neurons in each layer
numLayers = input('Enter number of layers: ');
neuronsPerLayer = input('Enter number of neurons in each layer (comma-separated): ', 's');

% Convert neuronsPerLayer to an array
neuronsPerLayer = str2num(neuronsPerLayer);

% Validate user input
if length(neuronsPerLayer) ~= numLayers
    error('The number of elements in neuronsPerLayer must equal numLayers.');
end

% Create a blank canvas
img = zeros(canvas_side, canvas_side, 3); % Create a 500x500 image with 3 color channels (RGB)
figure;
imshow(img);
hold on;

% Draw the neurons and connections
for i = 1:numLayers
    % Calculate the y-coordinate of the current layer
    y = i * canvas_side / (numLayers + 1);

    % Draw neurons in the current layer
    for j = 1:neuronsPerLayer(i)
        % Calculate the x-coordinate of the current neuron
        x = j * canvas_side / (neuronsPerLayer(i) + 1);

        % Draw the neuron
        viscircles([x y], 20, 'EdgeColor', 'b');

        % If this is not the first layer, draw connections to the previous layer
        if i > 1
            for k = 1:neuronsPerLayer(i-1)
                % Calculate the x-coordinate of the neuron in the previous layer
                x_prev = k * canvas_side / (neuronsPerLayer(i-1) + 1);

                % Draw the connection
                line([x x_prev], [y y-canvas_side/(numLayers+1)], 'Color', 'r');
            end
        end
    end
end

hold off;





%%

% canvas_side = 1000;
% 
% % Ask user for number of layers and number of neurons in each layer
% numLayers = input('Enter number of layers: ');
% neuronsPerLayer = input('Enter number of neurons in each layer (comma-separated): ', 's');
% 
% % Convert neuronsPerLayer to an array
% neuronsPerLayer = str2num(neuronsPerLayer);
% 
% % Validate user input
% if length(neuronsPerLayer) ~= numLayers
%     error('The number of elements in neuronsPerLayer must equal numLayers.');
% end
% 
% % Create a blank canvas
% img = zeros(canvas_side, canvas_side, 3); % Create a 1000x1000 image with 3 color channels (RGB)
% figure;
% imshow(img);
% hold on;
% 
% % Draw the neurons and connections
% for i = 1:numLayers
%     % Calculate the x-coordinate of the current layer
%     x = i * canvas_side / (numLayers + 1);
% 
%     % Draw neurons in the current layer
%     for j = 1:neuronsPerLayer(i)
%         % Calculate the y-coordinate of the current neuron
%         y = j * canvas_side / (neuronsPerLayer(i) + 1);
% 
%         % Draw the neuron
%         viscircles([x y], 20, 'EdgeColor', 'b');
% 
%         % If this is not the first layer, draw connections to the previous layer
%         if i > 1
%             for k = 1:neuronsPerLayer(i-1)
%                 % Calculate the y-coordinate of the neuron in the previous layer
%                 y_prev = k * canvas_side / (neuronsPerLayer(i-1) + 1);
% 
%                 % Draw the connection
%                 line([x x], [y y_prev], 'Color', 'r');
%             end
%         end
%     end
% end
% 
% hold off;
