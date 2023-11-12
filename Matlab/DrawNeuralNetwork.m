clc; clear; close all;

canvas_side = 1000;

numLayers = input('Enter number of layers: ');
neuronsPerLayer = input('Enter number of neurons in each layer (comma-separated): ', 's');
neuronsPerLayer = str2num(neuronsPerLayer);

if length(neuronsPerLayer) ~= numLayers
    error('The number of elements in neuronsPerLayer must equal numLayers.');
end

img = zeros(canvas_side, canvas_side, 3);
figure;
imshow(img);
hold on;

neuronSpacingX = canvas_side / (numLayers + 1);
neuronSpacingY = canvas_side / max(neuronsPerLayer + 1);

for i = 1:numLayers
    x = i * neuronSpacingX;
    
    for j = 1:neuronsPerLayer(i)
        y = j * neuronSpacingY;
        viscircles([x y], 20, 'EdgeColor', 'b');

        if i > 1
            for k = 1:neuronsPerLayer(i-1)
                x_prev = (i - 1) * neuronSpacingX;
                y_prev = k * neuronSpacingY;
                line([x, x_prev], [y, y_prev], 'Color', 'r', 'LineWidth', 2);
            end
        end
    end
end

hold off;
