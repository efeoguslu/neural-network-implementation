fileID = fopen('params.txt','r');
num_layers = fscanf(fileID, '%d', 1);
neurons_per_layer = fscanf(fileID, '%d', num_layers);
fclose(fileID);

DrawTheNeuralNetwork();

function DrawTheNeuralNetwork()
    fileID = fopen('params.txt','r');
    num_layers = fscanf(fileID, '%d', 1);
    neurons_per_layer = fscanf(fileID, '%d', num_layers);
    fclose(fileID);
    
    figure, hold on
    for i = 1:num_layers
    y = linspace(-max(neurons_per_layer)/2, max(neurons_per_layer)/2, neurons_per_layer(i));
    x = repmat(i, 1, neurons_per_layer(i));
    plot(x, y, 'o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b')
    end
    for i = 1:num_layers-1
    y1 = linspace(-max(neurons_per_layer)/2, max(neurons_per_layer)/2, neurons_per_layer(i));
    x1 = repmat(i, 1, neurons_per_layer(i));
    y2 = linspace(-max(neurons_per_layer)/2, max(neurons_per_layer)/2, neurons_per_layer(i+1));
    x2 = repmat(i+1, 1, neurons_per_layer(i+1));
    for j = 1:neurons_per_layer(i)
      for k = 1:neurons_per_layer(i+1)
          line([x1(j)';x2(k)'], [y1(j)';y2(k)'], 'Color', 'r')
      end
    end
    end
    axis equal
    hold off
end

