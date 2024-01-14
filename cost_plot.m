cost_data = dlmread('cost.txt');

figure(1);

plot(cost_data, '-', 'Color', [0, 0.4470, 0.7410], 'LineWidth', 2);  % Blue color
title('Cost Function Over Epochs');
xlabel('Epoch');
ylabel('Cost');
grid on;

saveas(gcf, 'cost_plot.png');





