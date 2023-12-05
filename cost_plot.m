% fileID = fopen('cost.txt','r');
% formatSpec = '%lf';
% A = fscanf(fileID, formatSpec);
% fclose(fileID);
% 
% figure;
% plot(A);
% title("Cost Function");
% xlabel("Epoch");
% ylabel("Value of Cost Func");


% Read data from the cost file
cost_data = dlmread('cost.txt');


figure(1);

plot(cost_data, '-', 'Color', [0, 0.4470, 0.7410], 'LineWidth', 2);  % Blue color
title('Cost Function Over Epochs');
xlabel('Epoch');
ylabel('Cost');
grid on;

saveas(gcf, 'cost_plot.png');