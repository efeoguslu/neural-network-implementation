fileID = fopen('cost.txt','r');
formatSpec = '%lf';
A = fscanf(fileID, formatSpec);
fclose(fileID);

figure;
plot(A);
title("Cost Function");
xlabel("Epoch");
ylabel("Value of Cost Func");
