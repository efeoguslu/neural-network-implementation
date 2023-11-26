clc, clear, close all;

%% Data

train = [
    0, 0; 
    1, 2; 
    2, 4; 
    3, 6; 
    4, 8];


%% Main

w = rand * 10;

eps = 1e-3;
rate = 1e-3;

c = cost(2, train);

%% Functions

function result = cost(w, train)
    result = 0.0;
    n = size(train, 1);

    for i = 1:n
        x = train(i, 1);
        y = x * w;
        d = y - train(i, 2);
        result = result + d * d;
    end

    result = result / n;
end


