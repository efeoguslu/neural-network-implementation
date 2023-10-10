#include "activation.h"

// 1 - 56.06


// the training set:

double train[][2] = {
    {0, 0},
    {1, 3},
    {2, 6},
    {3, 9},
    {4, 12},
};

#define train_count (sizeof(train)/sizeof(train[0]))

// returns random double from 0 to 10 (not quite there yet, we'll try again)

/*
double rand_double(void){
    return ((double)rand()/(double)RAND_MAX) * 10.0;
}
*/


double rand_double(double min, double max)
{
    double scale = rand() / (double)RAND_MAX; // [0, 1.0] 
    return min + scale * (max - min);         // [min, max] 
}



// --------------------------------------------------------------------------------------
// cost function: we want it to go to 0.

// x1, x2, x3, ...
// w1, w2, w3, ...
// y = x1*w1 + x2*w2 + x3*w3 + ... + bias

double cost(double w, double b){

    double result = 0.0;

    for(size_t i = 0; i < train_count; ++i){
        double x = train[i][0];
        double y = x*w + b; // single artificial neuron mathematical model
        double d = y - train[i][1];
        result += d*d;

        //printf("actual : %f, expected: %f\n", y, train[i][1]);
    }

    result /= train_count;

    return result;
}


int main(){

    menu();
    randomize();
    
    // mathematical model:
    // y = x * w + bias ; --> output = input * weight + bias 
    
    double w = rand_double(0.0, 10.0); // random number from 0 to 10
    double b = rand_double(0.0, 5.0);  // random number from 0 to 5

    double eps = 1e-3;
    double rate = 1e-3;


    for(size_t i = 0; i < 10000; ++i){
        
        double c = cost(w, b);
        // finite difference: approximation of derivative 
        double dw = ((cost(w + eps, b) - c)/eps);
        double db = ((cost(w, b + eps) - c)/eps);

        // 
        w -= rate*dw;
        b -= rate*db;

        printf("cost = %f, w = %f, b = %f\n", cost(w, b), w, b);

    }

    dline();
    printf("w = %f, b = %f\n", w, b);

}