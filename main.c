#include "activation.h"
#include "helpers.h"

// 1 - 

// OR-gate

double or_train[][3] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

// AND-gate

double and_train[][3] = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 1},

};

// the simple training set:

double train[][2] = {
    {0, 0},
    {1, 3},
    {2, 6},
    {3, 9},
    {4, 12},
};

#define train_count (sizeof(train)/sizeof(train[0]))
#define or_train_count (sizeof(or_train) / sizeof(or_train[0]))
#define and_train_count (sizeof(and_train)/sizeof(and_train[0]))

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

double cost_two_inputs(double w1, double w2){

    double result = 0.0;

    for(size_t i = 0; i < or_train_count; ++i){
        double x1 = or_train[i][0];
        double x2 = or_train[i][1];
        double y = x1*w1 + x2*w2; // single artificial neuron mathematical model (with two inputs)
        double d = y - or_train[i][1];
        result += d*d;

        //printf("actual : %f, expected: %f\n", y, train[i][1]);
    }

    result /= or_train_count;

    return result;
}


int main(){

    menu();
    randomize();
    
    // mathematical model:
    // y = x * w + bias ; --> output = input * weight + bias 
    
    double w1 = rand_double(0.0, 10.0); // random number from 0 to 10
    double w2 = rand_double(0.0, 10.0);

    // double b = rand_double(0.0, 5.0);  // random number from 0 to 5

    double eps = 1e-3;
    double rate = 1e-3;


    for(size_t i = 0; i < 1000; ++i){
        
        double c = cost_two_inputs(w1, w2);
        // finite difference: approximation of derivative 
        double dw1 = ((cost_two_inputs(w1 + eps, w2) - c)/eps);
        double dw2 = ((cost_two_inputs(w1, w2 + eps) - c)/eps);
        // double db = ((cost(w, b + eps) - c)/eps);

        // 
        w1 -= rate*dw1;
        w2 -= rate*dw2;

        // b -= rate*db;

        printf("cost = %f, w1 = %f, w2 = %f\n", cost_two_inputs(w1, w2), w1, w2);

    }

    dline();
    printf("w1 = %f, w2 = %f\n", w1, w2);

}
