#include "activation.h"
#include "helpers.h"

// ********************************************************************************

// ---- TRAINING SETS ---- 

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

// NAND-gate

double nand_train[][3] = {
    {0, 0, 1},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},

};

// the simple training set:

double train[][2] = {
    {0, 0},
    {1, 3},
    {2, 6},
    {3, 9},
    {4, 12},
};

// ********************************************************************************

// ---- TRAINING SET COUNTS ----

//#define train_count (sizeof(train)/sizeof(train[0]))

#define or_train_count (sizeof(or_train) / sizeof(or_train[0]))
#define and_train_count (sizeof(and_train)/sizeof(and_train[0]))
#define train_count(x) (sizeof(x)/sizeof(x[0]))

// ********************************************************************************

double rand_double(void) 
{
    return (double)rand() / (double)RAND_MAX;
}


// ********************************************************************************

/*
double cost(double w, double b){

    double result = 0.0;

    for(size_t i = 0; i < train_count; ++i){
        double x = train[i][0];
        double y = x*w + b; // single artificial neuron mathematical model
        double d = y - train[i][1];
        result += d*d;
    }

    result /= train_count;

    return result;
}
*/
// ------------------------------

double cost_two_inputs(double w1, double w2, double bias){

    double result = 0.0;

    for(size_t i = 0; i < train_count(or_train); ++i){
        double x1 = or_train[i][0];
        double x2 = or_train[i][1];
        double y = sigmoid(x1*w1 + x2*w2 + bias); // single artificial neuron mathematical model (with two inputs and a bias)
        double d = y - or_train[i][2];
        result += d*d;

    }

    result /= train_count(or_train);

    return result;
}

// ********************************************************************************

int main(){

    // menu();  
    randomize();
    
    double w1 = rand_double();
    double w2 = rand_double();
    double b  = rand_double();

    double eps = 1e-1;
    double rate = 1e-1;

    size_t epoch = 10000;

    for(size_t i = 0; i < epoch; ++i){
        
        double c = cost_two_inputs(w1, w2, b);

        // finite difference: approximation of derivative 
        double dw1 = ((cost_two_inputs(w1 + eps, w2, b) - c)/eps);
        double dw2 = ((cost_two_inputs(w1, w2 + eps, b) - c)/eps);
        double db  = ((cost_two_inputs(w1, w2, b + eps) - c)/eps);

        w1 -= rate*dw1;
        w2 -= rate*dw2;
        b  -= rate*db;

        printf("cost = %f, w1 = %f, w2 = %f, bias = %f\n", cost_two_inputs(w1, w2, b), w1, w2, b);

    }

    dline();
    printf("cost = %f, w1 = %f, w2 = %f, bias = %f\n", cost_two_inputs(w1, w2, b), w1, w2, b);

    test_model(w1, w2, b);
}
