#include "cost.h"
#include "helpers.h"
#include "activation.h"
#include "xor.h"

// ---- TRAINING SETS ---- 

// OR-gate

typedef double sample[3];

sample or_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

// AND-gate

sample and_train[] = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 1},

};

// NAND-gate

sample nand_train[] = {
    {0, 0, 1},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};

// XOR-gate --> not modelable by a single neuron: 

// ( x | y ) & ~( x & y )


sample xor_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};

sample *train = xor_train;



// ********************************************************************************

// ---- TRAINING SET COUNTS ----

//#define train_count (sizeof(train)/sizeof(train[0]))

#define or_train_count (sizeof(or_train) / sizeof(or_train[0]))
#define and_train_count (sizeof(and_train)/sizeof(and_train[0]))
// #define train_count(x) (sizeof(x)/sizeof(x[0]))

size_t train_count = 4;

double cost_two_inputs(double w1, double w2, double bias){

    double result = 0.0;

    for(size_t i = 0; i < train_count; ++i){
        double x1 = train[i][0];
        double x2 = train[i][1];
        double y = sigmoid(x1*w1 + x2*w2 + bias); // single artificial neuron mathematical model (with two inputs and a bias)
        double d = y - train[i][2];
        result += d*d;

    }

    result /= train_count;

    return result;
}

// ----------------------------------------------------------------------------------------------------

double forward(Xor m, double x1, double x2){

    // outputs of the first layer:
    double a = sigmoid(m.or_w1*x1 + m.or_w2*x2 + m.or_b);
    double b = sigmoid(m.nand_w1*x1 + m.nand_w2*x2 + m.nand_b);
    
    // returning the output of the last layer:
    return sigmoid(a*m.and_w1 + b*m.and_w2 + m.and_b);
}

// ------------------------------------------------------------------

size_t xor_train_count = 4;

// ------------------------------------------------------------------

double xor_cost(Xor m){

    double result = 0.0;

    for(size_t i = 0; i < xor_train_count; ++i){
        double x1 = train[i][0];
        double x2 = train[i][1];
        double y = forward(m, x1, x2);
        double d = y - train[i][2];
        result += d*d;
    }
    result /= xor_train_count;
    return result;
}

// ------------------------------------------------------------------

Xor rand_xor(){

    Xor m;
    m.or_w1 = rand_double();
    m.or_w2 = rand_double();
    m.or_b = rand_double();

    m.nand_w1 = rand_double();
    m.nand_w2 = rand_double();
    m.nand_b = rand_double();

    m.and_w1 = rand_double();
    m.and_w2 = rand_double();
    m.and_b = rand_double();

    return m;
}

// ------------------------------------------------------------------

void print_xor(Xor m){

    printf("or_w1 = %lf\n", m.or_w1);
    printf("or_w2 = %lf\n", m.or_w2);
    printf("or_b  = %lf\n", m.or_b);

    printf("nand_w1 = %lf\n", m.nand_w1);
    printf("nand_w2 = %lf\n", m.nand_w2);
    printf("nand_b  = %lf\n", m.nand_b);

    printf("and_w1 = %lf\n", m.and_w1);
    printf("and_w2 = %lf\n", m.and_w2);
    printf("and_b  = %lf\n", m.and_b);
}

// ------------------------------------------------------------------
