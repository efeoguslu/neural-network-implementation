#include "activation.h"


// -------------------------------------------------------------------------

double identity(double x){
    return x;
}

int d_identity(){
    return 1;
}

// -----------------------------------------------

int binary_step(double x){
    return x < 0 ? 0 : 1;
}

int d_binary_step(double x){
    return x != 0 ? 0 : 0; // not sure about this.
}

// -----------------------------------------------

/*
double sigmoid(double n){
    return (1 / (1 + pow(EULER_NUMBER, -n)));
}
*/

double sigmoid(double n){
    return (1.0 / (1.0 + exp(-n)));
}

double d_sigmoid(double n){
    return (sigmoid(n) * (1 - sigmoid(n)));
}

// -----------------------------------------------

double relu(double n){
    return n < 0 ? 0 : n;
}

ActivationFunction g_activation_func = sigmoid; // default to sigmoid