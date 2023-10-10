#include "activation.h"

// Move them elsewhere: 

void randomize(void){
    srand((unsigned int)time(0));
}

void dline(void){
    printf("-------------------------------------------\n");
}

// -------------------------------------------------------------------------

double identity(double x){
    return x;
}

int d_identity(double x){
    return 1;
}

int binary_step(double x){
    return x < 0 ? 0 : 1;
}

int d_binary_step(double x){
    return x != 0 ? 0 : 0; // not sure about this.
}

void menu(void){ // inputs should not be void 

    printf("\nWelcome to the Neural Network Implementation Project in C!\n");
    dline();
    printf("Please Choose an Activation Function: \n");
    printf("1. Identity\n2. Binary Step\n");

    // scan the input

    printf("\nPlease enter the number of layers: ");

    // scan the number of layers

    printf("\nPlease enter the number of neurons in each layer: ");

    // scan the number of neurons in each layer, with a for loop, according to the # of layers

    printf("\n");


}
