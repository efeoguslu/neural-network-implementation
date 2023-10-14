#include "helpers.h"

void randomize(void){
    srand((unsigned int)time(0));
}

void dline(void){
    printf("-------------------------------------------\n");
}

void menu(void){ // inputs should not be void 

    printf("\nWelcome to the Neural Network Implementation Project in C!\n");
    dline();
    printf("Please Choose an Activation Function: \n");
    printf("1. Identity\n2. Binary Step\n3. Sigmoid\n");

    // scan the input

    printf("\nPlease enter the number of layers: ");

    // scan the number of layers

    printf("\nPlease enter the number of neurons in each layer: ");

    // scan the number of neurons in each layer, with a for loop, according to the # of layers

    printf("\n");

}