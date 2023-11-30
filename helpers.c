#include "helpers.h"
#include "activation.h"
#include "xor.h"

void randomize(void){
    srand((unsigned int)time(0));
}

void dline(void){
    printf("----------------------------------------------------------------\n");
}

void menu(int* num_layers, int** neurons_per_layer, ActivationFunction* activation_func) {
    printf("\nWelcome to the Neural Network Implementation Project in C!\n");
    printf("\nPlease enter the number of layers: ");
    scanf("%d", num_layers);
    *neurons_per_layer = (int*)malloc(sizeof(int) * (*num_layers));
    printf("\nPlease enter the number of neurons in each layer: ");
    for (int i = 0; i < *num_layers; i++) {
        printf("\nEnter one number of neurons for layer %d: ", i+1);
        scanf("%d", &(*neurons_per_layer)[i]);
    }

    printf("\nPlease enter the activation function (1 for sigmoid, 2 for relu): ");
    int activationChoice;
    scanf("%d", &activationChoice);

    // Set activation function based on user choice
    switch (activationChoice) {
        case 1:
            *activation_func = sigmoid;
            break;
        case 2:
            *activation_func = relu;
            break;
        // Add more cases for additional activation functions if needed
        default:
            *activation_func = sigmoid; // Default to sigmoid
            break;
    }


    FILE *params_file = fopen("params.txt", "w");
    if (params_file == NULL) {
        printf("Error opening file!\n");
        return;
    }

    fprintf(params_file, "%d\n", *num_layers);

    for (int i = 0; i < *num_layers; i++) {
        fprintf(params_file, "%d\n", (*neurons_per_layer)[i]);
    }

    fclose(params_file);
}

// -----------------------------------------------------------------------------------------------

void test_model(double w1, double w2, double bias){
    printf("\nTest:\n");
    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){
            printf("%zu | %zu = %f\n", i, j, sigmoid(i*w1 + j*w2 + bias));
        }
    }
}


double rand_double(void) 
{
    return (double)rand() / (double)RAND_MAX;
}

