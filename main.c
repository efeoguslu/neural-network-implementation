#define MATRIX_IMPLEMENTATION
#include "activation.h"
#include "helpers.h"
#include "xor.h"
#include "matrix.h"

// ml_2 --> 53.00

int main(){

    Mat matrix = mat_alloc(2, 2);
    mat_print(matrix);


    return 0;
}

/*
    randomize();

    int num_layers;
    int* neurons_per_layer;
    menu(&num_layers, &neurons_per_layer);

    FILE *cost_file = fopen("cost.txt", "w");

    if(cost_file == NULL){
        printf("error opening file!\n");
        exit(1);
    }

    Xor m = rand_xor();
    printf("original and initial random model:\n");
    print_xor(m);
    dline();

    double eps = 1e-1;
    double rate = 1e-1;
    
    for(size_t i = 0; i < 50*1000; ++i){
        Xor g = finite_difference(m, eps);
        m = subtract_gradient(m, g, rate);
        double output = xor_cost(m);
        printf("cost = %lf\n", output);
        fprintf(cost_file, "%lf\n", output);
    }

    fclose(cost_file);

    dline();    
    test_xor_model(m);

    // TESTING:

    save_weights("weights.txt", m);

    return 0;

}
*/