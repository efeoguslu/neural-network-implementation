#include "activation.h"
#include "helpers.h"
#include "xor.h"
#include "matrix.h"

// training data:
double td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

int main(){

    randomize();

    size_t stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/stride;

    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2,
    }; 

    size_t arch[] = {2, 2, 1};
    
    // memory management:

    Network our_neural_network = nn_alloc(arch, ARRAY_LEN(arch));
    Network gradient = nn_alloc(arch, ARRAY_LEN(arch));

    double eps = 1e-1;
    double rate = 1e-1;
    size_t epoch = 50*1000;

    FILE *fp;  // File pointer
    fp = fopen("cost.txt", "w");  // Open file in write mode

    if (fp == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    nn_rand(our_neural_network, 0, 1);
    printf("cost = %lf\n", nn_cost(our_neural_network, ti, to));

    for(size_t i = 0; i < epoch; ++i){
        nn_finite_diff(our_neural_network, gradient, eps, ti, to);
        nn_learn(our_neural_network, gradient, rate);
        printf("cost = %lf\n", nn_cost(our_neural_network, ti, to));
        fprintf(fp, "%lf\n", nn_cost(our_neural_network, ti, to));
    }

    fclose(fp);

    printf("\nTest:\n");
    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){
            MAT_AT(NN_INPUT(our_neural_network), 0, 0) = i;
            MAT_AT(NN_INPUT(our_neural_network), 0, 1) = j;
            nn_forward(our_neural_network);
            printf("%u ^ %u = %lf\n", i, j, MAT_AT(NN_OUTPUT(our_neural_network), 0, 0));
        }
    }

    return 0;
}
