#include "activation.h"
#include "helpers.h"
#include "xor.h"
#include "matrix.h"
#include "mnist.h"
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

double td_xor[] = { 
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

double td_or[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1,
};

int main(){

    randomize();

    double *td = td_xor;

    size_t stride = 3;
    size_t n = 4;

    Mat ti = {
        .rows = n,
        .cols = 2, 
        .stride = stride, 
        .es = td,
    };

    Mat to = {
        .rows = n, 
        .cols = 1,
        .stride = stride,
        .es = td + 2,
    };

    size_t arch[] = {2, 2, 1};
    
    Network nn = nn_alloc(arch, ARRAY_LEN(arch));
    Network g  = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);

    NN_PRINT(nn);
    

    double rate = 1;
    size_t epoch = 10000;

    const char *output_cost_file_name = "cost.txt"; 

    FILE *cost_file = fopen(output_cost_file_name, "w"); 

    if (cost_file == NULL) {
        fprintf(stderr, "ERROR: could not open file %s for writing\n", output_cost_file_name);
        return 1;
    }

    // Training
    for(size_t i = 0; i < epoch; ++i){
        printf("cost: %lf\n", nn_cost(nn, ti, to));
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
        fprintf(cost_file, "%lf\n", nn_cost(nn, ti, to));

    }   

    // Testing
    printf("Testing the XOR gate:\n");
    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = j;
            nn_forward(nn);
            printf("%zu ^ %zu = %lf\n", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
        }
    }

    fclose(cost_file);
}