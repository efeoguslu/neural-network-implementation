#include "activation.h"
#include "helpers.h"
#include "xor.h"
#include "matrix.h"

int main(){
    
    randomize();
    
    // modeling XOR with matrix framework

    // first layer: 

    Mat w1 = mat_alloc(2, 2);
    Mat b1 = mat_alloc(1, 2);

    // second layer:

    Mat w2 = mat_alloc(2, 1);
    Mat b2 = mat_alloc(1, 1);

    // randomize all layers:
    
    mat_rand(w1, 0.0, 1.0);
    mat_rand(b1, 0.0, 1.0);
    mat_rand(w2, 0.0, 1.0);
    mat_rand(b2, 0.0, 1.0);

    mat_print(w1, "w1");
    dline();
    mat_print(b1, "b1");
    dline();
    mat_print(w2, "w2");
    dline();
    mat_print(b2, "b2");
   
    

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