#include "activation.h"
#include "helpers.h"
#include "cost.h"
#include "xor.h"

// ********************************************************************************
// 2.01.00 ------

int main(){

    randomize();

    FILE *file = fopen("cost.txt", "w");

    if(file == NULL){
        printf("error opening file!\n");
        exit(1);
    }

    Xor m = rand_xor();
    printf("original and initial random model:\n");
    print_xor(m);
    dline();

    //train_model(m);

    double eps = 1e-1;
    double rate = 1e-1;
    
    for(size_t i = 0; i < 50*1000; ++i){

        Xor g = finite_difference(m, eps);
        m = subtract_gradient(m, g, rate);
        double output = xor_cost(m);
        printf("cost = %lf\n", output);
        fprintf(file, "%lf\n", output);
    }

    fclose(file);

    dline();    
    test_xor_model(m);

    return 0;

}








    // --------------------------------------------------------------------------------------------
    
    /*
    // menu();  
    randomize();
    
    double w1 = rand_double();
    double w2 = rand_double();
    double b  = rand_double();

    //double eps = 1e-1;
    //double rate = 1e-1;

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

*/
