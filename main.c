#include "activation.h"
#include "helpers.h"
#include "xor.h"
#include "matrix.h"

typedef struct{

    Mat a0;
    Mat w1, b1, a1;
    Mat w2, b2, a2;

} Exor;

void forward_exor(Exor m){

    // single path:
    mat_dot(m.a1, m.a0, m.w1);
    mat_sum(m.a1, m.b1);
    mat_sig(m.a1);

    mat_dot(m.a2, m.a1, m.w2);
    mat_sum(m.a2, m.b2);
    mat_sig(m.a2);

}

double cost_exor(Exor m, Mat ti, Mat to){

    assert(ti.rows == to.rows);
    assert(to.cols == m.a2.cols);

    size_t n = ti.rows;

    double c = 0.0;

    for(size_t i = 0; i < n; ++i){

        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(m.a0, x);
        forward_exor(m);

        size_t q = to.cols;
        
        for(size_t j = 0; j < q; ++j){
            double d = MAT_AT(m.a2, 0, j) - MAT_AT(y, 0, j);
            c += d*d;
        }
    }

    return c/n;
}   



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

    mat_print(ti, "ti");
    mat_print(to, "to");

    Exor m;
    
    // modeling XOR with matrix framework

    // input, represented as a matrix:

    m.a0 = mat_alloc(1,2);

    // first layer: 

    m.w1 = mat_alloc(2, 2);
    m.b1 = mat_alloc(1, 2);

    // output of first layer:

    m.a1 = mat_alloc(1, 2);

    // second layer:

    m.w2 = mat_alloc(2, 1);
    m.b2 = mat_alloc(1, 1);

    m.a2 = mat_alloc(1, 1);

    // randomize all layers:
    
    mat_rand(m.w1, 0.0, 1.0);
    mat_rand(m.b1, 0.0, 1.0);
    mat_rand(m.w2, 0.0, 1.0);
    mat_rand(m.b2, 0.0, 1.0);

    printf("cost = %lf\n", cost_exor(m, ti, to));

    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){

            MAT_AT(m.a0, 0, 0) = i;
            MAT_AT(m.a0, 0, 1) = j;
            forward_exor(m);
            double y = *m.a2.es;

            printf("%zu ^ %zu = %lf\n", i, j, y);
        }
    }

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