#include "activation.h"
#include "helpers.h"
#include "xor.h"
#include "matrix.h"

//2 --> 2.18.00
typedef struct{
    
    Mat a0, a1, a2;
    Mat w1, b1;
    Mat w2, b2;

} Exor;

Exor xor_alloc(void) {
    Exor m;
    
    // Allocate memory for the input matrix
    m.a0 = mat_alloc(1, 2);
    
    // Allocate memory for the first layer weights and biases
    m.w1 = mat_alloc(2, 2);
    m.b1 = mat_alloc(1, 2);
    
    // Allocate memory for the output of the first layer
    m.a1 = mat_alloc(1, 2);
    
    // Allocate memory for the second layer weights and biases
    m.w2 = mat_alloc(2, 1);
    m.b2 = mat_alloc(1, 1);
    
    // Allocate memory for the output of the second layer
    m.a2 = mat_alloc(1, 1);
    
    return m;
}

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

void finite_diff(Exor m, Exor g, double eps, Mat ti, Mat to){
    double saved;
    double c = cost_exor(m, ti, to);

    for(size_t i = 0; i < m.w1.rows; ++i){
        for(size_t j = 0; j < m.w1.cols; ++j){
            saved = MAT_AT(m.w1, i, j);
            MAT_AT(m.w1, i, j) += eps;
            MAT_AT(g.w1, i, j) = (cost_exor(m, ti, to) - c)/eps;

            MAT_AT(m.w1, i, j) = saved;
        }
    }

    for(size_t i = 0; i < m.b1.rows; ++i){
        for(size_t j = 0; j < m.b1.cols; ++j){
            saved = MAT_AT(m.b1, i, j);
            MAT_AT(m.b1, i, j) += eps;
            MAT_AT(g.b1, i, j) = (cost_exor(m, ti, to) - c)/eps;

            MAT_AT(m.b1, i, j) = saved;
        }
    }

    for(size_t i = 0; i < m.w2.rows; ++i){
        for(size_t j = 0; j < m.w2.cols; ++j){
            saved = MAT_AT(m.w2, i, j);
            MAT_AT(m.w2, i, j) += eps;
            MAT_AT(g.w2, i, j) = (cost_exor(m, ti, to) - c)/eps;

            MAT_AT(m.w2, i, j) = saved;
        }
    }

    for(size_t i = 0; i < m.b2.rows; ++i){
        for(size_t j = 0; j < m.b2.cols; ++j){
            saved = MAT_AT(m.b2, i, j);
            MAT_AT(m.b2, i, j) += eps;
            MAT_AT(g.b2, i, j) = (cost_exor(m, ti, to) - c)/eps;

            MAT_AT(m.b2, i, j) = saved;
        }
    }
}

void xor_learn(Exor m, Exor g, double rate){

    for(size_t i = 0; i < m.w1.rows; ++i){
        for(size_t j = 0; j < m.w1.cols; ++j){
            MAT_AT(m.w1, i, j) -= rate*MAT_AT(g.w1, i, j);
        }
    }
    
    for(size_t i = 0; i < m.b1.rows; ++i){
        for(size_t j = 0; j < m.b1.cols; ++j){
            MAT_AT(m.b1, i, j) -= rate*MAT_AT(g.b1, i, j);
        }
    }

    for(size_t i = 0; i < m.w2.rows; ++i){
        for(size_t j = 0; j < m.w2.cols; ++j){
            MAT_AT(m.w2, i, j) -= rate*MAT_AT(g.w2, i, j);
        }
    }

    for(size_t i = 0; i < m.b2.rows; ++i){
        for(size_t j = 0; j < m.b2.cols; ++j){
            MAT_AT(m.b2, i, j) -= rate*MAT_AT(g.b2, i, j);
        }
    }
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

    // mat_print(ti, "ti");
    // mat_print(to, "to");

    Exor m = xor_alloc();
    Exor g = xor_alloc();

    // randomize all layers:
    
    mat_rand(m.w1, 0.0, 1.0);
    mat_rand(m.b1, 0.0, 1.0);
    mat_rand(m.w2, 0.0, 1.0);
    mat_rand(m.b2, 0.0, 1.0);

    double eps = 1e-1;
    double rate = 1e-1; 

    FILE *fp;  // File pointer
    fp = fopen("cost.txt", "w");  // Open file in write mode

    if (fp == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    printf("cost = %lf\n", cost_exor(m, ti, to));

    for(size_t i = 0; i < 50*1000; ++i){
        finite_diff(m, g, eps, ti, to);
        xor_learn(m, g, rate);
        printf("cost = %lf\n", cost_exor(m, ti, to));
        fprintf(fp, "%lf\n", cost_exor(m, ti, to));
    }

    dline();

    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){

            MAT_AT(m.a0, 0, 0) = i;
            MAT_AT(m.a0, 0, 1) = j;
            forward_exor(m);
            double y = *m.a2.es;

            printf("%u ^ %u = %lf\n", i, j, y);
        }
    }

    return 0;
}

