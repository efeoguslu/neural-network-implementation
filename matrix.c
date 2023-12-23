#include "matrix.h"
#include "helpers.h"
#include "activation.h"
#include "xor.h"

// 2 inputs, 2 hidden layers with 4 and 2 neurons respectively, and one output layer with 1 neuron
// size_t arch[] = {2, 2, 1};
// Network nn = nn_alloc(arch, ARRAY_LEN(layers));

Network nn_alloc(size_t *arch, size_t arch_count){

    MATRIX_ASSERT(arch_count > 0);

    Network nn;
    nn.count = arch_count - 1; // subtracting the input layer

    // preallocating arrays of matrices

    nn.ws = MATRIX_MALLOC(sizeof(*nn.ws)*nn.count);
    MATRIX_ASSERT(nn.ws != NULL);
    nn.bs = MATRIX_MALLOC(sizeof(*nn.bs)*nn.count);
    MATRIX_ASSERT(nn.bs != NULL);
    nn.as = MATRIX_MALLOC(sizeof(*nn.as)*(nn.count + 1));
    MATRIX_ASSERT(nn.as != NULL);

    // input vector with a single row:
    nn.as[0] = mat_alloc(1, arch[0]);

    for(size_t i = 1; i < arch_count; ++i){
        nn.ws[i - 1] = mat_alloc(nn.as[i - 1].cols, arch[i]);
        nn.bs[i - 1] = mat_alloc(1, arch[i]);
        nn.as[i]     = mat_alloc(1, arch[i]);
    }

    return nn;
}

void nn_print(Network nn, const char* name){
    
    char buf[256];
    printf("%s = [\n", name);

    for(size_t i = 0; i < nn.count; ++i){
        snprintf(buf, sizeof(buf), "ws%u", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%u", i);
        mat_print(nn.bs[i], buf, 4);
    }

    printf("]\n");
}

void nn_rand(Network nn, double low, double high){
    for(size_t i = 0; i < nn.count; ++i){
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(Network nn){
    for(size_t i = 0; i < nn.count; ++i){
        mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i + 1], nn.bs[i]);
        mat_sig(nn.as[i + 1]);
    }
}

double nn_cost(Network nn, Mat ti, Mat to){
    assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(nn).cols);

    size_t n = ti.rows;

    double c = 0;

    for(size_t i = 0; i < n; ++i){
        Mat x = mat_row(ti, i); // expected input
        Mat y = mat_row(to, i); // expected output

        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);
        size_t q = to.cols;

        for(size_t j = 0; j < q; ++j){
            double d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            c += d*d;
        }
    }

    return c/n;
}


void nn_finite_diff(Network nn, Network g, double eps, Mat ti, Mat to){

    double saved; 
    double c = nn_cost(nn, ti, to);

    for(size_t i = 0; i < nn.count; ++i){

        for(size_t j = 0; j < nn.ws[i].rows; ++j){
            for(size_t k = 0; k < nn.ws[i].cols; ++k){
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += eps;
                MAT_AT(g.ws[i], j, k) = (nn_cost(nn, ti, to) - c)/eps;
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }

        for(size_t j = 0; j < nn.bs[i].rows; ++j){
            for(size_t k = 0; k < nn.bs[i].cols; ++k){
                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) += eps;
                MAT_AT(g.bs[i], j, k) = (nn_cost(nn, ti, to) - c)/eps;
                MAT_AT(nn.bs[i], j, k) = saved;
            }
        }

    }
}

void nn_learn(Network nn, Network g, double rate){

    for(size_t i = 0; i < nn.count; ++i){

        for(size_t j = 0; j < nn.ws[i].rows; ++j){
            for(size_t k = 0; k < nn.ws[i].cols; ++k){
                MAT_AT(nn.ws[i], j, k) -= rate*MAT_AT(g.ws[i], j, k);
            }
        }

        for(size_t j = 0; j < nn.bs[i].rows; ++j){
            for(size_t k = 0; k < nn.bs[i].cols; ++k){
                MAT_AT(nn.bs[i], j, k) -= rate*MAT_AT(g.bs[i], j, k);

            }
        }

    }
}


Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = MATRIX_CALLOC(rows*cols, sizeof(*m.es)); 
    MATRIX_ASSERT(m.es != NULL);
    return m;
}

void mat_dot(Mat dst, Mat a, Mat b){
    // checking if matrices are eligible for dot product:
    MATRIX_ASSERT(a.cols == b.rows);
    size_t n = a.cols;
    MATRIX_ASSERT(dst.rows == a.rows);
    MATRIX_ASSERT(dst.cols == b.cols);
    // dot product operation:
    for(size_t i = 0; i < dst.rows; ++i){
        for(size_t j = 0; j < dst.cols; ++j){
            MAT_AT(dst, i, j) = 0;
            for(size_t k = 0; k < n; ++k){
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

Mat mat_row(Mat m, size_t row){
    return (Mat){ 
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0),
    };
}

void mat_copy(Mat dst, Mat src){
    MATRIX_ASSERT(dst.rows == src.rows);
    MATRIX_ASSERT(dst.cols == src.cols);

    for(size_t i = 0; i < dst.rows; ++i){
        for(size_t j = 0; j < dst.cols; ++j){
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_sum(Mat dst, Mat a){
    // checking if matrices are eligible to summation:
    MATRIX_ASSERT(dst.rows == a.rows);
    MATRIX_ASSERT(dst.cols == a.cols);
    // the summation:
    for(size_t i = 0; i < dst.rows; ++i){
        for(size_t j = 0; j < dst.cols; ++j){
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_print(Mat m, const char* name, size_t padding){

    // printf("\nMatrix with %u rows and %u columns:\n", m.rows, m.cols);
    
    printf("%*s%s = [\n", (int)padding, "", name);
    for(size_t i = 0; i < m.rows; ++i){
        printf("%*s     ", (int)padding, "");
        for(size_t j = 0; j < m.cols; ++j){
            printf("%lf ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
    
}

void mat_rand(Mat m, double low, double high){
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j < m.cols; ++j){
            MAT_AT(m, i, j) = rand_double()*(high - low) + low;
        }
        
    }
}

void mat_sig(Mat m){
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j < m.cols; ++j){
            MAT_AT(m, i, j) = g_activation_func(MAT_AT(m, i, j));
        }
    }
}


