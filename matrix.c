#include "matrix.h"
#include "helpers.h"
#include "activation.h"
#include "xor.h"


Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
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

void mat_print(Mat m){

    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j < m.cols; ++j){
            printf("%lf ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    
}

void mat_rand(Mat m, double low, double high){

    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j < m.cols; ++j){
            MAT_AT(m, i, j) = rand_double()*(high - low) + low;
        }
        
    }
}