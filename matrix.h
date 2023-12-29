#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef MATRIX_CALLOC
#include <stdlib.h>
#define MATRIX_CALLOC calloc 
#endif // MATRIX_CALLOC

#ifndef MATRIX_MALLOC
#include <stdlib.h>
#define MATRIX_MALLOC malloc 
#endif // MATRIX_MALLOC

#ifndef MATRIX_ASSERT
#include <assert.h>
#define MATRIX_ASSERT assert
#endif // MATRIX_ASSERT

// defining the shape of the matrix:
typedef struct{
    size_t rows;
    size_t cols;
    size_t stride;
    double *es; // pointer to the beginning of the matrix
} Mat;

typedef struct{
    size_t count; // amount of inner layers 
    Mat *ws; // weights 
    Mat *bs; // biases
    Mat *as; // inputs: one element bigger than ws and bs: count + 1 
} Network;

#define NN_INPUT(nn)    (nn).as[0]
#define NN_OUTPUT(nn)   (nn).as[(nn).count]

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])
#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

Mat mat_alloc(size_t rows, size_t cols);
void mat_dot(Mat dst, Mat a, Mat b); // destination is the first element --> consistent with std. C functions 
void mat_sum(Mat dst, Mat a);
void mat_print(Mat m, const char *name, size_t padding);
void mat_rand(Mat m, double low, double high);
void mat_sig(Mat m);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_fill(Mat m, double x);
void mat_save(FILE *file, Mat m);
void mat_free(Mat m);
// ---------------------------------------------------------------------------------

Network nn_alloc(size_t *arch, size_t arch_count);
void nn_print(Network nn, const char *name);
void nn_rand(Network nn, double low, double high);
void nn_forward(Network nn);
double nn_cost(Network nn, Mat ti, Mat to);
void nn_finite_diff(Network m, Network g, double eps, Mat ti, Mat to);
void nn_learn(Network nn, Network g, double rate);

// ---------------------------------------------------------------------------------

void nn_backprop(Network nn, Network g, Mat ti, Mat to);
void nn_zero(Network nn);



#define NN_PRINT(nn)    nn_print(nn, #nn)
#define MAT_PRINT(mat)  mat_print(mat, #mat, 0)

#endif // MATRIX_H

