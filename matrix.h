#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <stdio.h>

#ifndef MATRIX_CALLOC
#include <stdlib.h>
#define MATRIX_CALLOC calloc 
#endif // MATRIX_CALLOC

#ifndef MATRIX_ASSERT
#include <assert.h>
#define MATRIX_ASSERT assert
#endif // MATRIX_ASSERT

// defining the shape of the matrix:

typedef struct{
    size_t rows;
    size_t cols;
    double *es; // pointer to the beginning of the matrix
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).cols + (j)]

Mat mat_alloc(size_t rows, size_t cols);
void mat_dot(Mat dst, Mat a, Mat b); // destination is the first element --> consistent with std. C functions 
void mat_sum(Mat dst, Mat a);
void mat_print(Mat m);
void mat_rand(Mat m, double low, double high);

#endif // MATRIX_H

