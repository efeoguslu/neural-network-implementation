#ifndef CSV_H
#define CSV_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

#define MAX_LINE_LENGTH 1024
#define MAX_COLUMNS 14

// Structure to store a single row of data
typedef struct {
    double *values;
} DataRow;

// Structure to store the dataset
typedef struct {
    size_t num_rows;
    size_t num_columns;
    DataRow *rows;
} Dataset;


void free_dataset(Dataset *dataset);
Dataset *load_csv(const char *file_path);
void split_dataset(const Dataset *dataset, size_t label_column, Mat *features, Mat *labels);
void preprocess_data(Mat *features);


#endif // CSV_H