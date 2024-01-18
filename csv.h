#ifndef CSV_H
#define CSV_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include <float.h>

// BOSTON DATA MACROS
#define NUM_RECORDS_TRAIN 406
#define NUM_RECORDS_TEST 100

#define NUM_FEATURES 13
#define TOTAL_COLUMNS (NUM_FEATURES + 1) // 13 features + 1 target

#define NUM_FEATURES_FILTERED 6
#define TOTAL_COLUMNS_FILTERED (NUM_FEATURES_FILTERED + 1)

// FEATURE AVERAGE VALUES (PLACEHOLDERS)
#define AVG_RM 6.24
#define AVG_LSTAT 12.65
#define AVG_PTRATIO 18.45
#define AVG_CRIM 3.61
#define AVG_NOX 0.55
#define AVG_DIS 3.79

double *read_csv(const char *filename, int *num_records);
void printData(double *data, int num_records);
void printFilteredData(double *matrix, int num_records);
void printUserInput(double *userInput);

void filterDataToMatrix(double *data, int num_records, double *matrix);
void normalizeFeatures(double *data, int num_records, int num_features);

void getUserInput(double *userInput);
void normalizeUserInput(double *userInput, double *mins, double *maxes);
void findMinMax(double *data, int num_records, double *mins, double *maxes);
double revertNormalizedValue(double normalizedValue, double min, double max);

void trainNetwork(Network *nn, double *filteredData, int num_records, int epochs, double learning_rate, size_t *arch, size_t arch_len);
double predict(Network *nn, double *inputFeatures);

void nn_free(Network *nn);


#endif // CSV_H