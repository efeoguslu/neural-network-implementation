#include "matrix.h"
#include "activation.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// BOSTON DATA MACROS
#define NUM_RECORDS 506

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
void normalizeFeatures(double *data, int num_records, int num_features, double *minPrice, double *maxPrice);
void getUserInput(double *userInput);
void normalizeUserInput(double *userInput, double *mins, double *maxes);
void findMinMax(double *data, int num_records, double *mins, double *maxes);
double revertNormalizedValue(double normalizedValue, double min, double max);

void trainNetwork(Network *nn, double *filteredData, int num_records, int epochs, double learning_rate, size_t *arch, size_t arch_len);
double predict(Network *nn, double *inputFeatures);

void nn_free(Network *nn);

int main()
{
    int num_records;
    double *data = read_csv("housing.csv", &num_records);
    double *filteredData = malloc(num_records * TOTAL_COLUMNS_FILTERED * sizeof(double));

    filterDataToMatrix(data, num_records, filteredData);
    //normalizeFeatures(filteredData, num_records_train, NUM_FEATURES_FILTERED, &minPrice, &maxPrice);

    

    // Calculate the number of records for training and testing
    int num_records_train = (int)(0.8 * num_records);
    int num_records_test = num_records - num_records_train;

    // Allocate memory for training and testing data
    double *data_train = malloc(num_records_train * TOTAL_COLUMNS_FILTERED * sizeof(double));
    double *data_test = malloc(num_records_test * TOTAL_COLUMNS_FILTERED * sizeof(double));

    // Copy normalized data to training and testing arrays
    memcpy(data_train, filteredData, num_records_train * TOTAL_COLUMNS_FILTERED * sizeof(double));
    memcpy(data_test, filteredData + num_records_train * TOTAL_COLUMNS_FILTERED, num_records_test * TOTAL_COLUMNS_FILTERED * sizeof(double));

    double minPrice, maxPrice;

    // Calculate min and max values during normalization
    // normalizeFeatures(filteredData, num_records_train, NUM_FEATURES_FILTERED, &minPrice, &maxPrice);
    normalizeFeatures(filteredData, num_records_train, NUM_FEATURES_FILTERED, &minPrice, &maxPrice);


    Mat t_train = mat_alloc(num_records_train, TOTAL_COLUMNS_FILTERED);

    // Copy training data to a matrix
    for (int i = 0; i < num_records_train; ++i)
    {
        for (int j = 0; j < TOTAL_COLUMNS_FILTERED; ++j)
        {
            size_t pos = i * TOTAL_COLUMNS_FILTERED + j;
            MAT_AT(t_train, i, j) = data_train[pos];
        }
    }

    MAT_PRINT(t_train);

    Mat ti_train = {
        .rows = t_train.rows,
        .cols = 6,
        .stride = t_train.stride,
        .es = &MAT_AT(t_train, 0, 0),
    };

    Mat to_train = {
        .rows = t_train.rows,
        .cols = 1,
        .stride = t_train.stride,
        .es = &MAT_AT(t_train, 0, ti_train.cols),
    };

    size_t arch[] = {6, 8, 5, 1}; // 6 8 5 1 (example architecture)

    Network nn = nn_alloc(arch, ARRAY_LEN(arch));
    Network g = nn_alloc(arch, ARRAY_LEN(arch));

    nn_rand(nn, 0, 1);
    double rate = 3.8;
    size_t epoch = 50000;

    printf("Training...\n");
    for (size_t i = 0; i < epoch; ++i)
    {
        nn_backprop(nn, g, ti_train, to_train);
        nn_learn(nn, g, rate);

        printf("Epoch: %zu\t Cost: %lf\t Learning Rate: %.3lf\n", i, nn_cost(nn, ti_train, to_train), rate);
    }

    // Now test the network using data_test

    Mat t_test = mat_alloc(num_records_test, TOTAL_COLUMNS_FILTERED);

    // Copy testing data to a matrix
    for (int i = 0; i < num_records_test; ++i)
    {
        for (int j = 0; j < TOTAL_COLUMNS_FILTERED; ++j)
        {
            size_t pos = i * TOTAL_COLUMNS_FILTERED + j;
            MAT_AT(t_test, i, j) = data_test[pos];
        }
    }

    MAT_PRINT(t_test);

    Mat ti_test = {
        .rows = t_test.rows,
        .cols = 6,
        .stride = t_test.stride,
        .es = &MAT_AT(t_test, 0, 0),
    };

    // Perform forward pass on the testing data
    nn_forward(nn);

    // Display the predicted prices
    for (int i = 0; i < num_records_test; ++i)
    {
        double predictedPrice = MAT_AT(NN_OUTPUT(nn), i, 0);
        double originalPredictedPrice = revertNormalizedValue(predictedPrice, minPrice, maxPrice);
        printf("Test Record %d: Predicted Price: $%.2f\n", i + 1, originalPredictedPrice * 1000 * 8.12);
    }

    // Free allocated memory
    nn_free(&nn);
    free(filteredData);
    free(data_train);
    free(data_test);
    free(data);

    return 0;
}














// FUNCTIONS
double *read_csv(const char *filename, int *num_records)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        printf("Could not open the file\n");
        return NULL;
    }

    char line[1024];
    int count = 0;

    while (fgets(line, 1024, file) != NULL)
    {
        count++;
    }

    fseek(file, 0, SEEK_SET);

    // Allocate a single big array for all data
    double *data = malloc(count * TOTAL_COLUMNS * sizeof(double));
    if (!data)
    {
        printf("Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    int row = 0;
    while (fgets(line, 1024, file) != NULL)
    {
        char *token = strtok(line, " ");
        int column = 0;

        while (token && column < TOTAL_COLUMNS)
        {
            data[row * TOTAL_COLUMNS + column] = atof(token);
            token = strtok(NULL, " ");
            column++;
        }

        row++;
    }

    fclose(file);
    *num_records = count;
    return data;
}

void printFilteredData(double *matrix, int num_records)
{
    const char *featureNames[NUM_FEATURES_FILTERED] = {
        "RM", "LSTAT", "PTRATIO", "CRIM", "NOX", "DIS"};

    for (int i = 0; i < num_records; i++)
    {
        printf("Record %d:\n", i + 1);
        for (int j = 0; j < NUM_FEATURES_FILTERED; j++)
        {
            printf(" %s: %f", featureNames[j], matrix[i * TOTAL_COLUMNS_FILTERED + j]);
        }
        printf("\n Target (Price): %f\n\n", matrix[i * TOTAL_COLUMNS_FILTERED + NUM_FEATURES_FILTERED]);
    }
}

void filterDataToMatrix(double *data, int num_records, double *matrix)
{
    for (int i = 0; i < num_records; i++)
    {
        // Extracting specific features and placing them in the new matrix
        matrix[i * TOTAL_COLUMNS_FILTERED + 0] = data[i * TOTAL_COLUMNS + 5];  // RM
        matrix[i * TOTAL_COLUMNS_FILTERED + 1] = data[i * TOTAL_COLUMNS + 12]; // LSTAT
        matrix[i * TOTAL_COLUMNS_FILTERED + 2] = data[i * TOTAL_COLUMNS + 10]; // PTRATIO
        matrix[i * TOTAL_COLUMNS_FILTERED + 3] = data[i * TOTAL_COLUMNS + 0];  // CRIM
        matrix[i * TOTAL_COLUMNS_FILTERED + 4] = data[i * TOTAL_COLUMNS + 4];  // NOX
        matrix[i * TOTAL_COLUMNS_FILTERED + 5] = data[i * TOTAL_COLUMNS + 7];  // DIS

        matrix[i * TOTAL_COLUMNS_FILTERED + 6] = data[i * TOTAL_COLUMNS + 13]; // TARGET
    }
}

void getUserInput(double *userInput)
{
    printf("Enter the average number of rooms per dwelling (RM) or press Enter for default (%.2f): ", AVG_RM);
    if (scanf("%lf", &userInput[0]) != 1)
    {
        userInput[0] = AVG_RM;
        while (getchar() != '\n')
            ; // Clear the input buffer
    }

    printf("Enter the percentage of lower status of the population (LSTAT) or press Enter for default (%.2f): ", AVG_LSTAT);
    if (scanf("%lf", &userInput[1]) != 1)
    {
        userInput[1] = AVG_LSTAT;
        while (getchar() != '\n')
            ;
    }

    printf("Enter the pupil-teacher ratio by town (PTRATIO) or press Enter for default (%.2f): ", AVG_PTRATIO);
    if (scanf("%lf", &userInput[2]) != 1)
    {
        userInput[2] = AVG_PTRATIO;
        while (getchar() != '\n')
            ;
    }

    printf("Enter the per capita crime rate by town (CRIM) or press Enter for default (%.2f): ", AVG_CRIM);
    if (scanf("%lf", &userInput[3]) != 1)
    {
        userInput[3] = AVG_CRIM;
        while (getchar() != '\n')
            ;
    }

    printf("Enter the nitric oxides concentration (NOX) or press Enter for default (%.2f): ", AVG_NOX);
    if (scanf("%lf", &userInput[4]) != 1)
    {
        userInput[4] = AVG_NOX;
        while (getchar() != '\n')
            ;
    }

    printf("Enter the weighted distances to five Boston employment centres (DIS) or press Enter for default (%.2f): ", AVG_DIS);
    if (scanf("%lf", &userInput[5]) != 1)
    {
        userInput[5] = AVG_DIS;
        while (getchar() != '\n')
            ;
    }
}

void printUserInput(double *userInput)
{
    for (int j = 0; j < NUM_FEATURES_FILTERED; j++)
    {
        printf("%f ", userInput[j]);
    }
    printf("\nTarget: %f\n", 0.0); // Target is set to 0
}

void normalizeFeatures(double *data, int num_records, int num_features, double *minPrice, double *maxPrice)
{
    for (int j = 0; j < num_features; j++)
    {
        double min = DBL_MAX;
        double max = -DBL_MAX;

        // Finding min and max for each feature
        for (int i = 0; i < num_records; i++)
        {
            double value = data[i * TOTAL_COLUMNS_FILTERED + j];
            if (value < min)
                min = value;
            if (value > max)
                max = value;
        }

        // If it's the target variable (price), set minPrice and maxPrice
        if (j == num_features - 1)
        {
            *minPrice = min;
            *maxPrice = max;
        }

        // Normalizing each feature
        for (int i = 0; i < num_records; i++)
        {
            double value = data[i * TOTAL_COLUMNS_FILTERED + j];
            // Rescale to 0 to 1
            data[i * TOTAL_COLUMNS_FILTERED + j] = (value - min) / (max - min);
        }
    }
}


void normalizeUserInput(double *userInput, double *mins, double *maxes)
{
    for (int i = 0; i < NUM_FEATURES_FILTERED; i++)
    {
        userInput[i] = (userInput[i] - mins[i]) / (maxes[i] - mins[i]);
    }
}

void findMinMax(double *data, int num_records, double *mins, double *maxes) {
    // Initialize mins and maxes
    for (int i = 0; i < NUM_FEATURES_FILTERED; i++) {
        mins[i] = DBL_MAX;
        maxes[i] = -DBL_MAX;
    }

    // Iterate through the dataset
    for (int i = 0; i < num_records; i++) {
        for (int j = 0; j < NUM_FEATURES_FILTERED; j++) {
            double value = data[i * TOTAL_COLUMNS_FILTERED + j];
            if (value < mins[j]) {
                mins[j] = value;
            }
            if (value > maxes[j]) {
                maxes[j] = value;
            }
        }
    }
}

double revertNormalizedValue(double normalizedValue, double min, double max) {
    return normalizedValue * (max - min) + min;
}

double predict(Network *nn, double *inputFeatures)
{
    // Set the input features to the input layer of the network
    for (int j = 0; j < NUM_FEATURES_FILTERED; j++)
    {
        MAT_AT(NN_INPUT(*nn), 0, j) = inputFeatures[j];
    }

    // Perform forward pass
    nn_forward(*nn);

    // Assuming the output is a single value, extract it from the output layer
    double predictedPrice = MAT_AT(NN_OUTPUT(*nn), 0, 0);

    return predictedPrice;
}

void nn_free(Network *nn)
{
    if (nn != NULL)
    {
        // Free weights and biases for each layer
        for (size_t i = 0; i < nn->count; ++i)
        {
            mat_free(nn->ws[i]);
            mat_free(nn->bs[i]);
        }

        // Free the matrices arrays themselves
        free(nn->ws);
        free(nn->bs);

        // Free activations for each layer, including the input layer
        for (size_t i = 0; i <= nn->count; ++i)
        {
            mat_free(nn->as[i]);
        }
        free(nn->as);
    }
}