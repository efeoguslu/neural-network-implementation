#include "matrix.h"

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
void normalizeFeatures(double *data, int num_records, int num_features);

void getUserInput(double *userInput);
void normalizeUserInput(double *userInput, double *mins, double *maxes);

void trainNetwork(Network *nn, double *filteredData, int num_records, int epochs, double learning_rate, size_t *arch, size_t arch_len);
double predict(Network *nn, double *inputFeatures);

void nn_free(Network *nn);

int main()
{
    // DATA OPERATIONS
    int num_records;
    double *data = read_csv("housing.csv", &num_records);
    double *filteredData = malloc(num_records * TOTAL_COLUMNS_FILTERED * sizeof(double));

    filterDataToMatrix(data, num_records, filteredData);
    normalizeFeatures(filteredData, num_records, NUM_FEATURES_FILTERED);
    printFilteredData(filteredData, num_records);

    // NETWORK
    size_t arch[] = {6, 8, 8, 1};
    Network nn = nn_alloc(arch, sizeof(arch) / sizeof(arch[0]));
    nn_rand(nn, -1.0, 1.0);

    int epochs = 1000;
    double learning_rate = 0.01;

    trainNetwork(&nn, filteredData, num_records, epochs, learning_rate, arch, sizeof(arch) / sizeof(arch[0]));

    // PREDICTION
    double mins[NUM_FEATURES_FILTERED];
    double maxes[NUM_FEATURES_FILTERED];
    for (int j = 0; j < NUM_FEATURES_FILTERED; j++) {
        mins[j] = DBL_MAX;
        maxes[j] = -DBL_MAX;
        for (int i = 0; i < num_records; i++) {
            double value = filteredData[i * TOTAL_COLUMNS_FILTERED + j];
            if (value < mins[j]) mins[j] = value;
            if (value > maxes[j]) maxes[j] = value;
        }
    }

    double userInput[NUM_FEATURES_FILTERED];

    getUserInput(userInput);
    normalizeUserInput(userInput, mins, maxes); 

    double predictedPrice = predict(&nn, userInput);
    printf("Predicted Price: %f\n", predictedPrice);

    free(filteredData);
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

void normalizeFeatures(double *data, int num_records, int num_features) {
    for (int j = 0; j < num_features; j++) {
        double min = DBL_MAX;
        double max = -DBL_MAX;

        // Finding min and max for each feature
        for (int i = 0; i < num_records; i++) {
            double value = data[i * TOTAL_COLUMNS_FILTERED + j];
            if (value < min) min = value;
            if (value > max) max = value;
        }

        // Normalizing each feature
        for (int i = 0; i < num_records; i++) {
            double value = data[i * TOTAL_COLUMNS_FILTERED + j];
            // Rescale to -1 to 1
            data[i * TOTAL_COLUMNS_FILTERED + j] = 2 * (value - min) / (max - min) - 1;
        }
    }
}


void normalizeUserInput(double *userInput, double *mins, double *maxes) {
    for (int i = 0; i < NUM_FEATURES_FILTERED; i++) {
        userInput[i] = 2 * (userInput[i] - mins[i]) / (maxes[i] - mins[i]) - 1;
    }
}

void trainNetwork(Network *nn, double *filteredData, int num_records, int epochs, double learning_rate, size_t *arch, size_t arch_len)
{
    Network gradient = nn_alloc(arch, arch_len);

    // Creating matrices for a single training example
    Mat ti = mat_alloc(1, NUM_FEATURES_FILTERED); // One row, NUM_FEATURES_FILTERED columns for input
    Mat to = mat_alloc(1, 1);                     // One row, one column for target

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double total_cost = 0.0;

        for (int i = 0; i < num_records; i++)
        {
            // Setting up input features (ti) for this record
            for (int j = 0; j < NUM_FEATURES_FILTERED; j++)
            {
                MAT_AT(ti, 0, j) = filteredData[i * TOTAL_COLUMNS_FILTERED + j];
            }

            // Setting up target value (to) for this record
            MAT_AT(to, 0, 0) = filteredData[i * TOTAL_COLUMNS_FILTERED + NUM_FEATURES_FILTERED];

            // Forward pass
            mat_copy(NN_INPUT(*nn), ti); // Copy ti matrix to network's input
            nn_forward(*nn);

            // Compute cost (mean squared error)
            double predicted = MAT_AT(NN_OUTPUT(*nn), 0, 0);
            double cost = (predicted - MAT_AT(to, 0, 0)) * (predicted - MAT_AT(to, 0, 0));
            total_cost += cost;

            // Backward pass and update weights
            nn_backprop(*nn, gradient, ti, to);
            nn_learn(*nn, gradient, learning_rate);
        }

        total_cost /= num_records;
        printf("Epoch %d, Cost: %f\n", epoch, total_cost);
    }

    // Free the gradient network and matrices after training
    nn_free(&gradient);
    mat_free(ti);
    mat_free(to);
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
