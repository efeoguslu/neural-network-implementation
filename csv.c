#include "csv.h"


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

void normalizeFeatures(double *data, int num_records, int num_features)
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

double predict(Network *nn, double *input)
{
    // Normalize input features using the same mins and maxes values used during training
    for (int i = 0; i < NUM_FEATURES_FILTERED - 1; ++i)
    {
        input[i] = normalizeValue(input[i], mins[i], maxes[i]);
    }

    // Perform feedforward
    nn_feedforward(nn, input);

    // Output of the network (assuming one output neuron)
    return nn_get_output(nn, 0);
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