#include "matrix.h"
#include "activation.h"
#include "csv.h"
#include "helpers.h"

void printPredictions(Network *nn, Mat t_test, double *mins, double *maxes)
{
    printf("Sample\tActual Price\tPredicted Price\tError Percentage\n");

    for (int i = 0; i < t_test.rows; ++i)
    {
        // Perform forward pass
        nn_forward(*nn);

        // Get the predicted price
        double predictedPrice = MAT_AT(NN_OUTPUT(*nn), i, 0);

        // Get the actual price (reverted from normalization)
        double actualPrice = revertNormalizedValue(MAT_AT(t_test, i, TOTAL_COLUMNS_FILTERED - 1), mins[NUM_FEATURES_FILTERED - 1], maxes[NUM_FEATURES_FILTERED - 1]);

        // Calculate error percentage
        double errorPercentage = fabs((actualPrice - predictedPrice) / actualPrice) * 100.0;

        printf("%d\t%.2f\t\t%.2f\t\t%.2f%%\n", i + 1, actualPrice, predictedPrice, errorPercentage);
    }
}

int main()
{
    int num_records_train;
    double *data_train = read_csv("housing_train.csv", &num_records_train);
    double *filteredData_train = malloc(num_records_train * TOTAL_COLUMNS_FILTERED * sizeof(double));

    filterDataToMatrix(data_train, num_records_train, filteredData_train);
    normalizeFeatures(filteredData_train, num_records_train, NUM_FEATURES);

    Mat t = mat_alloc(NUM_RECORDS_TRAIN, TOTAL_COLUMNS_FILTERED);

    for (int i = 0; i < num_records_train; ++i)
    {
        for (int j = 0; j < TOTAL_COLUMNS_FILTERED; ++j)
        {
            size_t pos = i * TOTAL_COLUMNS_FILTERED + j;
            MAT_AT(t, i, j) = filteredData_train[pos];
        }
    }

    MAT_PRINT(t);
    
    Mat ti = {
        .rows = t.rows,
        .cols = 6,
        .stride = t.stride,
        .es = &MAT_AT(t, 0, 0),
    };

    Mat to = {
        .rows = t.rows,
        .cols = 1,
        .stride = t.stride,
        .es = &MAT_AT(t, 0, ti.cols),
    };

    MAT_PRINT(ti);
    MAT_PRINT(to);

    size_t arch[] = {6, 8, 5, 1}; 

    Network nn = nn_alloc(arch, ARRAY_LEN(arch));
    Network g  = nn_alloc(arch, ARRAY_LEN(arch));

    nn_rand(nn, 0, 1);
    double rate = 3.8;
    size_t epoch = 10000;

    printf("learning...\n");
    for(size_t i = 0; i < epoch; ++i){
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);

        printf("epoch: %zu\t cost: %lf\t learning rate: %.1lf\t\n", i, nn_cost(nn, ti, to), rate);
    }

    double mins[NUM_FEATURES_FILTERED];
    double maxes[NUM_FEATURES_FILTERED];

    findMinMax(filteredData_train, num_records_train, mins, maxes);

    
    // Load the test data
    int num_records_test;
    double *data_test = read_csv("housing_test.csv", &num_records_test);
    double *filteredData_test = malloc(num_records_test * TOTAL_COLUMNS_FILTERED * sizeof(double));

    filterDataToMatrix(data_test, num_records_test, filteredData_test);
    normalizeFeatures(filteredData_test, num_records_test, NUM_FEATURES_FILTERED);

    Mat t_test = mat_alloc(num_records_test, TOTAL_COLUMNS_FILTERED);

    for (int i = 0; i < num_records_test; ++i)
    {
        for (int j = 0; j < TOTAL_COLUMNS_FILTERED; ++j)
        {
            size_t pos = i * TOTAL_COLUMNS_FILTERED + j;
            MAT_AT(t_test, i, j) = filteredData_test[pos];
        }
    }

    MAT_PRINT(t_test);

    Mat ti_test = {
        .rows = t_test.rows,
        .cols = 6,
        .stride = t_test.stride,
        .es = &MAT_AT(t_test, 0, 0),
    };

    // Perform feedforward on the test data and compare network outputs with actual outputs
    printf("Sample\tActual Price\tPredicted Price\tError Percentage\n");
    dline();

    for (int i = 0; i < num_records_test; ++i)
{
    // Extract input features for the current sample
    double userInput[NUM_FEATURES_FILTERED];
    for (int j = 0; j < NUM_FEATURES_FILTERED; ++j)
    {
        userInput[j] = MAT_AT(ti_test, i, j);
    }

    // Predict using the trained network
    double predictedPrice = predict(&nn, userInput);

    // Revert normalization for the actual price
    double actualPrice = revertNormalizedValue(MAT_AT(t_test, i, TOTAL_COLUMNS_FILTERED - 1),
                                               mins[NUM_FEATURES_FILTERED - 1],
                                               maxes[NUM_FEATURES_FILTERED - 1]);

    // Calculate and print error percentage
    double errorPercentage = fabs((actualPrice - predictedPrice) / actualPrice) * 100.0;
    printf("%d\t%.2f\t\t%.2f\t\t%.2f%%\n", i + 1, actualPrice, predictedPrice, errorPercentage);
}

    nn_free(&nn);
    free(filteredData_train);
    free(filteredData_test);
    free(data_train);
    free(data_test);

    return 0;
}
