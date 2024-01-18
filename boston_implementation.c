#include "matrix.h"
#include "activation.h"
#include "csv.h"

int main()
{
    int num_records;
    double *data = read_csv("housing.csv", &num_records);
    double *filteredData = malloc(num_records * TOTAL_COLUMNS_FILTERED * sizeof(double));

    filterDataToMatrix(data, num_records, filteredData);
    normalizeFeatures(filteredData, num_records, NUM_FEATURES);

    Mat t = mat_alloc(NUM_RECORDS_TRAIN, TOTAL_COLUMNS_FILTERED);

    for (int i = 0; i < num_records; ++i)clea
    {
        for (int j = 0; j < TOTAL_COLUMNS_FILTERED; ++j)
        {
            size_t pos = i * TOTAL_COLUMNS_FILTERED + j;
            MAT_AT(t, i, j) = filteredData[pos];
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

    size_t arch[] = {6, 8, 5, 1}; // 6 16 16 1 cok iyiydi

    Network nn = nn_alloc(arch, ARRAY_LEN(arch));
    Network g  = nn_alloc(arch, ARRAY_LEN(arch));

    nn_rand(nn, 0, 1);
    double rate = 3.8;
    size_t epoch = 50000;

    printf("learning...\n");
    for(size_t i = 0; i < epoch; ++i){
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);

        printf("epoch: %zu\t cost: %lf\t learning rate: %.1lf\t\n", i, nn_cost(nn, ti, to), rate);
    }

    double userInput[NUM_FEATURES_FILTERED];
    getUserInput(userInput);

    double mins[NUM_FEATURES_FILTERED];
    double maxes[NUM_FEATURES_FILTERED];
    double minPrice = 5.0;
    double maxPrice = 50.0;

    findMinMax(filteredData, num_records, mins, maxes);
    normalizeUserInput(userInput, mins, maxes);

    double predictedPrice = predict(&nn, userInput);
    double originalPredictedPrice = revertNormalizedValue(predictedPrice, minPrice, maxPrice);
    printf("1970 Predicted Price: $%.2f\n", originalPredictedPrice*1000);
    printf("2023 Predicted Price: $%.2f\n", originalPredictedPrice*1000*8.12);

    nn_free(&nn);
    free(filteredData);
    free(data);

    return 0;
}
