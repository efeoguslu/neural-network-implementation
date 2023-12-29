#include "csv.h"

void free_dataset(Dataset *dataset) {
    for (size_t i = 0; i < dataset->num_rows; ++i) {
        free(dataset->rows[i].values);
    }
    free(dataset->rows);
    free(dataset);
}


// Function to load CSV data from a file
Dataset *load_csv(const char *file_path) {
    FILE *file = fopen(file_path, "r");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    char line[MAX_LINE_LENGTH];
    size_t num_rows = 0;

    // Count the number of rows in the CSV file
    while (fgets(line, sizeof(line), file)) {
        ++num_rows;
    }

    // Rewind the file to read data
    rewind(file);

    // Read the header to determine the number of columns
    fgets(line, sizeof(line), file);
    char *token = strtok(line, ",");
    size_t num_columns = 0;

    while (token) {
        ++num_columns;
        token = strtok(NULL, ",");
    }

    // Rewind the file again
    rewind(file);

    // Allocate memory for the dataset
    Dataset *dataset = (Dataset *)malloc(sizeof(Dataset));
    if (!dataset) {
        perror("Error allocating memory for dataset");
        fclose(file);
        return NULL;
    }

    dataset->num_rows = num_rows;
    dataset->num_columns = num_columns;
    dataset->rows = (DataRow *)malloc(num_rows * sizeof(DataRow));

    if (!dataset->rows) {
        perror("Error allocating memory for rows");
        free(dataset);
        fclose(file);
        return NULL;
    }

    // Read data from the CSV file
    for (size_t i = 0; i < num_rows; ++i) {
        dataset->rows[i].values = (double *)malloc(num_columns * sizeof(double));

        if (!dataset->rows[i].values) {
            perror("Error allocating memory for values");
            free_dataset(dataset);
            fclose(file);
            return NULL;
        }

        fgets(line, sizeof(line), file);
        token = strtok(line, ",");
        size_t j = 0;

        while (token) {
            if (strcmp(token, "NA") == 0) {
                // Handle missing values, replace with the mean or median of the column
                // For simplicity, we replace missing values with 0.0 in this example
                dataset->rows[i].values[j] = 0.0;
            } else {
                dataset->rows[i].values[j] = atof(token);
            }

            ++j;
            token = strtok(NULL, ",");
        }
    }

    fclose(file);
    return dataset;
}


// Function to split the dataset into features and labels
void split_dataset(const Dataset *dataset, size_t label_column, Mat *features, Mat *labels) {
    // Assume label_column is the column index for the target variable
    size_t num_features = dataset->num_columns - 1;

    // Allocate memory for features and labels matrices
    *features = mat_alloc(dataset->num_rows, num_features);
    *labels = mat_alloc(dataset->num_rows, 1);

    // Populate features and labels matrices
    for (size_t i = 0; i < dataset->num_rows; ++i) {
        for (size_t j = 0, feature_idx = 0; j < dataset->num_columns; ++j) {
            if (j != label_column) {
                MAT_AT(*features, i, feature_idx) = dataset->rows[i].values[j];
                ++feature_idx;
            } else {
                MAT_AT(*labels, i, 0) = dataset->rows[i].values[j];
            }
        }
    }
}

void preprocess_data(Mat *features) {
    for (size_t j = 0; j < features->cols; ++j) {
        double column_sum = 0.0;
        size_t num_valid_entries = 0;

        // Calculate the sum of the column and count valid entries
        for (size_t i = 0; i < features->rows; ++i) {
            if (MAT_AT(*features, i, j) != 0.0) {
                column_sum += MAT_AT(*features, i, j);
                ++num_valid_entries;
            }
        }

        // Calculate the mean of the column
        double column_mean = (num_valid_entries > 0) ? column_sum / num_valid_entries : 0.0;

        // Replace missing values with the mean
        for (size_t i = 0; i < features->rows; ++i) {
            if (MAT_AT(*features, i, j) == 0.0) {
                MAT_AT(*features, i, j) = column_mean;
            }
        }
    }
}