#include "activation.h"
#include "helpers.h"
#include "xor.h"
#include "matrix.h"
#include "mnist.h"
#include "csv.h"
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


int main() {
    const char *file_path = "/home/efeog/Desktop/neural-network-implementation";
    size_t label_column = 13; // Assuming the target variable (e.g., MEDV) is in the last column

    // Load CSV data
    Dataset *dataset = load_csv(file_path);
    if (!dataset) {
        fprintf(stderr, "Error loading CSV data\n");
        return 1;
    }

    // Split dataset into features and labels
    Mat features, labels;
    split_dataset(dataset, label_column, &features, &labels);

    // Display or use the features and labels as needed


    // Free allocated memory
    // free_dataset(dataset);
    // mat_free(features);
    // mat_free(labels);

    preprocess_data(&features);

    size_t arch[] = {14, 8, 1};
    Network nn = nn_alloc(arch, ARRAY_LEN(arch));

    NN_PRINT(nn);


    return 0;
}