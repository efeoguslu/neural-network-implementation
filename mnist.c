#include "mnist.h"

/*
// Function to load MNIST data
double** load_mnist_data(const char* image_file, const char* label_file, int num_images) {
    FILE* fp_images = fopen(image_file, "rb");
    FILE* fp_labels = fopen(label_file, "rb");
    
    if (!fp_images || !fp_labels) {
        printf("Failed to open MNIST files\n");
        return NULL;
    }

    // Skip the header
    fseek(fp_images, 16, SEEK_SET);
    fseek(fp_labels, 8, SEEK_SET);

    double** data = malloc(sizeof(double*) * num_images);
    
    for (int i = 0; i < num_images; i++) {
        data[i] = malloc(sizeof(double) * (IMAGE_SIZE * IMAGE_SIZE + 1));  // +1 for the label

        // Read the label
        uint8_t label;
        fread(&label, sizeof(label), 1, fp_labels);
        data[i][0] = (double)label;

        // Read the image
        for (int j = 0; j < IMAGE_SIZE * IMAGE_SIZE; j++) {
            uint8_t pixel;
            fread(&pixel, sizeof(pixel), 1, fp_images);
            data[i][j + 1] = (double)pixel;
        }
    }

    fclose(fp_images);
    fclose(fp_labels);

    return data;
}

// Function to preprocess MNIST data
void preprocess_data(double** data, int num_images) {
    for (int i = 0; i < num_images; i++) {
        for (int j = 1; j <= IMAGE_SIZE * IMAGE_SIZE; j++) {
            // Scale pixel values to [0, 1]
            data[i][j] /= 255.0;
        }
    }
}


*/