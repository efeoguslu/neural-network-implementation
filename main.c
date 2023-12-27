#include "activation.h"
#include "helpers.h"
#include "xor.h"
#include "matrix.h"
#include "mnist.h"
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


const char *out_directory = "/home/efeog/Desktop/mnist-matrix";
const char *out_file_name = "img.mat";

const char *img_file_path = "/home/efeog/Desktop/number_image/281.png";

const char *output_directory = "/home/efeog/Desktop/final_image_output";
const char *output_image_name = "output.png";

int main(int argc, char **argv){

    const char *program = args_shift(&argc, &argv);

    if(mkdir(out_directory, 0777) == -1 && errno != EEXIST) {
        perror("Error creating directory");
        return 1;
    }

    int img_width, img_height, img_comp;
    uint8_t *img_pixels = (uint8_t *)stbi_load(img_file_path, &img_width, &img_height, &img_comp, 0);

    if(img_pixels == NULL){
        fprintf(stderr, "ERROR: could not read image %s\n", img_file_path);
        return 1;
    }

    if(img_comp != 1){
        fprintf(stderr, "ERROR: %s is %d bits image. Only 8 bit grayscale images are supported.\n", img_file_path, img_comp*8);
        return 1;
    }

    printf("%s size %dx%d %d bits\n", img_file_path, img_width, img_height, img_comp*8);

    Mat t = mat_alloc(img_width*img_height, 3);

    for(int y = 0; y < img_height; ++y){
        for(int x = 0; x < img_width; ++x){
            size_t i = y*img_width + x;
            double nx = (double)x/(img_width - 1);
            double ny = (double)y/(img_height - 1);
            double nb = img_pixels[i]/255.0;
            MAT_AT(t, i, 0) = nx;
            MAT_AT(t, i, 1) = ny;
            MAT_AT(t, i, 2) = nb;
        }
    }

    Mat ti = {
        .rows = t.rows,
        .cols = 2,
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

    size_t arch[] = {2, 28, 28, 16, 1};
    Network nn = nn_alloc(arch, ARRAY_LEN(arch));
    Network g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);

    double rate = 2.5;
    size_t epoch = 50000;
    clock_t start, end;
    start = clock();

    for(size_t i = 0; i < epoch; ++i){
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
        if(i % 100 == 0){
            printf("epoch: %zu\t cost = %lf\tlearning rate = %.1lf\t\n", i, nn_cost(nn, ti, to), rate);
            printf("Enhanced Training Matrix at Epoch %zu: \n", i);
            dline();
            for(size_t y = 0; y < (size_t)img_height; ++y){
                for(size_t x = 0; x < (size_t)img_width; ++x){
                    MAT_AT(NN_INPUT(nn), 0, 0) = (double)x/(img_width - 1);
                    MAT_AT(NN_INPUT(nn), 0, 1) = (double)y/(img_height - 1);
                    nn_forward(nn);
                    uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.0;

                    if(pixel != 0) printf("%3u ", pixel);
                    else printf("    ");

                }
                printf("\n");
            }

        }
    }

    end = clock();



    dline();
    printf("Original Image Matrix:\n");
    dline();
    cool_terminal_print(img_height, img_width, img_pixels);
    dline();
    printf("Enhanced Training Matrix: \n");
    dline();
    for(size_t y = 0; y < (size_t)img_height; ++y){
        for(size_t x = 0; x < (size_t)img_width; ++x){
            MAT_AT(NN_INPUT(nn), 0, 0) = (double)x/(img_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (double)y/(img_height - 1);
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.0;
            printf("%3u ", pixel);
        }
        printf("\n");
    }

    if(mkdir(output_directory, 0777) == -1 && errno != EEXIST){
        perror("Error creating output image directory");
        return 1;
    }

    char output_image_path[256];
    snprintf(output_image_path, sizeof(output_image_path), "%s/%s", output_directory, output_image_name);

    FILE *output_image = fopen(output_image_path, "wb");
    if (output_image == NULL) {
        fprintf(stderr, "ERROR: could not open file %s\n", output_image_path);
        return 1;
    }

    uint8_t *output_pixels = (uint8_t *)malloc(img_width * img_height);

    for (size_t y = 0; y < (size_t)img_height; ++y) {
        for (size_t x = 0; x < (size_t)img_width; ++x) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (double)x / (img_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (double)y / (img_height - 1);
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.0;
            output_pixels[y * img_width + x] = pixel;
        }
    }

    if (stbi_write_png(output_image_path, img_width, img_height, 1, output_pixels, img_width) == 0) {
        fprintf(stderr, "ERROR: could not write PNG file %s\n", output_image_path);
        fclose(output_image);
        free(output_pixels);
        return 1;
    }      

    fclose(output_image);
    free(output_pixels);

    printf("Generated %s from trained matrix\n", output_image_path);

    double dblTime = ((double)(end - start)) / CLOCKS_PER_SEC;
    dline();
    printf("\n%lf seconds have elapsed during the training process.\n", dblTime);

    return 0;
}

