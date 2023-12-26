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

// training data:

// upscale --> 43.06

/*
double td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};
*/

const char *out_directory = "/home/efeog/Desktop/mnist-matrix";
const char *out_file_name = "img.mat";

const char *img_file_path = "/home/efeog/Desktop/number_image/281.png";

int main(int argc, char **argv){

    const char *program = args_shift(&argc, &argv);

    /*
    if(argc <= 0){
        fprintf(stderr, "Usage: %s <input>\n", program);
        fprintf(stderr, "ERROR: no input file provided\n");
        return 1;
    }

    const char *img_file_path = args_shift(&argc, &argv);
    */

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

    // return 0;

    /*/

    char out_file_path[256];
    snprintf(out_file_path, sizeof(out_file_path), "%s/%s", out_directory, out_file_name);
    FILE *out = fopen(out_file_path, "wb");

    if(out == NULL){
        fprintf(stderr, "ERROR: could not open file %s\n", out_file_path);
        return 1;
    }

    mat_save(out, t);
    fclose(out);

    printf("Generated %s from %s\n", out_file_path, img_file_path);

    // cool_terminal_print(img_height, img_width, img_pixels);
    */

    size_t arch[] = {2, 28, 28, 1};
    Network nn = nn_alloc(arch, ARRAY_LEN(arch));
    Network g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);

    double rate = 2.0;


    size_t epoch = 50000;

    for(size_t i = 0; i < epoch; ++i){
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
        if(i % 100 == 0) printf("%zu: cost = %lf\n", i, nn_cost(nn, ti, to));
    }

    cool_terminal_print(img_height, img_width, img_pixels);

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








    return 0;
}

    // ---------------------------------------------------------------------------------------------

    /*

    randomize();
    
    size_t stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/stride;

    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2,
    }; 

    size_t arch[] = {2, 2, 1};
    
    // memory management:

    Network network = nn_alloc(arch, ARRAY_LEN(arch));
    Network gradient = nn_alloc(arch, ARRAY_LEN(arch));


    FILE *fp;  // File pointer
    fp = fopen("cost.txt", "w");  // Open file in write mode

    if (fp == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    // double eps = 1e-1;
    double rate = 1e-1;
    size_t epoch = 50000;
    nn_rand(network, 0, 1);

    clock_t start, end;
    start = clock();

    printf("cost = %lf\n", nn_cost(network, ti, to));

    for(size_t i = 0; i < epoch; ++i){
#if 0
        double eps = 1e-1;
        nn_finite_diff(network, gradient, eps, ti, to);
#else
        nn_backprop(network, gradient, ti, to);
#endif
        //NN_PRINT(gradient);
        nn_learn(network, gradient, rate);
        printf("cost = %lf\n", nn_cost(network, ti, to));
        fprintf(fp, "%lf\n", nn_cost(network, ti, to));
    }

    end = clock();

    dline();
    NN_PRINT(gradient);
    dline();
    NN_PRINT(network);
    dline();

    fclose(fp);

    printf("\nTest:\n");
    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){
            MAT_AT(NN_INPUT(network), 0, 0) = i;
            MAT_AT(NN_INPUT(network), 0, 1) = j;
            nn_forward(network);
            printf("%lu ^ %lu = %lf\n", i, j, MAT_AT(NN_OUTPUT(network), 0, 0));
        }
    }

    
    double dblTime = ((double)(end - start)) / CLOCKS_PER_SEC;
    dline();
    printf("\n%lf seconds have elapsed during the training process.\n", dblTime);

    return 0;
}
*/