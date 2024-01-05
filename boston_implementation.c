#include "matrix.h"

const char *sample_file_path = "/file/path/yadayadada...";
 
#define ROWS 506
#define COLS 14

int main(){

    printf("hello baris!\n");

    double *data_Set = get_csv_as_array(sample_file_path);

    Mat t = mat_alloc(ROWS, COLS);

    for(int y = 0; y < ROWS; ++y){
        for(int x = 0; x < COLS; ++x){
            /*
            size_t i = y*COLS + x;
            MAT_AT(t, i, 0) = 
            MAT_AT(t, i, 1) = 
            MAT_AT(t, i, 2) = 
            MAT_AT(t, i, 3) = 
            ...
            MAT_AT(t, i, 13) = 
            */
        }
    }

    Mat ti = {
        .rows = t.rows,
        .cols = (COLS - 1),
        .stride = t.stride,
        .es = &MAT_AT(t, 0, 0),
    };

    Mat to = {
        .rows = t.rows,
        .cols = 1,
        .stride = t.stride,
        .es = &MAT_AT(t, 0, ti.cols),
    };
    

    size_t arch[] = {2, 16, 16, 1};
    Network nn = nn_alloc(arch, ARRAY_LEN(arch));
    Network g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);

    double rate = 2.5;
    size_t epoch = 25000;

    for(size_t i = 0; i < epoch; ++i){

        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);

        //fprintf(cost_file, "%lf\n", nn_cost(nn, ti, to));

        /*
        if(i % 100 == 0){
            
            /*
            for(size_t y = 0; y < (size_t)img_height; ++y){
                for(size_t x = 0; x < (size_t)img_width; ++x){
                    MAT_AT(NN_INPUT(nn), 0, 0) = (double)x/(img_width - 1);
                    MAT_AT(NN_INPUT(nn), 0, 1) = (double)y/(img_height - 1);
                    nn_forward(nn);
                    uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.0;
                    print_to_terminal(pixel);
                }
                printf("\n");
            }
            */


        }
        
        
    }

}