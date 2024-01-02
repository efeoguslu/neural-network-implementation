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

#define BITS 2

int main(){

    size_t n = (1 << BITS); // for BITS = 2: from 0001 ---> 0100
    size_t rows = n*n;
    Mat ti = mat_alloc(rows, 2*BITS);
    Mat to = mat_alloc(rows, BITS + 1);

    for(size_t i = 0; i < ti.rows; ++i){
        size_t x = i / n;
        size_t y = i % n;
        size_t z = x + y;
        MAT_AT(to, i, BITS) = z < n;
        for(size_t j = 0; j < BITS; ++j){
            if(z < n){
                MAT_AT(ti, i, j) = (x >> j)&1;
                MAT_AT(ti, i, j + BITS) = (y >> j)&1;
                MAT_AT(to, i, j) = (z >> j)&1;
            }
            else{
                MAT_AT(ti, i, j) = 0;
                MAT_AT(ti, i, j + BITS) = 0;
                MAT_AT(to, i, j) = 0;
            }
        }
    }

    MAT_PRINT(ti);
}