#ifndef XOR_H
#define XOR_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// ---------------------------------------------------------

typedef struct{
    double or_w1;
    double or_w2;
    double or_b;

    double nand_w1;
    double nand_w2;
    double nand_b;

    double and_w1;
    double and_w2;
    double and_b;
} Xor;

Xor rand_xor();
double forward(Xor m, double x1, double x2);
double xor_cost(Xor m);
void print_xor(Xor m);


#endif

