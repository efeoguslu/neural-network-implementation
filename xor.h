#ifndef XOR_H
#define XOR_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


// ---------------------------------------------------------

double cost_two_inputs(double w1, double w2, double bias);

// outline of what we expect it to act like. but it can find a better configuration

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
Xor finite_difference(Xor m, double eps);
Xor subtract_gradient(Xor m, Xor g, double rate);
void test_xor_model(Xor m);
void save_weights(const char *filename, Xor model);

#endif