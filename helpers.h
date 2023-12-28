#ifndef HELPERS_H
#define HELPERS_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include "activation.h"

void randomize(void);
void dline(void);
// void menu(int* num_layers, int** neurons_per_layer, ActivationFunction* activation_func);
void test_model(double w1, double w2, double b);
double rand_double(void);


char *args_shift(int *argc, char ***argv);
void cool_terminal_print(int img_height, int img_width, uint8_t *img_pixels);
void print_to_terminal(uint8_t pixel);

#endif

