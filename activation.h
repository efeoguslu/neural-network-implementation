#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define EULER_NUMBER 2.71828182846

// ---------------------------------------------------------

typedef double (*ActivationFunction)(double);
extern ActivationFunction g_activation_func;

double identity(double);
int binary_step(double);
int d_identity(double);
int d_binary_step(double);
double sigmoid(double n);
double d_sigmoid(double n);
double relu(double n);

#endif

