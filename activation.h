#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define EULER_NUMBER 2.71828182846

// ---------------------------------------------------------

double identity(double);
int binary_step(double);
int d_identity(double);
int d_binary_step(double);
double sigmoid(double n);
double d_sigmoid(double n);


#endif

