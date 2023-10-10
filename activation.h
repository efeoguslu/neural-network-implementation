#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// --- Move these elsewhere ---

void randomize(void);
void dline(void);
void menu(void);

// ---------------------------------------------------------

double identity(double);
int binary_step(double);
int d_identity(double);
int d_binary_step(double);


#endif
