#include "ctorch.h"
#include "stdlib.h"
#include "math.h"
#include "stdio.h"
#include "string.h"


float ctorch_util_normal(float mu, float sigma) {
    float f1;
    do {
        f1 = (float)rand() / (float)RAND_MAX;
    } while (f1 == 0.0);
    float f2 = (float)rand() / (float)RAND_MAX;
    float mag = sigma * sqrt(-2.0f * log(f1));
    mag = (mag * cos(2 * 3.14159f * f2) + mu);
    if (abs(mag) > rand_weight_clip) {
        mag = rand_weight_clip * (1 - 2 * (mag < 0));
    }
    return mag;
}

void ctorch_floatvector_print(float* input_vector, int size) {
    printf("[");
    for (int i = 0; i < size - 1; i++) {
        printf("%f,", input_vector[i]);
    }
    printf("%f]\n", input_vector[size - 1]);
}
