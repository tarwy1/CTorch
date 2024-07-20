#include "stdio.h"
#include "stdlib.h"
#define uint unsigned int

typedef struct{
    uint length;
    float *value_array;
} FloatVector;

typedef struct{
    uint *dimensions;
    FloatVector *vector_array;
} FloatTensor;

FloatVector* ctorch_floatvector_alloc(uint length);

FloatTensor* ctorch_floattensor_alloc(uint outer_dim, uint inner_dim);
