#include "ctorch.h"
#include "stdlib.h"
#include "math.h"
#include "stdio.h"
#include "string.h"


float ctorch_util_normal(float mu, float sigma){
    float f1;
    do{
        f1 = (float)rand() / (float)RAND_MAX;
    } while(f1==0.0);
    float f2 = (float)rand() / (float)RAND_MAX;
    float mag = sigma * sqrt(-2.0f * log(f1));
    return (mag * cos(2 * 3.14159f * f2) + mu);
}

UintVector* ctorch_uintvector_create(uint length){
    static UintVector temp_vector;
    temp_vector.length = length;
    temp_vector.value_array = malloc((sizeof *temp_vector.value_array) * length);
    memset(temp_vector.value_array, 0, length*(sizeof *temp_vector.value_array));
    return &temp_vector;
}
FloatVector* ctorch_floatvector_create(uint length){
    static FloatVector temp_vector;
    temp_vector.length = length;
    temp_vector.value_array = malloc((sizeof *temp_vector.value_array) * length);
    memset(temp_vector.value_array, 0.0f, length*(sizeof *temp_vector.value_array));
    return &temp_vector;
}

FloatTensor* ctorch_floattensor_create(uint outer_dim, uint inner_dim){
    static FloatTensor temp_tensor;
    temp_tensor.dimensions = malloc((sizeof *temp_tensor.dimensions) * 2);
    temp_tensor.dimensions[0] = outer_dim;
    temp_tensor.dimensions[1] = inner_dim;
    temp_tensor.vector_array = malloc((sizeof temp_tensor.vector_array->length) * outer_dim + (sizeof temp_tensor.vector_array->value_array) * outer_dim);
    for(int i = 0; i < outer_dim; i++){
        temp_tensor.vector_array[i].length = inner_dim;
        temp_tensor.vector_array[i].value_array = malloc((sizeof *temp_tensor.vector_array->value_array) * inner_dim);
        memset(temp_tensor.vector_array[i].value_array, 0, inner_dim*(sizeof *temp_tensor.vector_array[i].value_array));
    }
    return &temp_tensor;
}

int ctorch_floatvector_destroy(FloatVector *input_vector){
    free(input_vector->value_array);
}

int ctorch_floattensor_destroy(FloatTensor *input_tensor){
    for(int i = 0; i < input_tensor->dimensions[0]; i++){
        free(input_tensor->vector_array[i].value_array);
    }
    free(input_tensor->vector_array);
    free(input_tensor->dimensions);
}

void ctorch_floatvector_print(FloatVector *input_vector){
    printf("[");
    for(int i = 0; i < input_vector->length - 1; i++){
        printf("%f,", input_vector->value_array[i]);
    }
    printf("%f]\n", input_vector->value_array[input_vector->length-1]);
}
