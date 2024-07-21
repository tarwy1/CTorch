#include "ctorch.h"
#include "stdlib.h"


UintVector* ctorch_uintvector_create(uint length){
    static UintVector temp_vector;
    temp_vector.length = length;
    temp_vector.value_array = malloc((sizeof *temp_vector.value_array) * length);
    return &temp_vector;
}
FloatVector* ctorch_floatvector_create(uint length){
    static FloatVector temp_vector;
    temp_vector.length = length;
    temp_vector.value_array = malloc((sizeof *temp_vector.value_array) * length);
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
