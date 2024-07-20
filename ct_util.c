#include "ct_internal.h"

FloatVector* ctorch_floatvector_alloc(uint length){
    static FloatVector temp_vector;
    temp_vector.length = length;
    temp_vector.value_array = malloc((sizeof *temp_vector.value_array) * length);
    return &temp_vector;
}

FloatTensor* ctorch_floattensor_alloc(uint outer_dim, uint inner_dim){
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