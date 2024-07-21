#include "stdio.h"
#include "ctorch.h"

int main(){
    FloatTensor kek = *ctorch_floattensor_create(10, 10);
    for(int i = 0; i < kek.dimensions[0]; i++){
        for(int j = 0; j < kek.dimensions[1]; j++){
            kek.vector_array[i].value_array[j] = (float)i*j;
        }
    }
    for(int i = 0; i < 10; i++){
        for(int j = 0; j < 10; j++){
            printf("%f\n", kek.vector_array[i].value_array[j]);
        }
    }
}
