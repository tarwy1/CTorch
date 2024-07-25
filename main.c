#include "stdio.h"
#include "ctorch.h"
#include "time.h"
#include "stdlib.h"

int main(){
    srand ( time(NULL) );
    
    NetworkHandle net = *ctorch_network_create(4, 2);
    FloatVector vec = *ctorch_floatvector_create(4);
    for(int i = 0; i < 4; i++){vec.value_array[i] = i;}
    FloatVector *vect = ctorch_network_forward_vector(&net, &vec);
    ctorch_floatvector_print(vect);
}
