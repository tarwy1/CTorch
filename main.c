#include "stdio.h"
#include "ctorch.h"
#include "time.h"
#include "stdlib.h"


int main() {
    srand(time(NULL));
    float vec[100];
    for (int i = 0; i < 100; i++) { vec[i] = 1.0f; }
    NetworkHandle net = *ctorch_network_create(100, 100);
    
    ctorch_layer_dense_create(&net, 120, ctorch_network_activation_create(""), false);
    ctorch_layer_dense_create(&net, 100, ctorch_network_activation_create(""), false);
    ctorch_layer_dense_create(&net, 100, ctorch_network_activation_create(""), false);
    
    float* vect = ctorch_network_forward_vector(&net, vec);
    ctorch_floatvector_print(vect, 100);

    return 0;
}
