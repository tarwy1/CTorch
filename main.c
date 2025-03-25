#include "stdio.h"
#include "ctorch.h"
#include "time.h"
#include "stdlib.h"


int main() {
    srand(time(NULL));
    float vec[100];
    for (int i = 0; i < 100; i++) { vec[i] = 1.0f; }
    NetworkHandle net = *ct_network_create(100, 100);
    
    ct_layer_dense_create(&net, 120, ct_activation_create(""), false);
    ct_layer_dense_create(&net, 100, ct_activation_create(""), false);
    ct_layer_dense_create(&net, 100, ct_activation_create(""), false);
    
    float* vect = ct_network_forward_vector(&net, vec);
    ct_floatvector_print(vect, 100);

    return 0;
}
