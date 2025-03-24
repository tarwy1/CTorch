#include "stdio.h"
#include "ctorch.h"
#include "time.h"
#include "stdlib.h"


int main() {
    srand(time(NULL));
    float vec[100];
    for (int i = 0; i < 100; i++) { vec[i] = 0.1f; }
    NetworkHandle net = *ctorch_network_create(100, 100);
    
    ctorch_layer_dense_create(&net, 120, ctorch_network_activation_create("relu"), false);
    ctorch_layer_dense_create(&net, 100, ctorch_network_activation_create("relu"), false);
    ctorch_layer_dense_create(&net, 8, ctorch_network_activation_create("relu"), false);
    ctorch_layer_dense_create(&net, 8, ctorch_network_activation_create("relu"), false);
    ctorch_layer_dense_create(&net, 8, ctorch_network_activation_create("relu"), false);
    ctorch_layer_dense_create(&net, 8, ctorch_network_activation_create("relu"), false);
    ctorch_layer_dense_create(&net, 8, ctorch_network_activation_create("relu"), false);
    ctorch_layer_dense_create(&net, 100, ctorch_network_activation_create("relu"), false);
    
    float* vect = ctorch_network_forward_vector(&net, vec);
    ctorch_floatvector_print(vect, 100);
    //ctorch_floatvector_print(net.layers[1].nodes[1].weight_grads, net.layers[1].nodes[1].in_layer->num_nodes);

    int kek = 0;
    scanf("%d", &kek);
}
