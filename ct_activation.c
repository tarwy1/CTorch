#include <math.h>
#include "ctorch.h"

void ctorch_activation_relu(LayerHandle* layer) {
    for (int i = 0; i < layer->num_nodes; i++) {
        if (layer->nodes[i].activation <= 0.0f) layer->nodes[i].post_activation = 0.0f;
        else layer->nodes[i].post_activation = layer->nodes[i].activation;
    }
}
void ctorch_activation_leakyrelu(LayerHandle* layer) {
    for (int i = 0; i < layer->num_nodes; i++) {
        if (layer->nodes[i].activation <= 0.0f) layer->nodes[i].post_activation = 0.1f * layer->nodes[i].activation;
        else layer->nodes[i].post_activation = layer->nodes[i].activation;
    }
}
void ctorch_activation_sigmoid(LayerHandle* layer) {
    for (int i = 0; i < layer->num_nodes; i++) {
        layer->nodes[i].post_activation = ctorch_util_sigmoid(layer->nodes[i].activation);
    }
}
void ctorch_activation_passthrough(LayerHandle* layer){
    for(int i = 0; i < layer->num_nodes; i++){
        layer->nodes[i].post_activation = layer->nodes[i].activation;
    }
}


activation_function* ctorch_network_activation_create(char* activation) {
    static activation_function Activation;
    if (strcmp(activation, "leakyrelu") == 0) {
        Activation = ctorch_activation_leakyrelu;
    }
    else if (strcmp(activation, "relu") == 0) {
        Activation = ctorch_activation_relu;
    }
    else if (strcmp(activation, "sigmoid") ==0 ){
        Activation = ctorch_activation_sigmoid;
    }
    else{
        Activation = ctorch_activation_passthrough;
    }
    return &Activation;
}