#include "ctorch.h"
#include "math.h"
#include "string.h"

void ct_activation_relu(LayerHandle* layer) {
    for (int i = 0; i < layer->num_nodes; i++) {
        if (layer->nodes[i].activation <= 0.0f) layer->nodes[i].post_activation = 0.0f;
        else layer->nodes[i].post_activation = layer->nodes[i].activation;
    }
}
void ct_activation_leakyrelu(LayerHandle* layer) {
    for (int i = 0; i < layer->num_nodes; i++) {
        if (layer->nodes[i].activation <= 0.0f) layer->nodes[i].post_activation = 0.1f * layer->nodes[i].activation;
        else layer->nodes[i].post_activation = layer->nodes[i].activation;
    }
}
void ct_activation_sigmoid(LayerHandle* layer) {
    for (int i = 0; i < layer->num_nodes; i++) {
        layer->nodes[i].post_activation = ct_util_sigmoid(layer->nodes[i].activation);
    }
}
void ct_activation_passthrough(LayerHandle* layer){
    for(int i = 0; i < layer->num_nodes; i++){
        layer->nodes[i].post_activation = layer->nodes[i].activation;
    }
}


activation_function* ct_activation_create(char* activation) {
    static activation_function Activation;
    if (strcmp(activation, "leakyrelu") == 0) {
        Activation = ct_activation_leakyrelu;
    }
    else if (strcmp(activation, "relu") == 0) {
        Activation = ct_activation_relu;
    }
    else if (strcmp(activation, "sigmoid") ==0 ){
        Activation = ct_activation_sigmoid;
    }
    else{
        Activation = ct_activation_passthrough;
    }
    return &Activation;
}