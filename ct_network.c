#include "ctorch.h"
#include "stdlib.h"
#include "string.h"
#include "stdio.h"
#include "stdbool.h"
#include "math.h"

rand_weight_clip = 2.0f;

void ctorch_activation_relu(LayerHandle* layer) {
    for (int i = 0; i < layer->num_nodes; i++) {
        if (layer->nodes[i].activation <= 0.0f) layer->nodes[i].post_activation = 0.0f;
        layer->nodes[i].post_activation = layer->nodes[i].activation;
    }
}
void ctorch_activation_leakyrelu(LayerHandle* layer) {
    for (int i = 0; i < layer->num_nodes; i++) {
        if (layer->nodes[i].activation <= 0.0f) layer->nodes[i].post_activation = 0.1f * layer->nodes[i].activation;
        layer->nodes[i].post_activation = layer->nodes[i].activation;
    }
}

activation_function ctorch_network_activation_create(char* activation) {
    static activation_function Activation;
    if (strcmp(activation, "leakyrelu") == 0) {
        Activation = ctorch_activation_leakyrelu;
    }
    else if (strcmp(activation, "relu") == 0) {
        Activation = ctorch_activation_relu;
    }
    return Activation;
}

float* ctorch_network_forward_vector(NetworkHandle* network, float* input_vector) {
    static float *output;
    output = malloc(network->output_dim * sizeof(float));
    if (network->layers[0].is_dense) {
        for (int i = 0; i < network->input_dim; i++) {
            network->layers[0].nodes[i].activation = input_vector[i] + network->layers[0].nodes[i].bias;
            network->layers[0].activation(&network->layers[0]);
        }
    }
    for (int i = 1; i < network->num_layers; i++) {
        for (int j = 0; j < network->layers[i].num_nodes; j++) {
            for (int k = 0; k < network->layers[i-1].num_nodes; k++) {
                network->layers[i].nodes[j].activation += network->layers[i - 1].nodes[k].post_activation * network->layers[i].nodes[j].in_weights[k];
                //printf("%d %d %d %f %f\n", i, j, k, network->layers[i].nodes[j].in_weights[k], network->layers[i].nodes[j].activation);
            }
            network->layers[i].nodes[j].activation += network->layers[i].nodes[j].bias;
        }
        network->layers[i].activation(&network->layers[i]);
    }
    for (int i = 0; i < network->output_dim; i++) {
        output[i] = network->layers[network->num_layers - 1].nodes[i].post_activation;
    }
    return output;
}

NetworkHandle* ctorch_network_create(int input_dim, int output_dim) {
    static NetworkHandle Network;
    Network.num_layers = 0;
    Network.input_dim = input_dim;
    Network.output_dim = output_dim;
    Network.layers = malloc(sizeof(LayerHandle) * 2);
    ctorch_layer_dense_create(&Network, input_dim, ctorch_network_activation_create("relu"), true);
    ctorch_layer_dense_create(&Network, output_dim, ctorch_network_activation_create("relu"), false);
    return &Network;
}

void ctorch_layer_dense_create(NetworkHandle* network, int num_nodes, activation_function activation, bool is_input) {
    static struct Layer layer;
    layer.is_dense = true;
    layer.num_nodes = num_nodes;
    layer.activation = activation;
    layer.nodes = malloc(sizeof(NodeHandle) * num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        layer.nodes[i].activation = 0.0f;
        layer.nodes[i].post_activation = 0.0f;
        layer.nodes[i].bias = 0.0f;
        layer.nodes[i].bias_grad = 0.0f;
        layer.nodes[i].is_input = is_input;
        if (!is_input) {
            //layer.nodes[i].in_weights = malloc(network->layers[network->num_layers - 1].num_nodes * sizeof(float));
            //layer.nodes[i].weight_grads = malloc(network->layers[network->num_layers - 1].num_nodes * sizeof(float));
            //memset(layer.nodes[i].in_weights, 0.0f, network->layers[network->num_layers - 1].num_nodes * sizeof(float));
            //memset(layer.nodes[i].weight_grads, 0.0f, network->layers[network->num_layers - 1].num_nodes * sizeof(float));
            if (network->num_layers >= 2) {
                layer.nodes[i].in_weights = malloc(network->layers[network->num_layers - 2].num_nodes * sizeof(float));
                layer.nodes[i].weight_grads = malloc(network->layers[network->num_layers - 2].num_nodes * sizeof(float));
                layer.nodes[i].in_layer = &network->layers[network->num_layers - 2];
                for (int j = 0; j < network->layers[network->num_layers - 2].num_nodes; j++) {
                    layer.nodes[i].in_weights[j] = ctorch_util_normal(0.0f, 2 / (float)network->layers[network->num_layers - 2].num_nodes);
                }
            }
            else {
                layer.nodes[i].in_weights = malloc(network->layers[network->num_layers - 1].num_nodes * sizeof(float));
                layer.nodes[i].weight_grads = malloc(network->layers[network->num_layers - 1].num_nodes * sizeof(float));
                layer.nodes[i].in_layer = &network->layers[network->num_layers - 1];
                for (int j = 0; j < network->layers[network->num_layers - 1].num_nodes; j++) {
                    layer.nodes[i].in_weights[j] = ctorch_util_normal(0.0f, 2 / (float)network->layers[network->num_layers - 1].num_nodes);
                }
            }
            
            
        }
        else {
            layer.nodes[i].in_weights = malloc(sizeof(float));
            layer.nodes[i].weight_grads = malloc(sizeof(float));
        }
    }
    if (network->num_layers != 0) {
        LayerHandle* ptr;
        ptr = realloc(network->layers, sizeof(LayerHandle) * (network->num_layers + 1));
        if (ptr)
            network->layers = ptr;
    }
    if (network->num_layers >= 2) {
        network->layers[network->num_layers] = network->layers[network->num_layers - 1];
        //memcpy(&network->layers[network->num_layers], &network->layers[network->num_layers - 1], sizeof(LayerHandle));
        //network->layers[network->num_layers] = network->layers[network->num_layers - 1];
        for (int i = 0; i < network->layers[network->num_layers].num_nodes; i++) {
            network->layers[network->num_layers].nodes[i].in_layer = &layer;
            network->layers[network->num_layers].nodes[i].in_weights = realloc(network->layers[network->num_layers].nodes[i].in_weights, layer.num_nodes * sizeof(float));
            network->layers[network->num_layers].nodes[i].weight_grads = realloc(network->layers[network->num_layers].nodes[i].in_weights, layer.num_nodes * sizeof(float));
            
            memset(network->layers[network->num_layers].nodes[i].in_weights, 0, layer.num_nodes * sizeof(float));
            memset(network->layers[network->num_layers].nodes[i].weight_grads, 0, layer.num_nodes * sizeof(float));
            for (int j = 0; j < layer.num_nodes; j++) {
                network->layers[network->num_layers].nodes[i].in_weights[j] = ctorch_util_normal(0.0f, 2.0f / (float)(layer.num_nodes));
            }
        }
        network->layers[network->num_layers - 1] = layer;

    }
    else {
        network->layers[network->num_layers] = layer;
    }
    network->num_layers += 1;
}


