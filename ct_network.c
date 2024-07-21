#include "ctorch.h"
#include "stdlib.h"

NetworkHandle* ctorch_network_create(uint input_dim, uint output_dim){

}

LayerHandle* ctorch_layer_dense_create(NetworkHandle *network, uint num_nodes, activation_function activation){
    LayerHandle layer;
    layer.is_dense = true;
    layer.num_nodes = num_nodes;
    layer.activation = activation;
    NodeHandle *nodes;
    nodes = malloc((sizeof *nodes) * num_nodes);
    layer.nodes = nodes;
    for(int i = 0; i < num_nodes; i++){
        layer.nodes[i].activation = 0.0f;
        layer.nodes[i].post_activation = 0.0f;
        layer.nodes[i].bias = 0.0f;
        layer.nodes[i].is_input = false;
        layer.nodes[i].in_weights = ctorch_floatvector_create(10);
    }
}