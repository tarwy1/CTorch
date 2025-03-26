#include "stdbool.h"

extern float rand_weight_clip;

// struct Node;
// struct Layer;
// typedef struct Node {
//     bool is_input;
//     float activation;
//     float post_activation;
//     float *in_weights;
//     struct Layer* in_layer;
//     float bias;
//     float bias_grad;
//     float* weight_grads;
// } NodeHandle;

typedef struct {
    unsigned int num_layers;
    struct Layer* layers;
    struct Layer* grad_layers;
} ModelHandle;

// typedef float* (*loss_function)(NetworkHandle*, float*);
// typedef void (*activation_function)(struct Layer*);
// typedef void (*optimizer_function)(NetworkHandle*);

typedef struct Layer {
    int type; // 0==dense,1==conv2d

    /*Dense layer data*/
    int num_nodes;
    float* activation_array;
    float* post_activation_array;
    float* bias_array;
    float** weight_array;
} LayerHandle;

ModelHandle* ct_model_create();
void ct_model_destroy(ModelHandle* model);

// void ct_layer_dense_create(NetworkHandle* network, int num_nodes, activation_function* activation, bool is_input);

// void ct_activation_relu(LayerHandle* layer);

// float* ct_loss_mse(NetworkHandle* network, float *labels);
// float* ct_loss_logcosh(NetworkHandle* network, float *labels);

// void ct_optimizer_sgd(NetworkHandle* network);

// activation_function* ct_activation_create(char* activation);

// float* ct_network_forward_vector(NetworkHandle* network, float* input_vector);

// loss_function* ct_network_loss_create(char* loss);

// optimizer_function* ct_optimizer_create(char* optimizer);

// void ct_optimizer_grad();


void ct_floatvector_print(float* input_vector, int size);
void ct_floattensor2d_print(float** input_tensor, int* size);
float ct_util_normal(float mu, float sigma);
float ct_util_sigmoid(float x);
/*

network* ct_network_copy(network* network)

void ct_network_train_tensor(optimizer *optim, loss_function *loss, floattensor *input_tensor, floattensor *label_tensor)

int ct_network_save(network* network, char *path)
network* ct_network_load(char *path)


layer* ct_layer_conv2D_create()

optimizer* ct_optimizer_adam_create(float lr, float B1, float B2)
optimizer* ct_optimizer_sgd_create(float lr)

*/
