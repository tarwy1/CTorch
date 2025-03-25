#include "stdbool.h"

extern float rand_weight_clip;

struct Node;
struct Layer;
typedef struct Node {
    bool is_input;
    float activation;
    float post_activation;
    float *in_weights;
    struct Layer* in_layer;
    float bias;
    float bias_grad;
    float* weight_grads;
} NodeHandle;

typedef struct {
    unsigned int num_layers;
    unsigned int input_dim;
    unsigned int output_dim;
    struct Layer* layers;
} NetworkHandle;

typedef float* (*loss_function)(NetworkHandle*, float*);
typedef void (*activation_function)(struct Layer*);
typedef void (*optimizer_function)(NetworkHandle*);


typedef struct Layer {
    int num_nodes;
    bool is_dense;
    NodeHandle* nodes;
    activation_function activation;
} LayerHandle;

NetworkHandle* ct_network_create(int input_dim, int output_dim);

void ct_layer_dense_create(NetworkHandle* network, int num_nodes, activation_function* activation, bool is_input);

float ct_util_normal(float mu, float sigma);
float ct_util_sigmoid(float x);

void ct_activation_relu(LayerHandle* layer);

float* ct_loss_mse(NetworkHandle* network, float *labels);
float* ct_loss_logcosh(NetworkHandle* network, float *labels);

void ct_optimizer_sgd(NetworkHandle* network);

activation_function* ct_activation_create(char* activation);

float* ct_network_forward_vector(NetworkHandle* network, float* input_vector);

void ct_floatvector_print(float* input_vector, int size);

loss_function* ct_network_loss_create(char* loss);

optimizer_function* ct_optimizer_create(char* optimizer);

void ct_optimizer_grad();

/*

network* ct_network_copy(network* network)

void ct_network_train_tensor(optimizer *optim, loss_function *loss, floattensor *input_tensor, floattensor *label_tensor)

int ct_network_save(network* network, char *path)
network* ct_network_load(char *path)


layer* ct_layer_conv2D_create()

optimizer* ct_optimizer_adam_create(float lr, float B1, float B2)
optimizer* ct_optimizer_sgd_create(float lr)

*/
