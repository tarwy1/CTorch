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

NetworkHandle* ctorch_network_create(int input_dim, int output_dim);

void ctorch_layer_dense_create(NetworkHandle* network, int num_nodes, activation_function* activation, bool is_input);

float ctorch_util_normal(float mu, float sigma);
float ctorch_util_sigmoid(float x);

void ctorch_activation_relu(LayerHandle* layer);

float* ctorch_loss_mse(NetworkHandle* network, float *labels);
float* ctorch_loss_logcosh(NetworkHandle* network, float *labels);

void ctorch_optimizer_sgd(NetworkHandle* network);

activation_function* ctorch_network_activation_create(char* activation);

float* ctorch_network_forward_vector(NetworkHandle* network, float* input_vector);

void ctorch_floatvector_print(float* input_vector, int size);

loss_function* ctorch_network_loss_create(char* loss);

optimizer_function* ctorch_optimizer_create(char* optimizer);

void ctorch_optimizer_grad();

/*

network* ctorch_network_copy(network* network)

void ctorch_network_train_tensor(optimizer *optim, loss_function *loss, floattensor *input_tensor, floattensor *label_tensor)

int ctorch_network_save(network* network, char *path)
network* ctorch_network_load(char *path)


layer* ctorch_layer_conv2D_create()

optimizer* ctorch_optimizer_adam_create(float lr, float B1, float B2)
optimizer* ctorch_optimizer_sgd_create(float lr)

*/
