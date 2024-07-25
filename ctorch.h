#define uint unsigned int
#include "stdbool.h"

typedef struct{
    uint length;
    uint *value_array;
} UintVector;
typedef struct{
    uint length;
    float *value_array;
} FloatVector;
typedef struct{
    uint *dimensions;
    FloatVector *vector_array;
} FloatTensor;

struct Node;
struct Layer;
typedef struct Node{
    bool is_input;
    float activation;
    float post_activation; // activation after act. function
    FloatVector *in_weights;
    struct Layer *in_layer;
    float bias;
} NodeHandle;

typedef struct{
    unsigned int num_layers;
    unsigned int input_dim;
    unsigned int output_dim;
    struct Layer *layers;
} NetworkHandle;

typedef float (*loss_function)(NetworkHandle*, FloatVector*);
typedef void (*activation_function)(struct Layer*);

typedef struct Layer{
    int num_nodes;
    bool is_dense;
    NodeHandle *nodes;
    activation_function activation;
} LayerHandle;

UintVector* ctorch_uintvector_create(uint length);
FloatVector* ctorch_floatvector_create(uint length);
FloatTensor* ctorch_floattensor_create(uint outer_dim, uint inner_dim);

int ctorch_floatvector_destroy(FloatVector *input_vector);

int ctorch_floattensor_destroy(FloatTensor *input_tensor);

NetworkHandle* ctorch_network_create(int input_dim, int output_dim);

void ctorch_layer_dense_create(NetworkHandle *network, int num_nodes, activation_function activation, bool is_input);

float ctorch_util_normal(float mu, float sigma);

void ctorch_activation_relu(LayerHandle *layer);

activation_function ctorch_network_activation_create(char *activation);

FloatVector* ctorch_network_forward_vector(NetworkHandle *network, FloatVector *input_vector);

void ctorch_floatvector_print(FloatVector *input_vector);

//FloatTensor* ctorch_network_forward_tensor(FloatTensor *input_tensor);

/*

int ctorch_network_destroy(network* network)   // also destroy layers/loss/optim
network* ctorch_network_copy(network* network)

void ctorch_network_train_tensor(optimizer *optim, loss_function *loss, floattensor *input_tensor, floattensor *label_tensor)

int ctorch_network_save(network* network, char *path)
network* ctorch_network_load(char *path)


layer* ctorch_layer_conv2D_create()

int ctorch_network_layer_add(layer* layer)

optimizer* ctorch_optimizer_adam_create(float lr, float B1, float B2)
optimizer* ctorch_optimizer_sgd_create(float lr)

loss_function* ctorch_network_loss_create(char *loss)

*/
