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

typedef float (*loss_function)(FloatVector, FloatVector);
typedef FloatVector (*activation_function)(FloatVector);

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

typedef struct Layer{
    int num_nodes;
    bool is_dense;
    NodeHandle *nodes;
    activation_function activation;
} LayerHandle;

typedef struct{
    unsigned int num_layers;
    unsigned int input_dim;
    unsigned int output_dim;
    LayerHandle *layers;
} NetworkHandle;


UintVector* ctorch_uintvector_create(uint length);
FloatVector* ctorch_floatvector_create(uint length);
FloatTensor* ctorch_floattensor_create(uint outer_dim, uint inner_dim);

int ctorch_floatvector_destroy(FloatVector *input_vector);

int ctorch_floattensor_destroy(FloatTensor *input_tensor);

NetworkHandle* ctorch_network_create(uint input_dim, uint output_dim);

LayerHandle* ctorch_layer_dense_create(uint num_nodes, activation_function activation);



/*
floatvector* ctorch_network_forward_vector(floatvector *input_vector);
floattensor* ctorch_network_forward_tensor(floattensor *input_tensor);


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
activation_function* ctorch_network_activation_create(char *activation)
*/
