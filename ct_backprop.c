#include "ctorch.h"
#include "math.h"
#include "string.h"
#include "stdlib.h"

float* ct_loss_mse(NetworkHandle* network, float* labels) {
    static float *loss;
    loss = malloc(network->output_dim * sizeof(float));
    for (int i = 0; i < network->output_dim; i++) {
        loss[i] = powf((network->layers[network->num_layers - 1].nodes[i].post_activation - labels[i]), 2);
    }
    return loss;
}

float* ct_loss_logcosh(NetworkHandle* network, float* labels) {
    static float *loss;
    loss = malloc(network->output_dim * sizeof(float));
    for (int i = 0; i < network->output_dim; i++) {
        loss[i] = logf(coshf((network->layers[network->num_layers - 1].nodes[i].post_activation - labels[i])));
    }
    return loss;
}

loss_function* ct_network_loss_create(char* loss) {
    static loss_function loss_func;
    if (strcmp(loss, "mse") == 0) {
        loss_func = ct_loss_mse;
    }
    else if (strcmp(loss, "logcosh") == 0) {
        loss_func = ct_loss_logcosh;
    }
    return &loss_func;
}

void ct_optimizer_sgd(NetworkHandle* network) {
    /*
    forward prop
    calculate loss
    for i in layers:
        calculate grads

    */
}

optimizer_function* ct_optimizer_create(char* optimizer) {
    static optimizer_function opt_func;
    if (strcmp(optimizer, "sgd") == 0) {
        opt_func = ct_optimizer_sgd;
    }
    return &opt_func;
}
