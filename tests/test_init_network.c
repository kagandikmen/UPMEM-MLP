#include "mlp.h"
#include "test.h"

int test_init_network()
{
    int num_neurons_per_layer[3] = {7, 5, 2};
    NETWORK *n = init_network(7, 3, num_neurons_per_layer);

    // printf("%d\n", n->num_inputs);
    // printf("%d\n", n->num_layers);
    // printf("%d\n", n->l[0].num_neurons);
    // printf("%d\n", n->l[1].num_neurons);
    // printf("%d\n", n->l[2].num_neurons);
    // printf("%d\n", n->l[0].n[0].num_weights);
    // printf("%lf\n", n->l[0].n[0].lw[0]);
    // printf("%d\n", n->l[1].n[0].num_weights);
    // printf("%d\n", n->l[2].n[0].num_weights);

    int pass_fail = (n->num_inputs == 7)
                    && (n->num_layers == 3)
                    && (n->l[0].num_neurons == 7)
                    && (n->l[1].num_neurons == 5)
                    && (n->l[2].num_neurons == 2)
                    && (n->l[0].n[0].num_weights == 8)
                    && (n->l[0].n[0].lw[0] == n->l[0].n[0].w[0])
                    && (n->l[1].n[0].num_weights == 8)
                    && (n->l[2].n[0].num_weights == 6);

    return pass_fail;
}

int main()
{
    int test_pass_fail = test_init_network();
    
    TEST_PASS_FAIL(test_pass_fail)
}