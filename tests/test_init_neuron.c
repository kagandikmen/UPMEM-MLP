#include "mlp.h"
#include "test.h"

int test_init_neuron()
{
    NEURON *n = init_neuron(2);

    // printf("%d\n", n->num_weights);
    // printf("%lf\n", n->w[0]);
    // printf("%lf\n", n->lw[0]);

    return (n->num_weights == 2) && (n->w[0] <= 1) && (n->w[0] >= -1) && (n->lw[0] == n->w[0]);
}

int main()
{
    int test_pass_fail = test_init_neuron();
    
    TEST_PASS_FAIL(test_pass_fail)
}