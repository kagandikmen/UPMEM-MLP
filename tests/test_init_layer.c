#include "mlp.h"
#include "test.h"

int test_init_layer()
{
    LAYER *l = init_layer(3, 4);

    // printf("%d\n", l->num_neurons);
    // printf("%lf\n", l->n[0].w[0]);
    // printf("%lf\n", l->n[1].w[0]);
    // printf("%lf\n", l->n[2].w[0]);
    // printf("%lf\n", l->n[0].lw[0]);
    // printf("%d\n", l->n[0].num_weights);
    // printf("%d\n", l->n[1].num_weights);
    // printf("%d\n", l->n[2].num_weights);

    int test_pass_fail = (l->num_neurons == 3)
                        && (l->n[0].lw[0] == l->n[0].w[0])
                        && (l->n[1].w[0] >= 0)
                        && (l->n[1].w[0] <= 1)
                        && (l->n[2].num_weights == 4);

    return test_pass_fail;
}

int main()
{
    int test_pass_fail = test_init_layer();

    TEST_PASS_FAIL(test_pass_fail)
}