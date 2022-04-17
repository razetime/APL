#ifndef AVGPOOL
#define AVGPOOL

#include <tensor.h>

void avg_pool(tensor * out, tensor * in_x, int size, int padding, int stride);

#endif