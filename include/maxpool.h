#ifndef MAXPOOL
#define MAXPOOL

#include <tensor.h>

void max_pool(tensor * out, tensor * in_x, int size, int padding, int stride);

#endif