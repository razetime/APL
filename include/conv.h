#ifndef CONV
#define CONV

#include <tensor.h>

void conv(tensor * out, tensor * in_x, tensor * filter, tensor * bias, int padding, int stride, int groups);
void convxbias(tensor * out, tensor * in_x, tensor * filter, float bias, int padding, int stride, int groups);

#endif