#ifndef BATCHNORMALIZTAION
#define BATCHNORMALIZTAION

#include <tensor.h>

void batch_normalization(tensor * out, tensor * in_x, tensor * mean, tensor * variance, tensor * offset, tensor * scale, float epsilon);

#endif