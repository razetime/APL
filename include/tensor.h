#ifndef TENSOR
#define TENSOR

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>  //max_pool
#include <time.h>
#include <stdint.h>

struct tensor
{
    int n;
    int c;
    int h;
    int w;
    int dim;
    int size;
    float * data;
};

typedef struct tensor tensor;

int where_pos2(tensor *, int, int);
int where_pos4(tensor *, int, int, int, int);
tensor * make_tensor(tensor *, int, int, int, int);
float im2col_get_pixel(tensor * im, int height, int width, int channels, int row, int col, int channel, int pad);

void variable(tensor * out, int n, int c, int h, int w, const char * label);
void external(tensor * out, int n, int c, int h, int w);
void matmul(tensor * out, tensor * in_x, tensor * in_y);
void matmul_ft(tensor * out, tensor * in_x, tensor * in_y);
void mul(tensor * out, tensor * in_x, float value);
void add(tensor * out, tensor * in_x, tensor * in_y);
void softmax(tensor * out, tensor * in_x);
void reshape(tensor * out, tensor * in_x, int n, int c, int h, int w);
void squeeze(tensor * out, tensor * in_x, int n, int c, int h, int w);
void transpose(tensor * out, tensor * in_x, int n, int c, int h, int w);
void concat(tensor * out, tensor * in_x, tensor * in_y, int axis);
void conv(tensor * out, tensor * in_x, tensor * filter, tensor * bias, int padding, int stride, int groups);
void max_pool(tensor * out, tensor * in_x, int size, int padding, int stride);
void relu(tensor * out, tensor * in_x);
void relu6(tensor * out, tensor * in_x);
void sigmoid(tensor * out, tensor * in_x);
void convxbias(tensor * out, tensor * in_x, tensor * filter, float bias, int padding, int stride, int groups);
void convxt_bias(tensor * out, tensor * in_x, tensor * filter, tensor * bias, int padding, int stride, int groups);
void batch_normalization(tensor * out, tensor * in_x, tensor * mean, tensor * variance, tensor * offset, tensor * scale, float epsilon);
void avg_pool(tensor * out, tensor * in_x, int size, int padding, int stride);
void min(tensor * out, tensor * in_x, float y);
void sigmoid(tensor * out, tensor * in_x);
void bn_sqrt(tensor * out, tensor * in_x, float epsilon);
void bn_sub(tensor * out, tensor * in_x, tensor * in_y);
void bn_div(tensor * out, tensor * in_x, tensor * in_y);
void bn_mul(tensor * out, tensor * in_x, tensor * in_y);
void bn_add(tensor * out, tensor * in_x, tensor * in_y);
#endif
