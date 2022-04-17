#ifndef NNINFERENCE
#define NNINFERENCE

#include <add.h>
#include <external.h>
#include <matmul.h>
#include <softmax.h>
#include <variable.h>
#include <min.h>
#include <sigmoid.h>
#include <softmax.h>
#include <conv.h>
#include <relu.h>
#include <avgpool.h>
#include <maxpool.h>
#include <batch_normalization.h>
#include <reshape.h>

int N,C,H,W;
float y;
float bias, epsilon;
int padding, stride, groups, size, axis;
#endif