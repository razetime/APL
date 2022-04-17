#ifndef NN
#define NN

#include <string>
#include <nnType.h>

extern tensor<float> P[512];
extern tensor<float> nnInput[10];
extern tensor<float> nnOutput[10];
extern string KernelCode;
extern string nnefdirRun;

nnKernel nnCompileProgram(const char * str);
void nnRun(nnKernel &kernel, int type);
void nnRelease(nnKernel &kernel);
#endif
