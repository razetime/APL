#ifndef NNBUILT_IN
#define NNBUILT_IN

#include <nnType.h>
#include <assert.h>

#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif



namespace NNEF_RT
{
    tensor<float> c_min(tensor<float> &A, float B);
    tensor<float> c_add_tt(tensor<float> &A, tensor<float> &B);
    tensor<float> c_sub(tensor<float> &A, tensor<float> &B);
    tensor<float> c_div(tensor<float> &A, tensor<float> &B);
    tensor<float> c_select_ttt(tensor<float> &A, tensor<float> &B, tensor<float> &C);
    tensor<float> c_lt_tf(tensor<float> & A, float B);
    tensor<float> c_matmul_ttll(tensor<float> &A, tensor<float> &B, bool trA, bool trB);
    tensor<float> c_external_at(vector<int>& shape, tensor<float> &A);
    tensor<float> c_mul_ft(float B, tensor<float> &A);
    tensor<float> c_mul_tt(tensor<float> &A, tensor<float> &B);
    tensor<float> c_variable_as(vector<int>& shape, string label);
    tensor<float> c_relu_t(tensor<float> & A);
    tensor<float> c_sigmoid_t(tensor<float> & A);
    tensor<float> c_reshape_ta(tensor<float> &A, vector<int>& shape);
    tensor<float> c_transpose_ta(tensor<float> &A, vector<int>& shape);
    tensor<float> c_max_pool_tasaaa(tensor<float> &A, vector<int> &size, string &border, vector<int> &padding, vector<int> &stride, vector<int> &dilation);
	tensor<float> c_conv_with_depthwise(tensor<float> &input, tensor<float> &filter, float bias, string &border, vector<int> &padding, vector<int> &stride, vector<int> &dilatioan, int groups);
    tensor<float> c_conv_tttsaaai(tensor<float> &input, tensor<float> &filter, tensor<float> &bias, string &border, vector<int> &padding, vector<int> &stride, vector<int> &dilatioan, int groups);
    tensor<float> c_conv_ttfsaaai(tensor<float> &input, tensor<float> &filter, float bias, string &border, vector<int> &padding, vector<int> &stride, vector<int> &dilatioan, int groups);
    tensor<float> c_local_response_normalization_tafff(tensor<float> &input, vector<int>& size, float alpha, float beta, float bias);
    tensor<float> c_constant_aa(vector<int>& shape, vector<float>& value);
    tensor<float> c_softmax_ta(tensor<float> &input, vector<int>& axes);
    tensor<float> c_clamp(tensor<float> &input, tensor<float> &A, tensor<float> &B);
    tensor<float> c_clamp(tensor<float> &input, float A, float B);
    tensor<float> c_concat(vector<int>& values, int axis, tensor<float> &A, tensor<float> &B );
    tensor<float> c_mean_reduce(tensor<float> &A, vector<int> &axes);
    //Target CL
    tensor<float> cl_matmul(tensor<float> &A, tensor<float> &B, bool trA, bool trB);
    tensor<float> cl_conv(tensor<float> &input, tensor<float> &filter, tensor<float> &bias, string &border, vector<int> &padding, vector<int> &stride, vector<int> &dilatioan, int groups);
    tensor<float> cl_conv(tensor<float> &input, tensor<float> &filter, float bias, string &border, vector<int> &padding, vector<int> &stride, vector<int> &dilatioan, int groups);
    tensor<float> cl_max_pool(tensor<float> &A, vector<int> &size, string &border, vector<int> &padding, vector<int> &stride, vector<int> &dilation);
    //Target ANN
    tensor<float> ann_matmul_ttll(tensor<float> &A, tensor<float> &B, bool trA, bool trB);
    //Special
    tensor<float> c_conv(tensor<float> &input, tensor<float> &filter, tensor<float> &bias, string &border, vector<int> &padding, vector<int> &stride, vector<int> &dilatioan, int groups);
    tensor<float> c_conv(tensor<float> &input, tensor<float> &filter, float bias, string &border, vector<int> &padding, vector<int> &stride, vector<int> &dilatioan, int groups);
    tensor<float> c_conv(tensor<float> &input, tensor<float> &filter, float bias, vector<int> &padding, vector<int> &stride);
    tensor<float> c_max_pool(tensor<float> &A, vector<int> &size, string &border, vector<int> &padding, vector<int> &stride, vector<int> &dilation);
    tensor<float> c_avg_pool(tensor<float> &A, vector<int> &size, string &border, vector<int> &padding, vector<int> &stride, vector<int> &dilation);
    tensor<float> c_squeeze(tensor<float> &A, vector<int>& axes);
    tensor<float> c_squeeze(tensor<float> &A);
    tensor<float> c_batch_normalization(tensor<float> &input, tensor<float>& mean, tensor<float>& variance, tensor<float>& offset, float scale, float epsilon);
    tensor<float> c_batch_normalization(tensor<float> &input, tensor<float>& mean, tensor<float>& variance, tensor<float>& offset, tensor<float>& scale, float epsilon);

    class C
    {
        public:
            static tensor<float> add (tensor<float> &A, tensor<float> &B); 
            C(void);
            ~C();
    };

    class CL
    {
        public:
            cl_uint status;
            cl_platform_id platform;
            cl_device_id device;
            cl_context context;
            cl_command_queue commandQueue;
            cl_program program;
            cl_kernel kernel;
            bool RunOnce;
            tensor<float> matmul (tensor<float> &A, tensor<float> &B, bool trA, bool trB);
            CL(void);
            ~CL();
    };

    class ANN
    {
        public:
            ANN(void);
            ~ANN();
    };

};
#endif
