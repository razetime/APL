#ifndef NNREGFUN
#define NNREGFUN

#include "nnBuilt-in.h"
#include <cstring>
#include <time.h>

namespace NNEF_RT
{ 
    //Register target
    //vector <int> shape;
    //vector <int> perm;
    extern NNEF_RT::C C; 
    //extern NNEF_RT::CL CL;
    extern NNEF_RT::ANN ANN;

    // Target

    // C

    C::C(void)
    {
        cout << "Create C Target ...ok" << endl; 
    }

    C::~C()
    {
        cout << "Release C Target ...ok" << endl; 
    }

    // Init CL, it just need run one time
    CL::CL(void)
    {
        RunOnce = false;
        status = clGetPlatformIDs(1, &platform, NULL);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
        commandQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);
        const char * filename = "kernel.cl";
        string sourceStr;
        std::ifstream IF(filename);
        std::string line;
        std::string content;
        while(!IF.eof())
        {
            std::getline(IF, line);
            content += (line + '\n');
        }
        const char * source = content.c_str();
        size_t sourceSize[] = { strlen(source) };
        program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);
        status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);   
        kernel = clCreateKernel(program, "matrix_mult", NULL);
        cout << "Create CL Target ...ok" << endl;
    }

    CL::~CL()
    {
        cout << "Release CL Target ...ok" << endl; 
    }

    tensor<float> CL::matmul (tensor<float> &NN_A, tensor<float> &NN_B, bool trA, bool trB)
    {
        if (!RunOnce)
        {
            size_t cb = 256;
            string devname;
            devname.resize(cb);
            clGetPlatformInfo(platform, CL_PLATFORM_VERSION, cb, &devname[0], NULL);
            cout << devname << endl;
            clGetDeviceInfo(device, CL_DEVICE_NAME, cb, &devname[0], 0);
            cout << "Device: " << devname;
            clGetPlatformInfo(platform, CL_PLATFORM_NAME, cb, &devname[0], NULL);
            cout << ", " << devname;
            clGetDeviceInfo(device, CL_DEVICE_VENDOR, cb, &devname[0], NULL);
            cout << ", " << devname << endl;  
            RunOnce = true;        
        }
        const int Ndim = NN_A.shape[0];
        const int Mdim = NN_B.shape[1];
        const int Pdim = NN_A.shape[1];;
        int szA = Ndim * Pdim;
        int szB = Pdim * Mdim;
        int szC = Ndim * Mdim;

        //Run
        tensor<float> out;
        //Set shape
        out.shape = NN_A.shape;
        //Set size
        out.D.resize(NN_A.D.size());

        out.D.resize(szC);
        out.shape.resize(2);
        out.shape[0] = NN_A.shape[0];
        out.shape[1] = NN_B.shape[1]; 

        cl_mem memObjects[3] = { 0, 0, 0 };
        memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY |  CL_MEM_COPY_HOST_PTR,
                sizeof(float)* szA, &NN_A.D[0], NULL);
        memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY |  CL_MEM_COPY_HOST_PTR,
                sizeof(float)* szB, &NN_B.D[0], NULL);
        memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                sizeof(float)* szC, &out.D[0], NULL);
        if (memObjects[0] == NULL || memObjects[1] == NULL ||memObjects[2] == NULL) 
            perror("Error in clCreateBuffer.\n");


        if (status)
            cout << status << "  !!!!!!!!" <<endl;


        status = clSetKernelArg(kernel, 0, sizeof(int), &Ndim);
        status = clSetKernelArg(kernel, 1, sizeof(int), &Mdim);
        status = clSetKernelArg(kernel, 2, sizeof(int), &Pdim);
        status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjects[0]);
        status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjects[1]);
        status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &memObjects[2]);
        if (status)
            cout << "SetKernel error" << endl;

        size_t global[2];
        cl_event prof_event;
        //cl_ulong ev_start_time = (cl_ulong)0;
        //cl_ulong ev_end_time = (cl_ulong)0;
        //double rum_time;
        global[0] = (size_t)Ndim;
        global[1] = (size_t)Mdim;
        status = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
                global, NULL, 0, NULL, &prof_event);
        if (status)
            cout << "NDrange error" << endl;
        clFinish(commandQueue);

        //status = clGetEventProfilingInfo(prof_event,CL_PROFILING_COMMAND_QUEUED,sizeof(cl_ulong),&ev_start_time,NULL);
        //status = clGetEventProfilingInfo(prof_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&ev_end_time,NULL);
        if (status) 
            perror("Read error\n");
        //rum_time = (double)(ev_end_time - ev_start_time);

        status = clEnqueueReadBuffer(commandQueue, memObjects[2],CL_TRUE, 0,
                sizeof(float)* szC, &out.D[0],0, NULL, NULL);
        if (status) 
            perror("Read error\n");

        return out;
    }

    // ANN
    ANN::ANN(void)
    {
        cout << "Create ANN Target ...ok" << endl; 
    }

    ANN::~ANN()
    {
        cout << "Release ANN Target ...ok" << endl; 
    }

    // Entry point of function call 

	static tensor<float> min(tensor<float> &A, float B)
	{
		return c_min(A,B);
	}
	
	static tensor<float> mul(tensor<float> &A, tensor<float> &B)
    {
        return c_mul_tt(A, B);
    }

    static tensor<float> mul(float B, tensor<float> &A)
    {
        return c_mul_ft(B, A);
    }

    static tensor<float> add(tensor<float> &A, tensor<float> &B)
    {
        //return C.add(A, B);
        return c_add_tt(A, B);
    }

    static tensor<float> sub(tensor<float> &A, tensor<float> &B)
    {
        return c_sub(A, B);
    }

    static tensor<float> div(tensor<float> &A, tensor<float> &B)
    {
        return c_div(A, B);
    }

    static tensor<float> external(vector<int>& shape, tensor<float> &A)
    {
        return  c_external_at(shape, A);        
    }

    static tensor<float> variable(vector<int>& shape, string label)
    {
        return c_variable_as(shape, label);
    }

    static tensor<float> matmul(tensor<float> &A, tensor<float> &B, bool trA, bool trB)
    {
        //return CL.matmul(A, B, trA, trB);
        return  c_matmul_ttll(A, B, trA, trB);
        //return  cl_matmul(A, B, trA, trB);
    }

    static tensor<float> sigmoid(tensor<float> & A)
    {
        return c_sigmoid_t(A);
    }

    static tensor<float> relu(tensor<float> & A)
    {
        return c_relu_t(A);
    }

    static tensor<float> lt(tensor<float> & A, float B)
    {
        return c_lt_tf(A, B);
    }

    static tensor<float> select(tensor<float> &A, tensor<float> &B, tensor<float> &C)
    {
        assert(select);
        return c_select_ttt(A, B, C);
    }

    static tensor<float> reshape(tensor<float> &A, vector<int>& shape)
    {
        assert(c_reshape_ta);
        return c_reshape_ta(A, shape);
    }

    static tensor<float> squeeze(tensor<float> &A, vector<int>& axes)
    {
        return c_squeeze(A, axes);
    }

    static tensor<float> squeeze(tensor<float> &A)
    {
        return c_squeeze(A);
    }

    static tensor<float> transpose(tensor<float> &A, vector<int>& shape)
    {
        assert(c_transpose_ta);
        return c_transpose_ta(A, shape);
    }

    static tensor<float> max_pool(tensor<float> &A, vector<int> &size, string &border, vector<int> &padding, vector<int> &stride, vector<int> &dilation)
    {
        assert(c_max_pool_tasaaa);
        //return c_max_pool_tasaaa(A, size, border, padding, stride, dilation);
        return c_max_pool(A, size, border, padding, stride, dilation);
        //return cl_max_pool(A, size, border, padding, stride, dilation);
    }

    static tensor<float> avg_pool(tensor<float> &A, vector<int> &size, string &border, vector<int> &padding, vector<int> &stride, vector<int> &dilation)
    {
        return c_avg_pool(A, size, border, padding, stride, dilation);
    }

    static tensor<float> conv(tensor<float> &input, tensor<float> &filter, tensor<float> &bias, string &border, vector<int> &padding, vector<int> &stride, vector<int> &dilatioan, int groups)
    {
        //assert(c_conv_tttsaaai);
        //return c_conv_tttsaaai(input, filter, bias, border, padding, stride, dilatioan, groups);
        return c_conv(input, filter, bias, border, padding, stride, dilatioan, groups);
        //return cl_conv(input, filter, bias, border, padding, stride, dilatioan, groups);
    }

    static tensor<float> conv(tensor<float> &input, tensor<float> &filter, float bias, vector<int> &padding, vector<int> &stride)
    {
        return c_conv(input, filter, bias, padding, stride);
    }

    static tensor<float> conv(tensor<float> &input, tensor<float> &filter, float bias, string &border, vector<int> &padding, vector<int> &stride, vector<int> &dilatioan, int groups)
    {
        //assert(c_conv_ttfsaaai);
        //return c_conv_ttfsaaai(input, filter, bias, border, padding, stride, dilatioan, groups);

        //assert(c_conv_with_depthwise);
        //return c_conv_with_depthwise(input, filter, bias, border, padding, stride, dilatioan, groups);
        return c_conv(input, filter, bias, border, padding, stride, dilatioan, groups);

    }

    static tensor<float> local_response_normalization(tensor<float> &input, vector<int>& size, float alpha, float beta, float bias)
    {
        assert(c_local_response_normalization_tafff);
        return c_local_response_normalization_tafff(input, size, alpha, beta, bias);
    }

    static tensor<float> batch_normalization(tensor<float> &input, tensor<float>& mean, tensor<float>& variance, tensor<float>& offset, float scale, float epsilon)
    {
        return c_batch_normalization(input, mean, variance, offset, scale, epsilon);
    }

    static tensor<float> batch_normalization(tensor<float> &input, tensor<float>& mean, tensor<float>& variance, tensor<float>& offset, tensor<float>& scale, float epsilon)
    {
        return c_batch_normalization(input, mean, variance, offset, scale, epsilon);
    }

    static tensor<float> constant(vector<int>& shape, vector<float>& value)
    {
        return c_constant_aa(shape, value);
    }

    static tensor<float> softmax(tensor<float> &input, vector<int>& axes)
    {
        assert(c_softmax_ta);
        return c_softmax_ta(input, axes);
    }

    static tensor<float> clamp(tensor<float> &input, tensor<float> &A, tensor<float> &B)
    {
        return c_clamp(input, A, B);
    }

    static tensor<float> clamp(tensor<float> &input, float A, float B)
    {
        return c_clamp(input, A, B);
    }

    static tensor<float> concat(vector<int>& values, int axis, tensor<float> &A, tensor<float> &B )
    {
        return c_concat(values, axis, A, B);
    }

    static tensor<float> mean_reduce(tensor<float> &A, vector<int>& axes)
    {
        return c_mean_reduce(A, axes);
    }
   // -----------------------------
};

using namespace NNEF_RT;
#endif
