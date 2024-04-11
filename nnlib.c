#define CL_TARGET_OPENCL_VERSION 300
extern "C" {
#include <CL/opencl.h>
}
// #include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <float.h>  //max_pool
#include <iostream>

#define DEBUG_TIME
#define im2colxGEMM

#define INSTEPS (512*512*512)
#define ITERS (262144)
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

tensor T[512];
//int N,C,H,W;
//float y;
//float bias, epsilon;
//int padding, stride, groups, size;

const char *getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

void cherr(cl_int err) {
    if(err != CL_SUCCESS)
      std::cerr << getErrorString(err) << std::endl;
}


float im2col_get_pixel(tensor * im, int height, int width, int channels,
        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
            row >= height || col >= width) return 0;

    return im->data[col + width * (row + height * channel)];
}

int where_pos2(tensor * out, int H, int W) 
{
    return H * out->w + W;
}

int where_pos4(tensor * out, int N, int C, int H, int W) 
{
    int hXw = out->h * out->w;
    return N * out->c * hXw + C * hXw + H * out->w + W;
}

tensor * make_tensor(tensor * out, int n, int c, int h, int w) 
{

    out->n = n;
    out->c = c;
    out->h = h;
    out->w = w;
    out->dim = 4;

    if (n == 0)
    {
        out->n = 1;
        out->dim = 3;
    }

    if (c == 0)
    {
        out->c = 1;
        out->dim = 2;
    }

    if (h == 0)
    {
        out->h = 1;
        out->dim = 1;
    }

    if (w == 0)
    {
        out->w = 1;
        out->dim = 0;
    }
    #define MALLOC_ALIGN    16
    out->size = out->n * out->c * out->h * out->w;
    if (!out->data) posix_memalign((void **)(&(out->data)), MALLOC_ALIGN, out->size  * sizeof(float));
    //if (!out->data) out->data = (float * ) calloc(out->size, sizeof(float));
    return out;
}

void variable(tensor * out, int n, int c, int h, int w, const char * label)
{
	FILE * pFile;
	out = make_tensor(out, n, c, h, w);

	char * file_name = (char *) calloc(strlen(label) + strlen(".dat") + 1, sizeof(char));
	sprintf(file_name, "%s%s", label, ".dat");
    pFile = fopen ( file_name , "rb" );

    /*
        string file_name(label);
        file_name = file_name + ".dat";
        pFile = fopen ( file_name.c_str() , "rb" );
    */
	if (pFile==NULL) {fputs ("File error\n",stderr); assert (0);}

	fseek(pFile, 0, SEEK_END);
	int file_size = ftell(pFile);
	fseek(pFile, 0, SEEK_SET);
	//printf("file_size %d\n", file_size);

	char metadata[4];
	fread((char *)metadata, sizeof(unsigned char), 4, pFile);
	unsigned char magic1 = metadata[0];
	unsigned char magic2 = metadata[1];
	unsigned char major  = metadata[2];
	unsigned char minor  = metadata[3];
	//printf("magic1 : %c\n", magic1);

	uint32_t data_length;
	fread((char *)&data_length, sizeof(uint32_t), 1, pFile);
	uint32_t rank;
	fread((char *)&rank, sizeof(uint32_t), 1, pFile);
	uint32_t ranks[4];
	uint32_t *shape = ranks;
	//printf("data_length : %d\n", data_length);
	//printf("rank : %d\n", rank);

	int checkVersion = 1;

	for (int i = 0; i < rank; ++i)
	{
		uint32_t *p = shape + i;
		fread((char *)p, sizeof(uint32_t), 1, pFile);
		checkVersion = checkVersion * (*p);
	}
	//printf("checkVersion : %d\n", checkVersion);

	int header_size = 128;
	int version = 0;
	if (file_size == header_size + data_length && (checkVersion * 4) == data_length)
	{
		assert((file_size - header_size) == (checkVersion * 4));
		//printf("Ver 2.0 - 60ba79d\n");
		version = 2;
	}
	else
	{
		//printf("Ver 1.0 - 02a3916\n");
		version = 1;
	}

	uint8_t code, bits;
	fread((char *)&code, sizeof(uint8_t), 1, pFile);
	fread((char *)&bits, sizeof(uint8_t), 1, pFile);
	uint16_t qlen;
	fread((char *)&qlen, sizeof(uint16_t), 1, pFile);

	uint64_t quantization;
	fread((char *)&quantization, sizeof(uint64_t), qlen, pFile);
	size_t count = rank ? 1 : 0;
	for (int i = 0; i < rank; ++i)
		count *= shape[i];

	//printf("count : %ld]\n", count);

	//NNEF store file format Version 2
	if (version == 2)
	{
		fseek(pFile, header_size, SEEK_SET);
		int result = fread (out->data, sizeof(float), out->size, pFile);
		if (result != out->size) {fputs ("Reading error\n",stderr); assert (0);}
	}
	else if (version == 1)
	{
		int result = fread(out->data, sizeof(float), count, pFile);
		if (result != count) {fputs ("Reading error\n",stderr); assert (0);}
	}
	/*
	   printf("file_name %s\n", file_name);
	   printf("out->n : %d\n", out->n);
	   printf("out->c : %d\n", out->c);
	   printf("out->h : %d\n", out->h);
	   printf("out->w : %d\n", out->w);
	   printf("out->dim : %d\n", out->dim);
	   printf("out->size : %d\n", out->size);
	   printf("out->data[0] : %g\n", out->data[0]);
	 */
	fclose(pFile);
}

void external(tensor * out, int n, int c, int h, int w)
{
    out = make_tensor(out, n, c, h, w);

    printf ("out->n : %d\n", out->n);
    printf ("out->c : %d\n", out->c);
    printf ("out->h : %d\n", out->h);
    printf ("out->w : %d\n", out->w);
    printf ("out->dim : %d\n", out->dim);
    printf ("out->size : %d\n", out->size);
}


const char* mmul_buf = "\n" \
"__kernel void matmul(const int M, const int N, const int K,\n" \
"                      const __global float* A,\n" \
"                      const __global float* B,\n" \
"                      __global float* C) {\n" \
"    \n" \
"    const int globalRow = get_global_id(0); // Row ID of C (0..M)\n" \
"    const int globalCol = get_global_id(1); // Col ID of C (0..N)\n" \
" \n" \
"    float acc = 0.0f;\n" \
"    for (int k=0; k<K; k++) {\n" \
"        acc += A[k*M + globalRow] * B[globalCol*K + k];\n" \
"    }\n" \
" \n" \
"    C[globalCol*M + globalRow] = acc;\n" \
"}\n" \
"\n";
void matmul(tensor * out, tensor * in_x, tensor * in_y)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif
    // // [m*p][p*m] = [m*n]
    // for (int i=0; i < m; i++)
    // {
    //     for (int j=0; j < n; j++)
    //     {
    //         float sum = 0.0;
    //         for(int k = 0; k < p; k++)
    //         {
    //             sum += in_x->data[where_pos2(in_x, i, k)] * in_y->data[where_pos2(in_y, k, j)];
    //         }
    //         out->data[where_pos2(out, i, j)] = sum ;
    //     }
    // }

    int m = in_x->h; 
    int p = in_x->w;
    int n = in_y->w;
    out = make_tensor(out, 0, 0, m, n);

    cl_mem d_x;
    cl_mem d_y;
    cl_mem d_out;
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
//    unsigned int n = 100000;
    //size_t bytes = x->size*sizeof(float);
    int nsteps = INSTEPS;
    int niters = ITERS;
    size_t globalSize, localSize;
    cl_int err;
    localSize = 64;
    globalSize = nsteps/niters;
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    cherr(err);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    program = clCreateProgramWithSource(context, 1,
                            (const char **) &mmul_buf, NULL, &err);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "matmul", &err);
    d_x = clCreateBuffer(context, CL_MEM_READ_ONLY, in_x->size*sizeof(float), NULL, NULL);
    d_y = clCreateBuffer(context, CL_MEM_READ_ONLY, in_y->size*sizeof(float), NULL, NULL);
    d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, out->size*sizeof(float), NULL, NULL);
    err = clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0,
                                   in_x->size*sizeof(float), in_x->data, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_y, CL_TRUE, 0,
                                   in_y->size*sizeof(float), in_y->data, 0, NULL, NULL);
                                   
    err  = clSetKernelArg(kernel, 0, sizeof(int), &m);
    err  = clSetKernelArg(kernel, 1, sizeof(int), &n);
    err  = clSetKernelArg(kernel, 2, sizeof(int), &p);
    err  = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_x);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_y);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_out);
 
    // Execute the kernel over the entire range of the data set
    const int TS = 32;
  const size_t local[2] = { TS, TS };
  const size_t global[2] = { m, n };
    //err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local,
    //                                                          0, NULL, NULL);
 
    // Wait for the command queue to get serviced before reading back results
    // clFinish(queue);


    // Read the results from the device
    //clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
    //                            out->size*sizeof(float), out->data, 0, NULL, NULL );



#ifdef DEBUG_TIME
        double end = clock();
        printf("[matmul time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void matmul_ft(tensor * out, tensor * in_x, tensor * in_y)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    int m = in_x->h;
    int p = in_x->w;
    int n = in_y->h;
    out = make_tensor(out, 0, 0, m, n);

    // [m*p][p*m] = [m*n]
    for (int i=0; i < m; i++)
    {
        for (int j=0; j < n; j++)
        {
            float sum = 0.0;
            for(int k = 0; k < p; k++)
            {
                sum += in_x->data[where_pos2(in_x, i, k)] * in_y->data[where_pos2(in_y, j, k)];
            }
            out->data[where_pos2(out, i, j)] = sum ;
        }
    }

#ifdef DEBUG_TIME
        double end = clock();
        printf("[matmul time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void mul(tensor * out, tensor * in_x, float value)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif
    out = make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);
    out->dim = in_x->dim;
    for (int i = 0; i < in_x->size; i++)
	out->data[i] = in_x->data[i] * value;

#ifdef DEBUG_TIME
        double end = clock();
        printf("[mul time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}


// https://www.olcf.ornl.gov/tutorials/opencl-vector-addition/
const char* add_pbuf = "\n" \
"kernel void tensor_add(__global float* a, __global float* b, __global float* c,\n" \
"                       const unsigned int n) {\n" \
"    int id = get_global_id(0);\n" \
"    if(id<n)\n" \
"        c[id] = a[id] + b[id];\n" \
"}\n" \
"\n";
void cl_add(tensor* out, tensor* x, tensor* y) {
    out = make_tensor(out, x->n, x->c, x->h, x->w);

    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;
 
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel

    unsigned int n = 100000;
    size_t bytes = x->size*sizeof(float);

    int nsteps = INSTEPS;
    int niters = ITERS;
    size_t globalSize, localSize;
    cl_int err;

    // Number of work items in each local work group
    localSize = 64;
 
    globalSize = nsteps/niters;

    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
 
    // Create a context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
 
    // Create a command queue
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);

    program = clCreateProgramWithSource(context, 1,
                            (const char **) &add_pbuf, NULL, &err);
 
    // Build the program executable
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
 
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "tensor_add", &err);

    // Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
 
    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   bytes, x->data, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                   bytes, y->data, 0, NULL, NULL);
 
    // // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &x->size);
 
    // // Execute the kernel over the entire range of the data set 
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL);
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);


    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                                bytes, out->data, 0, NULL, NULL );
    
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}


void add(tensor * out, tensor * in_x, tensor * in_y)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    out = make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);
    // Note : that there may be problems with different shapes
	if (in_x->dim == in_y->dim)
	{
		for (int i = 0; i < out->size; i++)
			out->data[i] = in_x->data[i] + in_y->data[i];
        // cl_add(out, in_x, in_y);
	}
	else if ((in_x->dim == 4) &&  (in_y->dim == 2))
	{
		int a_batch = in_x->n;
		int a_channel = in_x->c;
		int a_h = in_x->h;
		int a_w = in_x->w;

		int b_batch = in_y->h;
		int b_channel = in_y->w;
		assert(a_channel == b_channel);
		
		float bvalue = 0;
		for(int ch = 0; ch< a_channel; ch++)
		{
			bvalue = in_y->data[ch];
			for(int h = 0; h < a_h; h++)
			{
				for(int w = 0; w < a_w; w++)
				{
					out->data[ch * out->h * out->w + h * out->w + w] = in_x->data[ch * in_x->h * in_x->w + h * in_x->w + w] + bvalue;
				}
			}
		}
	}
#ifdef DEBUG_TIME
        double end = clock();
        printf("[add time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void softmax(tensor * out, tensor * in_x)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);
    float sum = 0;

    for (int i = 0; i < out->size; i++)
    {
        out->data[i] = expf(in_x->data[i]);
        sum = sum + out->data[i];
    }

    assert(sum != 0);

    for (int i = 0; i < out->size; i++)
        out->data[i] = out->data[i] / sum;

#ifdef DEBUG_TIME
    double end = clock();
    printf("[softmax time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

//reshape = reshape(input_Placeholder, shape = [-1, 28, 28, 1]);
void reshape(tensor * out, tensor * in_x, int n, int c, int h, int w)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

    int shapeLength = 1;
    int tensorLength = 1;
    int index = -1;
    int shape[4];
    shape[0] = n;
    shape[1] = c;
    shape[2] = h;
    shape[3] = w;
    tensorLength = in_x->size;

    for (size_t i = 0; i < 4; ++i) 
    {
        if (shape[i] == -1) {
          // no -1 before
          assert(shapeLength > 0);
          index = i;
        }
        if (shape[i] == 0)
            shape[i] = 1;
        shapeLength *= shape[i];
    }

    if (shapeLength < 0) 
    {
        if (shapeLength != tensorLength) 
        {
            shapeLength = abs(shapeLength);
            shape[index] = tensorLength / shapeLength;
            shapeLength = tensorLength;
        }
    }

    assert(tensorLength == shapeLength);

    // Run
    out->n = shape[0];
    out->c = shape[1];
    out->h = shape[2];
    out->w = shape[3];

    if (n == 0)
    {
        out->n = 1;
        out->dim = 3;
    }

    if (c == 0)
    {
        out->c = 1;
        out->dim = 2;
    }

    if (h == 0)
    {
        out->h = 1;
        out->dim = 1;
    }

    if (w == 0)
    {
        out->w = 1;
        out->dim = 0;
    }

    for (int i = 0; i < out->size; i++)
        out->data[i] = in_x->data[i];

#ifdef DEBUG_TIME
    double end = clock();
    printf("[reshape time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}


void concat(tensor * out, tensor * in_x, tensor * in_y, int axis)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    //1-D
    //Chack
    assert (in_x->dim == in_y->dim);

    //Run
    out->n = in_x->n; out->c = in_x->c; out->h = in_x->h; out->w = in_x->w;

    if (axis == 0)
        out->n = out->n + in_y->n;
    else if (axis == 1)
        out->c = out->c + in_y->c;
    else if (axis == 2)
        out->n = out->n + in_y->h;
    else if (axis == 3)
        out->w = out->w + in_y->w;

    make_tensor(out, out->n, out->c, out->h, out->w);

    // push size
    int push_size = 1;
    int run_size = 1;
    for (int i = in_x->dim - 1; i >=axis; i--)
    {
        int shap_num;
        if (i == 0)
            shap_num = in_x->n;
        else if (i == 1)
            shap_num = in_x->c;
        else if (i == 2)
            shap_num = in_x->h;
        else if (i == 3)
            shap_num = in_x->w;
        push_size = push_size * shap_num;
    }
    for (int i = axis - 1; i >=0; i--)
    {
        int shap_num;
        if (i == 0)
            shap_num = in_x->n;
        if (i == 1)
            shap_num = in_x->c;
        if (i == 2)
            shap_num = in_x->h;
        if (i == 3)
            shap_num = in_x->w;
        run_size = run_size * shap_num;
    }

    // total size check
    int cnt = 0;
    for (int i = 0; i < run_size; i++)
    {
        for (int j = 0; j < push_size; j++)
        {
            out->data[cnt] = in_x->data[i * push_size + j];
            cnt++;
        }

        for (int j = 0; j < push_size; j++)
        {
            out->data[cnt] = in_y->data[i * push_size + j];
            cnt++;
        }
    }
#ifdef DEBUG_TIME
    double end = clock();
    printf("[concat time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void squeeze(tensor * out, tensor * in_x, int n, int c, int h, int w)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

    int axes_size = 0;
    if (n != 0)
        axes_size++;
    if (c != 0)
        axes_size++;
    if (h != 0)
        axes_size++;
    if (w != 0)
        axes_size++;

    int count = 0;
    int cnt = 0;
    //run
    for (int i = 0; i < in_x->dim; i++)
    {
        if (count < axes_size)
        {
            int axes;
            if (count == 0)
                axes = n;
            if (count == 1)
                axes = c;
            if (count == 2)
                axes = h;
            if (count == 3)
                axes = w;
            if (i == axes)
            {
                count++;
                continue;
                //drop out
            }
        }

        int A_shape;
        if (i == 0)
            A_shape = in_x->n;
        else if (i == 1)
            A_shape = in_x->c;
        else if (i == 2)
            A_shape = in_x->h;
        else if (i == 3)
            A_shape = in_x->w;

        if (cnt == 0)
            out->n = A_shape;
        else if (cnt == 1)
            out->c = A_shape;
        else if (cnt == 2)
            out->h = A_shape;
        else if (cnt == 3)
            out->w = A_shape;
        cnt++;
        //out.shape.push_back(A.shape[i]);
    }

    for (int i = 0; i < in_x->size; i++)
        out->data[i] = in_x->data[i];

#ifdef DEBUG_TIME
    double end = clock();
    printf("[squeeze time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void transpose(tensor * out, tensor * in_x, int n, int c, int h, int w)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

    int newA[4];
    int pos[4];
    int transPos[4];
    int shape[4];
    int in_x_shape[4];
    in_x_shape[0] = in_x->n;
    in_x_shape[1] = in_x->c;
    in_x_shape[2] = in_x->h;
    in_x_shape[3] = in_x->w;
    shape[0] = n;
    shape[1] = c;
    shape[2] = h;
    shape[3] = w;

    //newShape 
    out->n = in_x_shape[shape[0]]; 
    out->c = in_x_shape[shape[1]]; 
    out->h = in_x_shape[shape[2]]; 
    out->w = in_x_shape[shape[3]];     

    // init           
    for (int index = 0; index < 4; index++)
        pos[index] = 0;

    int S = 1;
    for (int index = 0; index < 4; index++)
        S = S * in_x_shape[index];

    for (int i = 0; i < S; i++)
    {
        int carryOut = 0;
        int index = i;
        // e.g 2 * 3 * 4, Loop of variable
        for (int j = 0; j < 4; j++)
        {   
            pos[j] = index % in_x_shape[j];
            carryOut = index / in_x_shape[j];
            index = carryOut;
        }

        for (int index = 0; index < 4; index++)
            transPos[index] = pos[shape[index]];

        out->data[where_pos4(out, transPos[0], transPos[1], transPos[2], transPos[3])]
        = in_x->data[where_pos4(in_x, pos[0], pos[1], pos[2], pos[3])];

    }

#ifdef DEBUG_TIME
    double end = clock();
    printf("[transpose time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void conv(tensor * out, tensor * in_x, tensor * filter, tensor * bias, int padding, int stride, int groups)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif
    //shape
    int inPic = in_x->n;
    int filterKernelNum = filter->n;

    assert(in_x->h >= filter->h);
    assert(in_x->w >= filter->w);

    int v_offset_Y = 0;
    int v_offset_X = 0;

    //virtual_height, virtual_weight
    int v_height = 0;
    int v_width = 0;

    //virtual_bound_height , virtual_bound_weight
    int vb_height = 0;
    int vb_width = 0;

    int pad = 0;

    if (padding)
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)in_x->h)/((float)stride));
        out->w = ceil(((float)in_x->w)/((float)stride));
        
        //padding
        int newY = filter->h + (out->h - 1) * stride;
        int newX = filter->w + (out->w - 1) * stride;

        v_offset_Y = (newY - in_x->h) / 2;
        v_offset_X = (newX - in_x->w) / 2;

        vb_height = in_x->h + v_offset_Y;
        vb_width  = in_x->w + v_offset_X;
        
        pad = ((out->h - 1) * stride + filter->h - in_x->h) / 2;
    }
    else
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)(in_x->h - filter->h+ 1))/((float)stride));
        out->w = ceil(((float)(in_x->w - filter->w+ 1))/((float)stride));

        vb_height = in_x->h;
        vb_width  = in_x->w;
        
        pad = 0;
    }

    //virtual_height, virtual_weight
    v_height = v_offset_Y;
    v_width = v_offset_X;

    make_tensor(out, out->n, out->c, out->h, out->w);

#ifdef im2colxGEMM

    int out_w,out_h;
    int workspace_size;

    out_w = out->h;
    out_h = out->w;
    workspace_size = out_h * out_w * filter->h * filter->h * in_x->c;
    float * colD = 0;

    if (!colD) colD = (float *) calloc(workspace_size, sizeof(float));    
    int c,h,w;

    int height_col = out_h;
    int width_col = out_w;
    int channels_col = in_x->c * filter->h * filter->h;

    for (int Pic = 0; Pic < inPic; Pic++)
    {
        for (c = 0; c < channels_col; ++c) 
        {
            for (h = 0; h < height_col; ++h) 
            {
                for (w = 0; w < width_col; ++w) 
                {
                    int w_offset = c % filter->h;
                    int h_offset = (c / filter->h) % filter->h;
                    int c_im = c / filter->h / filter->h;
                    int im_row = h_offset + h * stride;
                    int im_col = w_offset + w * stride;
                    int col_index = (c * height_col + h) * width_col + w;
                    //int col_index = (h * width_col + w) * channels_col + c;
                    colD[col_index] = im2col_get_pixel(in_x , in_x->h, in_x->w, in_x->c, im_row, im_col, c_im, pad);
                }
            }
        }

        int m = filter->n; // input height N
        int n = out_w * out_h; // filter width = number of filter = 9
        int p = filter->c * filter->h * filter->w; // CHW = input width = filter height = channel*ksize*ksize

        for (int i=0; i < m; i++) //2
        {
            for (int j=0; j < n; j++) //9
            {
                float sum = 0.0;
                for(int k = 0; k < p; k++) //18
                {
                    sum += filter->data[i * p + k] * colD[k * n + j];
                }
                out->data[i*n+j] = sum + bias->data[i];
            }
        }

        free(colD);
    }
#else    
    for (int Pic = 0; Pic < inPic; Pic++)
    {
        for (int filterKernel = 0; filterKernel < filterKernelNum; filterKernel++)// 32
        {
            for (int height = 0; height < out->h; height = height + 1)//28
            {
                for (int width = 0; width < out->w; width = width + 1)//28
                {
                    float featureValue = 0;
                    int offsetY = (height * stride);
                    int offsetX = (width  * stride);

                    for (int z = 0; z < filter->c; z++)
                    {
                        for (int y = 0; y < filter->h; y++)
                        {
                             for (int x = 0; x < filter->w; x++)
                             {
                                // logical_height, logical_weight
                                int l_height = y + offsetY;
                                int l_weight = x + offsetX;

                                if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
                                    featureValue = featureValue + in_x->data[Pic * in_x->c * in_x->h * in_x->w + z * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)] * filter->data[filterKernel * filter->c * filter->h * filter->w + z * filter->h * filter->w + y * filter->w + x];
                            }
                        }
                    }
                    out->data[Pic * out->c * out->h * out->w + filterKernel * out->h * out->w + height * out->w + width] = featureValue + bias->data[filterKernel];
                }
            }
        }
    }
#endif    

#ifdef DEBUG_TIME
    double end = clock();
    printf("[conv time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}
//max_pool = max_pool(relu, border = 'constant', dilation = [], padding = [], size = [1, 1, 2, 2], stride = [1, 1, 2, 2]);
void max_pool(tensor * out, tensor * in_x, int size, int padding, int stride)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif
    //Run
    //tensor<float> out;
    //out.shape.resize(4);

    //Chack
    assert(in_x->h >= size);
    assert(in_x->w >= size);

    int v_offset_T = 0;
    int v_offset_Z = 0;
    int v_offset_Y = 0;
    int v_offset_X = 0;

    //virtual_height, virtual_weight
    int v_height = 0;
    int v_width = 0;

    //virtual_bound_height , virtual_bound_weight
    int vb_height = 0;
    int vb_width = 0;

    if (padding)
    {
        out->n = in_x->n;
        out->c = in_x->c;
        out->h = (int)(ceil((float)(in_x->h)/(float)stride));
        out->w = (int)(ceil((float)(in_x->w)/(float)stride));

        int newY = size + (out->h - 1) * stride;
        int newX = size + (out->w - 1) * stride;

        v_offset_Y = (newY - in_x->h) / 2;
        v_offset_X = (newX - in_x->w) / 2;

        vb_height = in_x->h + v_offset_Y;
        vb_width = in_x->w + v_offset_X;
    }
    else
    {
        out->n = in_x->n;
        out->c = in_x->c;
        out->h = ceil(((float)(in_x->h - size + 1))/((float)stride));
        out->w = ceil(((float)(in_x->w - size + 1))/((float)stride));

        vb_height = in_x->h;
        vb_width = in_x->w;
    }

    //virtual_height, virtual_weight
    v_height = v_offset_Y;
    v_width = v_offset_X;

    make_tensor(out, out->n, out->c, out->h, out->w);

    // Tensor is [batch, height, width, channels], NNEF not
    // NNEF is [batch, channels, height, width]
    for (int N = 0; N < out->n; N++)
        //#pragma omp parallel for
        for (int C = 0; C < out->c; C++)
            for (int H = 0; H < out->h; H++)
                for (int W = 0; W < out->w; W++)
                {
                    float MaxValue = -FLT_MAX;
                    int offsetY = (H  * stride);
                    int offsetX = (W  * stride);

                    //for (int x = 0; x < size[0]; x++)
                        //for (int y = 0; y < size[1]; y++)
                    for (int z = 0; z < size; z++)
                        for (int t = 0; t < size; t++)
                            {
                                // logical_height, logical_weight
                                int l_height = z + offsetY;
                                int l_weight = t + offsetX;

                                if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
                                {
                                    float value = in_x->data[N * in_x->c * in_x->h * in_x->w + C * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)];
                                    if (MaxValue < value)
                                        MaxValue = value;
                                }
                            }
                    out->data[N * out->c * out->h * out->w + C * out->h * out->w + H * out->w + W] = MaxValue;
                }

#ifdef DEBUG_TIME
    double end = clock();
    printf("[maxpool time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

const char* relukernel = "\n" \
  "kernel void tensor_relu(__global float* a, __global float* out,\n" \
  "                      const unsigned int n) {                  \n" \
  "  int id = get_global_id(1);                                   \n" \
  "  if(id<n)                                                     \n" \
  "      out[id] = fmax(a[id], 0.0f);                                \n" \
  "}                                                              \n" \
  "\n";

void relu(tensor * out, tensor * in_x)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

    // // Device input buffers
    // cl_mem d_a;
    // // Device output buffer
    // cl_mem d_c;
 
    // cl_platform_id cpPlatform;        // OpenCL platform
    // cl_device_id device_id;           // device ID
    // cl_context context;               // context
    // cl_command_queue queue;           // command queue
    // cl_program program;               // program
    // cl_kernel kernel;                 // kernel

    // size_t bytes = in_x->size*sizeof(float);

    // int nsteps = INSTEPS;
    // int niters = ITERS;
    // size_t globalSize, localSize;
    // cl_int err;

    // // Number of work items in each local work group
    // localSize = 64;
 
    // globalSize = nsteps/niters;

    // // Bind to platform
    // err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
    // // Get ID for the device
    // err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
 
    // // Create a context 
    // context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
 
    // // Create a command queue
    // queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);

    // program = clCreateProgramWithSource(context, 1,
    //                         (const char **) &relukernel, NULL, &err);
 
    // // Build the program executable
    // clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
 
    // // Create the compute kernel in the program we wish to run
    // kernel = clCreateKernel(program, "tensor_relu", &err);

    // // Create the input and output arrays in device memory for our calculation
    // d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    // d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
 
    // // Write our data set into the input array in device memory
    // err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
    //                                bytes, in_x->data, 0, NULL, NULL);
 
    // // Set the arguments to our compute kernel
    // err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    // err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_c);
    // err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &in_x->size);
 
    // // Execute the kernel over the entire range of the data set 
    // err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
    //                                                           0, NULL, NULL);
 
    // // Wait for the command queue to get serviced before reading back results
    // clFinish(queue);


    // // Read the results from the device
    // clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
    //                             bytes, out->data, 0, NULL, NULL );
    
    // printf("size: %i %i\n", in_x->size, out->size);
    // printf("first elem: %f %f\n", in_x->data[100], out->data[100]);

    for (int i = 0; i < out->size; i++) 
        out->data[i] = fmax(in_x->data[i], 0.0);


    // for (int i = 0; i < out->size; i++) {
    //     float a = fmax(in_x->data[i], 0.0);
    //     if(out->data[i] != a){
    //         printf("mismatch %f %f\n", out->data[i], a);
    //     }
    // }
    // printf("first elem: %f %f\n", in_x->data[100], out->data[100]);

#ifdef DEBUG_TIME
    double end = clock();
    printf("[relu time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void relu6(tensor * out, tensor * in_x)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

    for (int i = 0; i < out->size; i++)
    {
        out->data[i] = fmin(fmax(in_x->data[i], 0.0), 6.0);
    }
#ifdef DEBUG_TIME
    double end = clock();
    printf("[relu6 time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void sigmoid(tensor * out, tensor * in_x)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

    for (int i = 0; i < out->size; i++)
        out->data[i] = 1.0 / (1.0 + expf(-in_x->data[i]));

#ifdef DEBUG_TIME
    double end = clock();
    printf("[sigmoid time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void convxbias(tensor * out, tensor * in_x, tensor * filter, float bias, int padding, int stride, int groups)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif
    //shape
    int inPic = in_x->n;
    int filterKernelNum = filter->n;

    assert(in_x->h >= filter->h);
    assert(in_x->w >= filter->w);

    int v_offset_Y = 0;
    int v_offset_X = 0;

    //virtual_height, virtual_weight
    int v_height = 0;
    int v_width = 0;

    //virtual_bound_height , virtual_bound_weight
    int vb_height = 0;
    int vb_width = 0;

    int pad = 0;

    if (padding)
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)in_x->h)/((float)stride));
        out->w = ceil(((float)in_x->w)/((float)stride));
        
        //padding
        int newY = filter->h + (out->h - 1) * stride;
        int newX = filter->w + (out->w - 1) * stride;

        v_offset_Y = (newY - in_x->h) / 2;
        v_offset_X = (newX - in_x->w) / 2;

        vb_height = in_x->h + v_offset_Y;
        vb_width  = in_x->w + v_offset_X;
        
        pad = ((out->h - 1) * stride + filter->h - in_x->h) / 2;
    }
    else
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)(in_x->h - filter->h+ 1))/((float)stride));
        out->w = ceil(((float)(in_x->w - filter->w+ 1))/((float)stride));

        vb_height = in_x->h;
        vb_width  = in_x->w;
        
        pad = 0;
    }

    //virtual_height, virtual_weight
    v_height = v_offset_Y;
    v_width = v_offset_X;

    make_tensor(out, out->n, out->c, out->h, out->w);

	if(groups ==1) //general convolution
	{
#ifdef im2colxGEMM

		int out_w,out_h;
		int workspace_size;

		out_w = out->h;
		out_h = out->w;
		workspace_size = out_h * out_w * filter->h * filter->h * in_x->c;
		float * colD = 0;
		
		if (!colD) colD = (float *) calloc(workspace_size, sizeof(float));    
		int c,h,w;

		int height_col = out_h;
		int width_col = out_w;
		int channels_col = in_x->c * filter->h * filter->h;
		
		for (int Pic = 0; Pic < inPic; Pic++)
		{
			for (c = 0; c < channels_col; ++c) 
			{
				for (h = 0; h < height_col; ++h) 
				{
					for (w = 0; w < width_col; ++w) 
					{
						int w_offset = c % filter->h;
						int h_offset = (c / filter->h) % filter->h;
						int c_im = c / filter->h / filter->h;
						int im_row = h_offset + h * stride;
						int im_col = w_offset + w * stride;
						int col_index = (c * height_col + h) * width_col + w;
						//int col_index = (h * width_col + w) * channels_col + c;
						colD[col_index] = im2col_get_pixel(in_x , in_x->h, in_x->w, in_x->c, im_row, im_col, c_im, pad);
					}
				}
			}

			int m = filter->n; // input height N
			int n = out_w * out_h; // filter width = number of filter = 9
			int p = filter->c * filter->h * filter->w; // CHW = input width = filter height = channel*ksize*ksize

			for (int i=0; i < m; i++) //2
			{
				for (int j=0; j < n; j++) //9
				{
					float sum = 0.0;
					for(int k = 0; k < p; k++) //18
					{
						// [ik][kj]
						sum += filter->data[i * p + k] * colD[k * n + j];
					}
					out->data[i*n+j] = sum + bias;
				}
			}

			free(colD);
		}
#else    
		for (int Pic = 0; Pic < inPic; Pic++)
		{
			for (int filterKernel = 0; filterKernel < filterKernelNum; filterKernel++)// 32
			{
				for (int height = 0; height < out->h; height = height + 1)//28
				{
					for (int width = 0; width < out->w; width = width + 1)//28
					{
						float featureValue = 0;
						int offsetY = (height * stride);
						int offsetX = (width  * stride);

						for (int z = 0; z < filter->c; z++)
						{
							for (int y = 0; y < filter->h; y++)
							{
								 for (int x = 0; x < filter->w; x++)
								 {
									// logical_height, logical_weight
									int l_height = y + offsetY;
									int l_weight = x + offsetX;

									if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
										featureValue = featureValue + in_x->data[Pic * in_x->c * in_x->h * in_x->w + z * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)] * filter->data[filterKernel * filter->c * filter->h * filter->w + z * filter->h * filter->w + y * filter->w + x];
								}
							}
						}
						out->data[Pic * out->c * out->h * out->w + filterKernel * out->h * out->w + height * out->w + width] = featureValue + bias;
					}
				}
			}
		}
#endif
	}
	else
	{
        int count = 0;
		for (int Pic = 0; Pic < inPic; Pic++)
		{
			for (int filterKernel = 0; filterKernel < filterKernelNum; filterKernel++)// 32
			{
				for (int height = 0; height < out->h; height = height + 1)//28
				{
					for (int width = 0; width < out->w; width = width + 1)//28
					{
						float featureValue = 0;
						int offsetY = (height * stride);
						int offsetX = (width  * stride);

                        for (int y = 0; y < filter->h; y++)
                        {
                             for (int x = 0; x < filter->w; x++)
                             {
                                // logical_height, logical_weight
                                int l_height = y + offsetY;
                                int l_weight = x + offsetX;

                                if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
                                {
                                    featureValue = featureValue + in_x->data[Pic * in_x->c * in_x->h * in_x->w + filterKernel * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)] * filter->data[filterKernel * filter->c * filter->h * filter->w + 0 * filter->h * filter->w + y * filter->w + x];
                                    //colD[count] = in_x->data[Pic * in_x->c * in_x->h * in_x->w + filterKernel * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)]; 
                                    //colD[count] = colD[count] * filter->data[filterKernel * filter->c * filter->h * filter->w + 0 * filter->h * filter->w + y * filter->w + x];
                                    //featureValue = featureValue + colD[count];                                      
                                }
                            }
                        }
						out->data[Pic * out->c * out->h * out->w + filterKernel * out->h * out->w + height * out->w + width] = featureValue + bias;
					}
				}
			}
		}
	}
#ifdef DEBUG_TIME
    double end = clock();
    printf("[convxbias time = %1.3f seconds, groups = %d]\n",(end-start)/CLOCKS_PER_SEC, groups);
#endif
}

void convxt_bias(tensor * out, tensor * in_x, tensor * filter, tensor * bias, int padding, int stride, int groups)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif
    //shape
    int inPic = in_x->n;
    int filterKernelNum = filter->n;

    assert(in_x->h >= filter->h);
    assert(in_x->w >= filter->w);

    int v_offset_Y = 0;
    int v_offset_X = 0;

    //virtual_height, virtual_weight
    int v_height = 0;
    int v_width = 0;

    //virtual_bound_height , virtual_bound_weight
    int vb_height = 0;
    int vb_width = 0;

    int pad = 0;

    if (padding)
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)in_x->h)/((float)stride));
        out->w = ceil(((float)in_x->w)/((float)stride));

        //padding
        int newY = filter->h + (out->h - 1) * stride;
        int newX = filter->w + (out->w - 1) * stride;

        v_offset_Y = (newY - in_x->h) / 2;
        v_offset_X = (newX - in_x->w) / 2;

        vb_height = in_x->h + v_offset_Y;
        vb_width  = in_x->w + v_offset_X;

        pad = ((out->h - 1) * stride + filter->h - in_x->h) / 2;
    }
    else
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)(in_x->h - filter->h+ 1))/((float)stride));
        out->w = ceil(((float)(in_x->w - filter->w+ 1))/((float)stride));

        vb_height = in_x->h;
        vb_width  = in_x->w;

        pad = 0;
    }

    //virtual_height, virtual_weight
    v_height = v_offset_Y;
    v_width = v_offset_X;

    make_tensor(out, out->n, out->c, out->h, out->w);

	if(groups == 1) //general convolution
	{
#ifdef im2colxGEMM

		int out_w,out_h;
		int workspace_size;

		out_w = out->h;
		out_h = out->w;
		workspace_size = out_h * out_w * filter->h * filter->h * in_x->c;
		float * colD = 0;

		if (!colD) colD = (float *) calloc(workspace_size, sizeof(float));
		int c,h,w;

		int height_col = out_h;
		int width_col = out_w;
		int channels_col = in_x->c * filter->h * filter->h;

		for (int Pic = 0; Pic < inPic; Pic++)
		{
			for (c = 0; c < channels_col; ++c)
			{
				for (h = 0; h < height_col; ++h)
				{
					for (w = 0; w < width_col; ++w)
					{
						int w_offset = c % filter->h;
						int h_offset = (c / filter->h) % filter->h;
						int c_im = c / filter->h / filter->h;
						int im_row = h_offset + h * stride;
						int im_col = w_offset + w * stride;
						int col_index = (c * height_col + h) * width_col + w;
						//int col_index = (h * width_col + w) * channels_col + c;
						colD[col_index] = im2col_get_pixel(in_x , in_x->h, in_x->w, in_x->c, im_row, im_col, c_im, pad);
					}
				}
			}

			int m = filter->n; // input height N
			int n = out_w * out_h; // filter width = number of filter = 9
			int p = filter->c * filter->h * filter->w; // CHW = input width = filter height = channel*ksize*ksize

			for (int i=0; i < m; i++) //2
			{
				for (int j=0; j < n; j++) //9
				{
					float sum = 0.0;
					for(int k = 0; k < p; k++) //18
					{
						// [ik][kj]
						sum += filter->data[i * p + k] * colD[k * n + j];
					}
					out->data[i*n+j] = sum + bias->data[i];
				}
			}
			free(colD);
		}
#else
    assert(0);
#endif
	}
	else
	{
        assert(0);
	}

#ifdef DEBUG_TIME
    double end = clock();
    printf("[convxbias time = %1.3f seconds, groups = %d]\n",(end-start)/CLOCKS_PER_SEC, groups);
#endif
}
void batch_normalization(tensor * out, tensor * in_x, tensor * mean, tensor * variance, tensor * offset, tensor * scale, float epsilon)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    out = make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

	for (int B = 0; B < in_x->n; B++)
		for (int C = 0; C < in_x->c; C++)
			for (int H = 0; H < in_x->h; H++)
				for (int W = 0; W < in_x->w; W++)
				{
					int out_index = B * out->c * out->h * out->w + C * out->h * out->w + H * out->w + W;
					int in_index  = B * in_x->c * in_x->h * in_x->w + C * in_x->h * in_x->w + H * in_x->w + W;
					out->data[out_index] = (in_x->data[in_index] - mean->data[C]) / sqrt(variance->data[C] + epsilon);
					out->data[out_index] = scale->data[C] * out->data[out_index] + offset->data[C];
				}

#ifdef DEBUG_TIME
    double end = clock();
    printf("[batch_normalization time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void avg_pool(tensor * out, tensor * in_x, int size, int padding, int stride)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif
    //Run
    //tensor<float> out;
    //out.shape.resize(4);

    //Check
    assert(in_x->h >= size);
    assert(in_x->w >= size);

    int v_offset_T = 0;
    int v_offset_Z = 0;
    int v_offset_Y = 0;
    int v_offset_X = 0;

    //virtual_height, virtual_weight
    int v_height = 0;
    int v_width = 0;

    //virtual_bound_height , virtual_bound_weight
    int vb_height = 0;
    int vb_width = 0;

    if (padding)
    {
        out->n = in_x->n;
        out->c = in_x->c;
        out->h = (int)(ceil((float)(in_x->h)/(float)stride));
        out->w = (int)(ceil((float)(in_x->w)/(float)stride));

        int newY = size + (out->h - 1) * stride;
        int newX = size + (out->w - 1) * stride;

        v_offset_Y = (newY - in_x->h) / 2;
        v_offset_X = (newX - in_x->w) / 2;

        vb_height = in_x->h + v_offset_Y;
        vb_width = in_x->w + v_offset_X;
    }
    else
    {
        out->n = in_x->n;
        out->c = in_x->c;
        out->h = ceil(((float)(in_x->h - size + 1))/((float)stride));
        out->w = ceil(((float)(in_x->w - size + 1))/((float)stride));

        vb_height = in_x->h;
        vb_width = in_x->w;
    }

    //virtual_height, virtual_weight
    v_height = v_offset_Y;
    v_width = v_offset_X;

    make_tensor(out, out->n, out->c, out->h, out->w);

    // Tensor is [batch, height, width, channels], NNEF not
    // NNEF is [batch, channels, height, width]
    for (int N = 0; N < out->n; N++)
        //#pragma omp parallel for
        for (int C = 0; C < out->c; C++)
            for (int H = 0; H < out->h; H++)
                for (int W = 0; W < out->w; W++)
                {
                    float AvgValue = 0.0;
                    int Div = 0;
                    int offsetY = (H  * stride);
                    int offsetX = (W  * stride);
					int out_index = N * out->c * out->h * out->w + C * out->h * out->w + H * out->w + W;
					out->data[out_index] = 0;
                    //for (int x = 0; x < size[0]; x++)
                        //for (int y = 0; y < size[1]; y++)
                    for (int z = 0; z < size; z++)
                        for (int t = 0; t < size; t++)
                            {
                                // logical_height, logical_weight
                                int l_height = z + offsetY;
                                int l_weight = t + offsetX;

                                if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
                                {
                                    float value = in_x->data[N * in_x->c * in_x->h * in_x->w + C * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)];
                                    AvgValue = AvgValue + value;
                                }
                            }
                            out->data[out_index] = AvgValue / (size * size);

                }


#ifdef DEBUG_TIME
    double end = clock();
    printf("[avgpool time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void min(tensor * out, tensor * in_x, float y)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

	float tmp;
    for (int i = 0; i < out->size; i++)
	{
        tmp = in_x->data[i];
		out->data[i] = (tmp > y) ? y : tmp;
	}

#ifdef DEBUG_TIME
        double end = clock();
        printf("[min time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void bn_sqrt(tensor * out, tensor * in_x, float epsilon)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    out = make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

    for (int i = 0; i < in_x->size; i++)
    {
        out->data[i] = sqrt(in_x->data[i] + epsilon);
    }

#ifdef DEBUG_TIME
        double end = clock();
        printf("[bn_sqrt time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void bn_sub(tensor * out, tensor * in_x, tensor * in_y)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    // Note : that there may be problems with different shapes
    out = make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);
    for (int B = 0; B < in_x->n; B++)
        for (int C = 0; C < in_x->c; C++)
            for (int H = 0; H < in_x->h; H++)
                for (int W = 0; W < in_x->w; W++)
                {
                    out->data[where_pos4(out, B, C, H, W)] = in_x->data[where_pos4(in_x, B, C, H, W)] - in_y->data[C];
                }

#ifdef DEBUG_TIME
    double end = clock();
        printf("[bn_sub time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void bn_div(tensor * out, tensor * in_x, tensor * in_y)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    // Note : that there may be problems with different shapes
    out = make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);
    for (int B = 0; B < in_x->n; B++)
        for (int C = 0; C < in_x->c; C++)
            for (int H = 0; H < in_x->h; H++)
                for (int W = 0; W < in_x->w; W++)
                {
                    out->data[where_pos4(out, B, C, H, W)] = in_x->data[where_pos4(in_x, B, C, H, W)] / in_y->data[C];
                }

#ifdef DEBUG_TIME
    double end = clock();
        printf("[bn_div time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void bn_mul(tensor * out, tensor * in_x, tensor * in_y)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    // Note : that there may be problems with different shapes
    out = make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);
    for (int B = 0; B < in_x->n; B++)
        for (int C = 0; C < in_x->c; C++)
            for (int H = 0; H < in_x->h; H++)
                for (int W = 0; W < in_x->w; W++)
                {
                    out->data[where_pos4(out, B, C, H, W)] = in_x->data[where_pos4(in_x, B, C, H, W)] * in_y->data[C];
                }
#ifdef DEBUG_TIME
    double end = clock();
        printf("[bn_mul time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void bn_add(tensor * out, tensor * in_x, tensor * in_y)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    out = make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);
    for (int B = 0; B < in_x->n; B++)
        for (int C = 0; C < in_x->c; C++)
            for (int H = 0; H < in_x->h; H++)
                for (int W = 0; W < in_x->w; W++)
                {
                    out->data[where_pos4(out, B, C, H, W)] = in_x->data[where_pos4(in_x, B, C, H, W)] + in_y->data[C];
                }
#ifdef DEBUG_TIME
    double end = clock();
        printf("[bn_add time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}
