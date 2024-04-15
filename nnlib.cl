kernel void tensor_add(__global float* a, __global float* b, __global float* c,
                       const unsigned int n) {
    int id = get_global_id(0);
    if(id<n)
        c[id] = a[id] + b[id];
}

kernel void tensor_add42(__global float* a, __global float* b, __global float* c,
                       const unsigned int n) {

}

kernel void tensor_relu(__global float* a, __global float* out,
                        const unsigned int n) {
    int id = get_global_id(0);
    if(id<n)
        out[id] = fmax(a[id], 0.0f);
}
kernel void softmax(__global float* x, __global float* out,
                      const unsigned int n) {                  
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        out[i] = exp(x[i]);
        sum = sum + out[i];
    }

    for (int i = 0; i < n; i++)
        out[i] = out[i] / sum;
}

kernel void batch_add(__global float* x, __global float* y, __global float* out,
                       const unsigned int channel,
                       const unsigned int x_w,
                       const unsigned int x_h,
                       const unsigned int o_w,
                       const unsigned int o_h) {
    int ch = get_global_id(0);
    if(ch<channel) {
        float bvalue = y[ch];
        for(int h = 0; h < x_h; h++)
        {
            for(int w = 0; w < x_w; w++)
            {
                out[ch * o_h * o_w + h * o_w + w] = x[ch * x_h * x_w + h * x_w + w] + bvalue;
            }
        }
    }
}

  kernel void conv(__global float* out, __global float* x,
                   __global float* filter, __global float* bias,
                   const int stride, const int vb_height,
                   const int vb_weight, const int v_offset_X,
                   const int v_offset_Y, const int x_c, const int x_h,
                   const int x_w, const int f_c, const int f_h, const int f_w,
                   const int inPic, const int filterKernelNum, const int o_c,
                   const int o_h, const int o_w, const int v_width,
                   const int v_height,const int vb_width) {
    int pic = get_global_id(0);
    // for (int pic = 0; pic < inPic; pic++)
    // {
        for (int filterKernel = 0; filterKernel < filterKernelNum; filterKernel++)// 32
        {
            for (int height = 0; height < o_h; height = height + 1)//28
            {
                for (int width = 0; width < o_w; width = width + 1)//28
                {
                    float featureValue = 0;
                    int offsetY = (height * stride);
                    int offsetX = (width  * stride);

                    for (int z = 0; z < f_c; z++)
                    {
                        for (int y = 0; y < f_h; y++)
                        {
                             for (int x = 0; x < f_w; x++)
                             {
                                // logical_height, logical_weight
                                int l_height = y + offsetY;
                                int l_weight = x + offsetX;

                                if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
                                    featureValue = featureValue + x[pic * x_c * x_h * x_w + z * x_h * x_w + (l_height - v_offset_Y) * x_w + (l_weight - v_offset_X)] * filter[filterKernel * f_c * f_h * f_w + z * f_h * f_w + y * f_w + x];
                            }
                        }
                    }
                    out[pic * o_c * o_h * o_w + filterKernel * o_h * o_w + height * o_w + width] = featureValue + bias[filterKernel];
                }
            }
        }
    // }
}