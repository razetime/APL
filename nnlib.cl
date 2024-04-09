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