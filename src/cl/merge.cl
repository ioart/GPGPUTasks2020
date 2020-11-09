#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef WORK_GROUP_SIZE
#  define WORK_GROUP_SIZE 128
#endif


__kernel void merge(
        __global float* result,
        const __global float* as,
        unsigned as_size,
        unsigned block_size)
{
    unsigned global_id = get_global_id(0);
    if (global_id >= as_size) {
        return;
    }

    unsigned x0 = global_id / (2 * block_size) * (2 * block_size);
    unsigned y0 = min(x0 + block_size, as_size);
    unsigned yn = min(y0 + block_size, as_size);
    unsigned local_id = global_id % (2 * block_size);

    int l = max((int)local_id - (int)(yn - y0), 0) - 1;
    int r = min(block_size, local_id);
    while (l < r - 1) {
        int m = (l + r) / 2;
        if (as[x0 + m] <= as[y0 + local_id - m - 1]) {
            l = m;
        } else {
            r = m;
        }
    }

    unsigned x = x0 + r;
    unsigned y = y0 + local_id - r;
    unsigned from = (x < y0 && (y >= yn || as[x] <= as[y])) ? x : y;

#ifdef DEBUG
    printf("%d -> %d\n", from, global_id);
#endif
    result[global_id] = as[from];
}