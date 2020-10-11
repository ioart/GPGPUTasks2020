#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef TILE_SIZE
#  define TILE_SIZE 16
#endif
#  define WORK_GROUP_SIZE 16


__kernel void matrix_transpose(
        __global const float *a,
        __global float *at,
        unsigned m,
        unsigned k)
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t local_i = get_local_id(0);
    size_t local_j = get_local_id(1);

    // Extra column for preventing bank conflict.
    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    tile[local_j][local_i] = a[j * k + i];

    barrier(CLK_LOCAL_MEM_FENCE);
    float tmp = tile[local_j][local_i];
    tile[local_j][local_i] = tile[local_i][local_j];
    tile[local_i][local_j] = tmp;

    barrier(CLK_LOCAL_MEM_FENCE);
    // Check the array indexes which can be smaller than the work size.
    if (j < k && i < m)
        at[i * m + j] = tile[local_j][local_i];
}
