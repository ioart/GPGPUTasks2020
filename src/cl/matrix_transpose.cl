#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef TILE_SIZE
#  define TILE_SIZE 16
#endif


__kernel void matrix_transpose1(
        __global const float *a,
        __global float *at,
        unsigned W,
        unsigned H)
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

    if (i < W && j < H) {
        at[i * H + j] = a[j * W + i];
    }
}


__kernel void matrix_transpose2(
        __global const float *a,
        __global float *at,
        unsigned W,
        unsigned H)
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t local_i = get_local_id(0);
    size_t local_j = get_local_id(1);

    // Extra column for preventing bank conflict.
    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    tile[local_j][local_i] = a[j * W + i];

    barrier(CLK_LOCAL_MEM_FENCE);
    // Check the array indexes which can be smaller than the work size.
    unsigned ii = i - local_i + local_j;
    unsigned jj = j - local_j + local_i;
    if (ii < W && jj < H) {
        at[ii * H + jj] = tile[local_i][local_j];
    }
}
