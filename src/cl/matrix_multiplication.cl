#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef TILE_SIZE
#  define TILE_SIZE 16
#endif


__kernel void matrix_multiplication(
        __global const float *a,
        __global const float *b,
        __global float *c,
        unsigned M,
        unsigned K,
        unsigned N)
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t local_i = get_local_id(0);
    size_t local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0.0;

    for (size_t tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        tileA[local_j][local_i] = a[j * K + (tileK * TILE_SIZE + local_i)];
        tileB[local_j][local_i] = b[i + (tileK * TILE_SIZE + local_j) * N];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (size_t k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[local_j][k] * tileB[k][local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[j * N + i] = sum;
}
