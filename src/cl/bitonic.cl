#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef WORK_GROUP_SIZE
#  define WORK_GROUP_SIZE 128
#endif

#define SWAP(a, b) {float tmp = a; a = b; b = tmp; }  \

#define COMP_AND_SWAP(as, as_size, from, to) {        \
    if (to < as_size && as[from] > as[to]) {          \
        SWAP(as[from], as[to]);                       \
    }                                                 \
}                                                     \

void get_from_to(unsigned id, unsigned box_size, unsigned block_size, unsigned* from, unsigned* to) {
    unsigned shift = (id / block_size) * block_size * 2;
    unsigned offset = id % block_size;

    *from = shift + offset;
    if (block_size == box_size) {
        *to = shift + (2 * block_size - 1) - offset;
    }
    else {
        *to = *from + block_size;
    }
}


__kernel void bitonic_local(
        __global float* as,
        unsigned as_size,
        unsigned box_size,
        unsigned block_size)
{
    unsigned global_id = get_global_id(0);
    unsigned local_id = get_local_id(0);

    __local float local_as[2 * WORK_GROUP_SIZE];

    if (2 * global_id + 1 < as_size) {
        local_as[2 * local_id] = as[2 * global_id];
        local_as[2 * local_id + 1] = as[2 * global_id + 1];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned local_block_size = block_size; local_block_size > 0; local_block_size /= 2) {
        unsigned from, to;
        get_from_to(local_id, box_size, local_block_size, &from, &to);
        COMP_AND_SWAP(local_as, as_size, from, to);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (2 * global_id + 1 < as_size) {
        as[2 * global_id] = local_as[2 * local_id];
        as[2 * global_id + 1] = local_as[2 * local_id + 1];
    }
}

__kernel void bitonic(
        __global float* as,
        unsigned as_size,
        unsigned box_size,
        unsigned block_size)
{
    if (block_size <= WORK_GROUP_SIZE) {
        bitonic_local(as, as_size, box_size, block_size);
    }
    else {
        unsigned global_id = get_global_id(0);
        unsigned from, to;
        get_from_to(global_id, box_size, block_size, &from, &to);
        COMP_AND_SWAP(as, as_size, from, to);
    }
}
