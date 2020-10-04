#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef WORK_GROUP_SIZE
#  define WORK_GROUP_SIZE 128
#endif

#ifndef VALUES_PER_WORK_ITEM
#  define VALUES_PER_WORK_ITEM 64
#endif

__kernel void sum1(
        __global unsigned* result,
        __global const unsigned* as,
        unsigned n)
{
    const unsigned id = get_global_id(0);
    atomic_add(result, as[id]);
}

__kernel void sum2(
        __global unsigned* result,
        __global const unsigned* as,
        unsigned n)
{
    const unsigned id = get_global_id(0);

    if (id > n / VALUES_PER_WORK_ITEM) {
        return;
    }

    unsigned sum = 0;
    for (unsigned i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        unsigned idx = id * VALUES_PER_WORK_ITEM + i;
        sum += as[idx];
    }

    atomic_add(result, sum);
}

__kernel void sum3(
        __global unsigned* result,
        __global const unsigned* as,
        unsigned n)
{
    const unsigned local_id = get_local_id(0);
    const unsigned group_id = get_group_id(0);
    const unsigned group_size = get_local_size(0);

    if (group_id * group_size > n / VALUES_PER_WORK_ITEM) {
        return;
    }

    unsigned sum = 0;
    for (unsigned i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        unsigned idx = group_id * group_size * VALUES_PER_WORK_ITEM + i * group_size + local_id;
        if (idx >= n) {
            break;
        }
        sum += as[idx];
    }

    atomic_add(result, sum);
}

__kernel void sum4(
        __global unsigned* result,
        __global const unsigned* as,
        unsigned n)
{
    const unsigned global_id = get_global_id(0);
    const unsigned local_id = get_local_id(0);

    __local unsigned local_as[WORK_GROUP_SIZE];
    local_as[local_id] = as[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (2 * local_id < nvalues) {
            unsigned right = local_as[local_id + nvalues/2];
            local_as[local_id] += right;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        atomic_add(result, local_as[0]);
    }
}
