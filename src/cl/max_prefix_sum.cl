#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef WORK_GROUP_SIZE
#  define WORK_GROUP_SIZE 128
#endif

__kernel void max_prefix_sum(
        __global int* next_sum,
        __global int* next_max_sum,
        __global int* next_indexes,
        __global const int* curr_sum,
        __global const int* curr_max_sum,
        __global const int* curr_indexes,
        unsigned int n)
{
    const unsigned global_id = get_global_id(0);
    const unsigned local_id = get_local_id(0);
    const unsigned group_id = get_group_id(0);

    __local int local_sum[WORK_GROUP_SIZE];
    __local int local_max_sum[WORK_GROUP_SIZE];
    __local int local_indexes[WORK_GROUP_SIZE];

    if (global_id < n) {
        local_sum[local_id] = curr_sum[global_id];
        local_max_sum[local_id] = curr_max_sum[global_id];
        local_indexes[local_id] = curr_indexes[global_id];
    } else {
        local_sum[local_id] = 0;
        local_max_sum[local_id] = 0;
        local_indexes[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        int sum = 0;
        int max_sum = 0;
        int index = 0;

        for (unsigned i = 0; i < WORK_GROUP_SIZE; ++i) {
            if (sum + local_max_sum[i] > max_sum) {
                max_sum = sum + local_max_sum[i];
                index = local_indexes[i];
            }
            sum += local_sum[i];
        }

        next_sum[group_id] = sum;
        next_max_sum[group_id] = max_sum;
        next_indexes[group_id] = index;
    }
}