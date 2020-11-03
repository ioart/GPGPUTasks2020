#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef WORK_GROUP_SIZE
#  define WORK_GROUP_SIZE 128
#endif


/**
 * Set 1 if value's bit #bit is 0.
 */
__kernel void init_by_bit(
        __global unsigned* prefix_sums,
        const __global unsigned* as,
        unsigned as_size,
        unsigned bit)
{
    unsigned global_id = get_global_id(0);
    if (global_id < as_size) {
        prefix_sums[global_id] = 1 - (as[global_id] >> bit) & 1;
    }
}

/**
 * Get prefix sum on each block in tree structure.
 */
__kernel void prefix_tree(
        __global unsigned* as,
        unsigned as_size,
        unsigned block_size)
{
    unsigned global_id = get_global_id(0);

    if (global_id >= as_size / (block_size * 2)) {
        return;
    }

    unsigned from = global_id * block_size * 2  + block_size - 1;
    unsigned to = from + block_size;

    if (to < as_size) {
#ifdef DEBUG
        printf("%d:   %d\t%d\n", block_size, from, to);
#endif
        as[to] += as[from];
    }
}

/**
 * Get prefix sum for each element of array.
 */
__kernel void prefix_sum_on_tree(
        __global unsigned* prefix_sums,
        const __global unsigned* as,
        unsigned as_size)
{
    unsigned global_id = get_global_id(0);
    unsigned local_id = get_local_id(0);

    if (global_id >= as_size) {
        return;
    }

    __local unsigned local_as[WORK_GROUP_SIZE];
    local_as[local_id] = as[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned summ = 0;
    for (unsigned mask = 1; mask < WORK_GROUP_SIZE; mask *= 2) {
        if ((global_id + 1) & mask) {
             unsigned index = global_id - (global_id + 1) % mask;
             summ += local_as[index % WORK_GROUP_SIZE];
        }
    }
    for (unsigned mask = WORK_GROUP_SIZE; mask <= as_size; mask *= 2) {
        if ((global_id + 1) & mask) {
             unsigned index = global_id - (global_id + 1) % mask;
             summ += as[index];
        }
    }
    prefix_sums[global_id] = summ;
}

/**
 * Radix sort using prefix sums.
 */
__kernel void radix(
        __global unsigned* result,
        const __global unsigned* as,
        const __global unsigned* prefix_sums,
        unsigned as_size,
        unsigned bit)
{
    unsigned global_id = get_global_id(0);
    if (global_id < as_size) {
        unsigned is_one = as[global_id] & (1 << bit);
        unsigned sum_zeros = prefix_sums[global_id];
        unsigned all_zeros = prefix_sums[as_size - 1];
        // Number of 'zeros' on the left side.
        unsigned to = sum_zeros - 1;
        if (is_one) {
            // Number of 'ones' on the left size + total number of 'zeros'.
            to = (global_id - sum_zeros) + all_zeros;
        }
#ifdef DEBUG
        printf("set %u -> %u\n", global_id, to);
#endif
        result[to] = as[global_id];
    }
}