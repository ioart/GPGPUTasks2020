#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum1(
        __global unsigned* result,
        __global const unsigned* as,
        unsigned n)
{
    const unsigned id = get_global_id(0);

    if (id >= n) {
        return;
    }

    atomic_add(result, as[id]);
}

__kernel void sum2(
        __global unsigned* result,
        __global const unsigned* as,
        unsigned n,
        unsigned values_per_work_item)
{
    const unsigned id = get_global_id(0);

    if (id > n / values_per_work_item) {
        return;
    }

    unsigned sum = 0;
    for (unsigned i = 0; i < values_per_work_item; ++i) {
        unsigned idx = id * values_per_work_item + i;
        if (idx >= n) {
            break;
        }
        sum += as[idx];
    }

    atomic_add(result, sum);
}
