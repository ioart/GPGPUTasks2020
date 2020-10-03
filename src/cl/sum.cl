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
