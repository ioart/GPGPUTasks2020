#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_transpose_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int H = 1024;
    unsigned int W = 1024;

    std::vector<float> as(H * W, 0);
    std::vector<float> as_t(H * W, 0);

    FastRandom r(H + W);
    for (auto& x : as) {
        x = r.nextf();
    }
    std::cout << "Data generated for H=" << H << ", W=" << W << "!" << std::endl;

    gpu::gpu_mem_32f as_gpu, as_t_gpu;
    as_gpu.resizeN(H * W);
    as_t_gpu.resizeN(W * H);

    as_gpu.writeN(as.data(), H * W);

    unsigned int tile_size = 16;
    unsigned int work_size_x = (H + tile_size - 1) / tile_size * tile_size;
    unsigned int work_size_y = (W + tile_size - 1) / tile_size * tile_size;
    auto work_size = gpu::WorkSize(tile_size, tile_size, work_size_x, work_size_y);

    {
        std::string defines = "-D TILE_SIZE=" + std::to_string(tile_size);
        ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose", defines);
        matrix_transpose_kernel.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            matrix_transpose_kernel.exec(work_size, as_gpu, as_t_gpu, H, W);

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << H * W / 1e6 / t.lapAvg() << " millions/s" << std::endl;
    }

    as_t_gpu.readN(as_t.data(), H * W);

    // Проверяем корректность результатов
    for (unsigned j = 0; j < H; ++j) {
        for (unsigned i = 0; i < W; ++i) {
            float a = as[j * W + i];
            float b = as_t[i * H + j];
            if (a != b) {
                std::cerr << "Not the same!" << std::endl;
                return 1;
            }
        }
    }

    return 0;
}
