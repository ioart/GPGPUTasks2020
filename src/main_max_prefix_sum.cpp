#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include <numeric>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:22
#include "cl/max_prefix_sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    ocl::Kernel max_prefix_sum(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");
    max_prefix_sum.compile();

    unsigned benchmarkingIters = 10;
    unsigned max_n = (1u << 24u);

    for (unsigned n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / (int)n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (unsigned i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (unsigned i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = (int)i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (unsigned iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (unsigned i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = (int)i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1e6) / t.lapAvg() << " millions/s" << std::endl;
        }
        {
            // TODO: implement on OpenCL
            unsigned int work_group_size = 128;

            // Fill array indexes by natural numbers: 1, 2, 3, ..., n.
            std::vector<int> indexes(n);
            std::iota(std::begin(indexes), std::end(indexes), 1);

            auto init_sum = gpu::gpu_mem_32i::createN(n);
            auto init_max_sum = gpu::gpu_mem_32i::createN(n);
            auto init_indexes = gpu::gpu_mem_32i::createN(n);

            init_sum.writeN(as.data(), n);
            init_max_sum.writeN(as.data(), n);
            init_indexes.writeN(indexes.data(), n);

            auto curr_sum = gpu::gpu_mem_32i::createN(n);
            auto curr_max_sum = gpu::gpu_mem_32i::createN(n);
            auto curr_indexes = gpu::gpu_mem_32i::createN(n);
            auto next_sum = gpu::gpu_mem_32i::createN(n);
            auto next_max_sum = gpu::gpu_mem_32i::createN(n);
            auto next_indexes = gpu::gpu_mem_32i::createN(n);

            timer t;
            for (unsigned i = 0; i < benchmarkingIters; ++i) {
                // We should run each benchmark item under the same conditions,
                // otherwise the problem will be actually solved only for the first time.
                unsigned int data_size = n;
                init_sum.copyToN(curr_sum, n);
                init_max_sum.copyToN(curr_max_sum, n);
                init_indexes.copyToN(curr_indexes, n);

                while (data_size > 1) {
                    max_prefix_sum.exec(
                            gpu::WorkSize(work_group_size, data_size),
                            next_sum, next_max_sum, next_indexes,
                            curr_sum, curr_max_sum, curr_indexes,
                            data_size);

                    curr_sum.swap(next_sum);
                    curr_max_sum.swap(next_max_sum);
                    curr_indexes.swap(next_indexes);

                    data_size = (data_size + work_group_size - 1) / work_group_size;
                }

                int max_sum = 0;
                int result = 0;
                curr_max_sum.readN(&max_sum, 1);
                curr_indexes.readN(&result, 1);

                EXPECT_THE_SAME(reference_max_sum, max_sum, "GPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "GPU result should be consistent!");
                t.nextLap();
            }

            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1e6) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
