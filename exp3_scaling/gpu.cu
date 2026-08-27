// ============================================================================
// Experiment 3 — GPU thread scaling
//
// Same element kernel as the CPU side, launched with a grid-stride loop so the
// logical thread count G can vary freely. Total work (N, K) is fixed; only G
// changes:
//
//   G == 1          -> kernel<<<1,1>>>   (cross-validates Experiment 1)
//   1 < G <= 1024   -> kernel<<<1,G>>>
//   G > 1024        -> kernel<<<ceil(G/256),256>>> with an early-exit guard
//
// Usage:
//   ./exp3_gpu --threads 65536 --size 1000000 --compute-iterations 500
// ============================================================================
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "common.h"
#include "../common/cuda_utils.h"

namespace {

__global__ void elem_kernel(const float* in, float* out, long N, int K, float a, float b,
                            long G) {
    const long tid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= G) return;  // exact logical thread count (extra launched threads exit)
    for (long i = tid; i < N; i += G) {
        float x = in[i];
        for (int k = 0; k < K; ++k) x = x * a + b;
        out[i] = x;
    }
}

// Pick a launch configuration with EXACTLY G active threads.
void launch_scaled(dim3& grid, dim3& block, long G) {
    if (G <= 1) {
        grid = dim3(1);
        block = dim3(1);
    } else if (G <= 1024) {
        grid = dim3(1);
        block = dim3(G);
    } else {
        const int b = 256;
        grid = dim3(static_cast<unsigned>((G + b - 1) / b));
        block = dim3(b);
    }
}

const std::vector<std::string> HEADER = {
    "platform", "threads", "size", "k", "latency_ms", "throughput_elem_s", "gflops"};

void emit(const std::string& csv_file, const std::vector<std::string>& row) {
    print_csv_row(row);
    if (!csv_file.empty()) CsvSink(csv_file, HEADER).write(row);
}

}  // namespace

int main(int argc, char** argv) {
    const long G = get_arg_long(argc, argv, "--threads", 65536);
    const long N = get_arg_long(argc, argv, "--size", 1000000L);
    const int K = get_arg_int(argc, argv, "--compute-iterations", 500);
    const float a = static_cast<float>(get_arg_double(argc, argv, "--a", 0.5));
    const float b = static_cast<float>(get_arg_double(argc, argv, "--b", 1.0));
    const int warmup = get_arg_int(argc, argv, "--warmup", 3);
    const int iters = get_arg_int(argc, argv, "--iters", 10);
    const std::string csv_file = get_arg(argc, argv, "--csv-file", "");

    std::vector<float> in = gen_lcg_floats(N, 4242u);
    std::vector<float> out(static_cast<size_t>(N), 0.0f);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * N));

    dim3 grid, block;
    launch_scaled(grid, block, G);

    // --show-launch: how logical threads map to blocks / warps (no benchmark).
    if (has_arg(argc, argv, "--show-launch")) {
        std::printf("Requested logical threads : %ld\n", G);
        std::printf("Block size                : %u\n", block.x);
        std::printf("Grid size                 : %u\n", grid.x);
        std::printf("Logical warps             : %ld\n", (G + 31) / 32);
        std::printf("Launch                    : <<<%u, %u>>>\n", grid.x, block.x);
        return 0;
    }

    auto pre = [&] { CUDA_CHECK(cudaMemcpy(d_in, in.data(), sizeof(float) * N, cudaMemcpyHostToDevice)); };
    auto launch = [&] { elem_kernel<<<grid, block>>>(d_in, d_out, N, K, a, b, G); };

    CudaEventTimer timer;
    std::vector<double> k_samples;
    for (int i = 0; i < warmup; ++i) { pre(); launch(); CUDA_CHECK(cudaDeviceSynchronize()); }
    for (int i = 0; i < iters; ++i) {
        pre();
        timer.begin();
        launch();
        timer.end();
        CUDA_CHECK(cudaDeviceSynchronize());
        k_samples.push_back(timer.elapsed_ms());
    }
    const double k_ms = compute_stats(k_samples).median;

    // Correctness: copy back and check against double-precision reference.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, sizeof(float) * N, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long i = 0; i < N && ok; ++i) {
        double r = in[static_cast<size_t>(i)];
        for (int k = 0; k < K; ++k) r = r * static_cast<double>(a) + static_cast<double>(b);
        ok = nearly_equal(out[static_cast<size_t>(i)], static_cast<float>(r), 1e-2f);
    }
    report_check("gpu element kernel (threads=" + std::to_string(G) + ")", ok);
    if (!ok) {
        std::fprintf(stderr, "  out[i] != reference at some index\n");
        std::exit(2);
    }

    const double throughput = static_cast<double>(N) / (k_ms * 1e-3);
    const double gflops = 2.0 * static_cast<double>(N) * static_cast<double>(K) / (k_ms * 1e-3) / 1e9;

    std::fprintf(stderr, "[gpu] threads=%ld: kernel=%.4f ms  %.3e elem/s  %.3f GFLOPS\n", G, k_ms,
                throughput, gflops);

    if (human_format(argc, argv)) {
        HumanReport r;
        r.title = "Experiment 3 (GPU): " + std::to_string(G) + " threads";
        r.add("Platform", "GPU");
        r.add("Logical threads", std::to_string(G));
        r.add("Block size", std::to_string(block.x));
        r.add("Grid size", std::to_string(grid.x));
        r.add("Logical warps", std::to_string((G + 31) / 32));
        r.add("Elements", std::to_string(N));
        r.add("Compute iterations/elem", std::to_string(K));
        r.add("Kernel latency", fmt(k_ms) + " ms");
        r.add("Throughput", fmt(throughput, 3) + " elem/s");
        r.add("GFLOPS", fmt(gflops, 3));
        r.print();
    } else {
        emit(csv_file,
             {"GPU", std::to_string(G), std::to_string(N), std::to_string(K), fmt(k_ms),
              fmt(throughput, 3), fmt(gflops, 3)});
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
