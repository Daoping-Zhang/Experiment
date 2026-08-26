// ============================================================================
// Experiment 2B — GPU work/control mapping (warp divergence)
//
// Same 50% heavy / 50% light element workload as the CPU side, fixed logical
// thread count, only the distribution changes:
//   grouped : heavy elements contiguous, then light elements contiguous
//             (each warp is uniform → no divergence)
//   mixed   : heavy/light alternate element by element
//             (each warp holds both branches → serialized → divergence)
//
// Usage:
//   ./exp2_gpu_control --distribution grouped --threads 65536 --tasks 1048576
// ============================================================================
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "common.h"
#include "../common/cuda_utils.h"

namespace {

constexpr int HEAVY = 200;
constexpr int LIGHT = 20;

__global__ void control_kernel(const float* vals, const char* is_heavy, float* out, long N) {
    const long tid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long stride = static_cast<long>(gridDim.x) * blockDim.x;
    float acc = 0.0f;
    for (long i = tid; i < N; i += stride) {
        float r = vals[i];
        if (is_heavy[i]) {
            for (int j = 0; j < HEAVY; ++j) r = r * 0.99f + 0.01f;
        } else {
            for (int j = 0; j < LIGHT; ++j) r = r * 0.99f + 0.01f;
        }
        acc += r;
    }
    out[tid] = acc;
}

const std::vector<std::string> HEADER = {
    "platform", "distribution", "threads", "tasks", "heavy_iters", "light_iters",
    "latency_ms", "throughput_tasks_s"};

void emit(const std::string& csv_file, const std::vector<std::string>& row) {
    print_csv_row(row);
    if (!csv_file.empty()) CsvSink(csv_file, HEADER).write(row);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dist = get_arg(argc, argv, "--distribution", "grouped");
    const long N = get_arg_long(argc, argv, "--tasks", 1048576L);
    long threads = get_arg_long(argc, argv, "--threads", 65536);
    const int block = get_arg_int(argc, argv, "--block", 256);
    const int warmup = get_arg_int(argc, argv, "--warmup", 3);
    const int iters = get_arg_int(argc, argv, "--iters", 10);
    const std::string csv_file = get_arg(argc, argv, "--csv-file", "");

    const int grid = static_cast<int>((threads + block - 1) / block);
    threads = grid * block;

    std::vector<float> vals = gen_lcg_floats(N, 999u);
    std::vector<char> is_heavy(static_cast<size_t>(N));
    for (long i = 0; i < N; ++i)
        is_heavy[static_cast<size_t>(i)] = (dist == "grouped") ? (i < N / 2 ? 1 : 0)
                                                               : ((i % 2 == 0) ? 1 : 0);

    // --dry-run: show the warp -> H/L placement (use a small --threads, e.g. 64).
    if (has_arg(argc, argv, "--dry-run")) {
        const long num_warps = threads / 32;
        const long show = std::min<long>(num_warps, 32L);
        std::printf("Distribution: %s   (threads=%ld, tasks=%ld, warps=%ld, H=heavy L=light)\n\n",
                    dist.c_str(), threads, N, num_warps);
        for (long w = 0; w < show; ++w) {
            std::printf("Warp %ld: ", w);
            for (int lane = 0; lane < 32; ++lane) {
                const long i = w * 32 + lane;
                if (i >= N) break;
                std::printf("%c ", is_heavy[static_cast<size_t>(i)] ? 'H' : 'L');
            }
            std::printf("\n");
        }
        if (num_warps > show) std::printf("... (%ld more warps)\n", num_warps - show);
        return 0;
    }

    // Host reference.
    double reference = 0.0;
    for (long i = 0; i < N; ++i) {
        double r = vals[static_cast<size_t>(i)];
        const int k = is_heavy[static_cast<size_t>(i)] ? HEAVY : LIGHT;
        for (int j = 0; j < k; ++j) r = r * 0.99 + 0.01;
        reference += r;
    }

    float *d_vals = nullptr, *d_out = nullptr;
    char* d_heavy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vals, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * threads));
    CUDA_CHECK(cudaMalloc(&d_heavy, sizeof(char) * N));
    CUDA_CHECK(cudaMemcpy(d_vals, vals.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_heavy, is_heavy.data(), sizeof(char) * N, cudaMemcpyHostToDevice));

    auto launch = [&] { control_kernel<<<grid, block>>>(d_vals, d_heavy, d_out, N); };

    CudaEventTimer timer;
    std::vector<double> k_samples;
    for (int i = 0; i < warmup; ++i) { launch(); CUDA_CHECK(cudaDeviceSynchronize()); }
    for (int i = 0; i < iters; ++i) {
        timer.begin();
        launch();
        timer.end();
        CUDA_CHECK(cudaDeviceSynchronize());
        k_samples.push_back(timer.elapsed_ms());
    }
    const double k_ms = compute_stats(k_samples).median;

    std::vector<float> out(static_cast<size_t>(threads));
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, sizeof(float) * threads, cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (float v : out) sum += v;
    const bool ok = nearly_equal(static_cast<float>(sum), static_cast<float>(reference), 5e-3f);
    report_check("gpu control (" + dist + ") vs reference", ok);
    if (!ok) {
        std::fprintf(stderr, "  sum=%.6f reference=%.6f\n", static_cast<float>(sum),
                     static_cast<float>(reference));
        std::exit(2);
    }

    const double throughput = static_cast<double>(N) / (k_ms * 1e-3);
    std::fprintf(stderr, "[gpu] %s: kernel=%.3f ms  %.3e tasks/s\n", dist.c_str(), k_ms, throughput);

    if (human_format(argc, argv)) {
        HumanReport r;
        r.title = "Experiment 2B (GPU): " + dist + " distribution";
        r.add("Platform", "GPU");
        r.add("Distribution", dist);
        r.add("Logical threads", std::to_string(threads));
        r.add("Tasks", std::to_string(N));
        r.add("Heavy / Light", std::to_string(HEAVY) + " / " + std::to_string(LIGHT) + " iters");
        r.add("Kernel latency", fmt(k_ms) + " ms");
        r.add("Throughput", fmt(throughput, 3) + " tasks/s");
        r.print();
    } else {
        emit(csv_file,
             {"GPU", dist, std::to_string(threads), std::to_string(N), std::to_string(HEAVY),
              std::to_string(LIGHT), fmt(k_ms), fmt(throughput, 3)});
    }

    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_heavy));
    return 0;
}
