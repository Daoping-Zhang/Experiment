// ============================================================================
// Experiment 1 — GPU single thread
//
// Strictly kernel<<<1,1>>> : exactly ONE CUDA thread, no blocks beyond 1.
// Same workloads as the CPU side (dependent chain / independent chains /
// branch). Two latencies are reported separately:
//   kernel latency  : CUDA-event time of the kernel only (pure GPU compute)
//   end-to-end       : launch + (data transfer) + kernel + sync (offload cost)
//
// Usage:
//   ./exp1_gpu --case dependent --iterations 100000000
//   ./exp1_gpu --case independent --chains 4 --iterations 100000000
//   ./exp1_gpu --case branch --data random --iterations 10000000
// ============================================================================
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "common.h"
#include "../common/cuda_utils.h"

namespace {

constexpr int BRANCH_INNER = 8;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
__global__ void dependent_kernel(float* out, long N, float A, float B) {
    float x = 1.0f;
    for (long i = 0; i < N; ++i) x = x * A + B;
    out[0] = x;
}

template <int C>
__global__ void independent_kernel(float* out, long per, float A, float B) {
    float x[C];
#pragma unroll
    for (int c = 0; c < C; ++c) x[c] = 1.0f;
    for (long i = 0; i < per; ++i) {
#pragma unroll
        for (int c = 0; c < C; ++c) x[c] = x[c] * A + B;
    }
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < C; ++c) s += x[c];
    out[0] = s;
}

__device__ __forceinline__ float branch_taken_gpu(float v) {
    float r = v;
    for (int k = 0; k < BRANCH_INNER; ++k) r = r * 0.99f + 0.01f;
    return r;
}

__device__ __forceinline__ float branch_not_taken_gpu(float v) {
    float r = v;
    for (int k = 0; k < BRANCH_INNER; ++k) r = r * 0.99f + 0.01f;
    return r;
}

__global__ void branch_kernel(const float* data, float* out, long N) {
    float x = 0.0f;
    for (long i = 0; i < N; ++i) {
        if (data[i] > 0.0f)
            x += branch_taken_gpu(data[i]);
        else
            x += branch_not_taken_gpu(data[i]);
    }
    out[0] = x;
}

// ---------------------------------------------------------------------------
// Host reference implementations (double precision)
// ---------------------------------------------------------------------------
double chain_f64(long N, double A, double B) {
    double x = 1.0;
    for (long i = 0; i < N; ++i) x = x * A + B;
    return x;
}

double independent_f64(long per, int C, double A, double B) {
    std::vector<double> x(static_cast<size_t>(C), 1.0);
    for (long i = 0; i < per; ++i)
        for (int c = 0; c < C; ++c) x[static_cast<size_t>(c)] = x[static_cast<size_t>(c)] * A + B;
    double s = 0.0;
    for (int c = 0; c < C; ++c) s += x[static_cast<size_t>(c)];
    return s;
}

double branch_ref(const std::vector<float>& data) {
    double x = 0.0;
    for (float v : data) {
        double r = v;
        for (int k = 0; k < BRANCH_INNER; ++k) r = r * 0.99 + 0.01;
        x += r;
    }
    return x;
}

const std::vector<std::string> EXP1_HEADER = {
    "platform",       "workload",        "variant",      "threads",
    "iterations",     "latency_ms",      "throughput_ops_s", "gflops",
    "cycles",         "instructions",    "ipc",
    "branches",       "branch_misses",   "branch_miss_rate",
    "kernel_latency_ms", "end_to_end_ms"};

void emit(const std::string& csv_file, const std::vector<std::string>& row) {
    print_csv_row(row);
    if (!csv_file.empty()) CsvSink(csv_file, EXP1_HEADER).write(row);
}

// Run `kernel` warmup+iters times; collect kernel latency (CUDA events) and
// end-to-end wall time (including any per-iteration data transfer passed in).
template <typename KernelLauncher, typename PreTransfer>
void run_gpu(KernelLauncher&& launch, PreTransfer&& pre_transfer,
             int warmup, int iters, double* k_ms, double* e2e_ms) {
    CudaEventTimer timer;
    if (!timer.ok) {
        std::fprintf(stderr, "failed to create CUDA events\n");
        std::exit(1);
    }
    auto measure_once = [&]() {
        pre_transfer();                       // optional H2D
        timer.begin();
        launch();                             // kernel launch (async)
        timer.end();
        CUDA_CHECK(cudaDeviceSynchronize());
        const double k = timer.elapsed_ms();  // kernel-only
        return k;
    };

    std::vector<double> k_samples, e_samples;
    k_samples.reserve(iters);
    e_samples.reserve(iters);
    for (int i = 0; i < warmup; ++i) measure_once();
    for (int i = 0; i < iters; ++i) {
        const double t0 = now_seconds();
        pre_transfer();
        timer.begin();
        launch();
        timer.end();
        CUDA_CHECK(cudaDeviceSynchronize());
        const double t1 = now_seconds();
        k_samples.push_back(timer.elapsed_ms());
        e_samples.push_back((t1 - t0) * 1e3);
    }
    *k_ms = compute_stats(std::move(k_samples)).median;
    *e2e_ms = compute_stats(std::move(e_samples)).median;
}

void run_chain_case(int argc, char** argv) {
    const std::string workload = get_arg(argc, argv, "--case", "dependent");
    const long N = get_arg_long(argc, argv, "--iterations", 100000000L);
    const int chains = get_arg_int(argc, argv, "--chains", 1);
    const float A = static_cast<float>(get_arg_double(argc, argv, "--a", 0.5));
    const float B = static_cast<float>(get_arg_double(argc, argv, "--b", 1.0));
    const int warmup = get_arg_int(argc, argv, "--warmup", 3);
    const int iters = get_arg_int(argc, argv, "--iters", 10);
    const std::string csv_file = get_arg(argc, argv, "--csv-file", "");

    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    long eff_iterations = N;
    double reference = 0.0;
    std::string variant;
    if (workload == "dependent") {
        variant = "dependent";
        eff_iterations = N;
        reference = chain_f64(N, A, B);
    } else {
        variant = "chains=" + std::to_string(chains);
        const long per = N / chains;
        eff_iterations = per * chains;
        reference = independent_f64(per, chains, A, B);
    }

    double k_ms = 0.0, e2e_ms = 0.0;
    auto no_transfer = [] {};
    if (workload == "dependent") {
        run_gpu(
            [&] { dependent_kernel<<<1, 1>>>(d_out, N, A, B); }, no_transfer, warmup,
            iters, &k_ms, &e2e_ms);
    } else {
        const long per = N / chains;
        switch (chains) {
            case 1: run_gpu([&] { independent_kernel<1><<<1, 1>>>(d_out, per, A, B); },
                            no_transfer, warmup, iters, &k_ms, &e2e_ms); break;
            case 2: run_gpu([&] { independent_kernel<2><<<1, 1>>>(d_out, per, A, B); },
                            no_transfer, warmup, iters, &k_ms, &e2e_ms); break;
            case 4: run_gpu([&] { independent_kernel<4><<<1, 1>>>(d_out, per, A, B); },
                            no_transfer, warmup, iters, &k_ms, &e2e_ms); break;
            case 8: run_gpu([&] { independent_kernel<8><<<1, 1>>>(d_out, per, A, B); },
                            no_transfer, warmup, iters, &k_ms, &e2e_ms); break;
            default:
                std::fprintf(stderr, "chains must be 1/2/4/8\n");
                std::exit(1);
        }
    }

    float host_out = 0.0f;
    CUDA_CHECK(cudaMemcpy(&host_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    const bool ok = nearly_equal(host_out, static_cast<float>(reference), 5e-3f);
    report_check("gpu " + workload + " (vs double reference)", ok);
    if (!ok) {
        std::fprintf(stderr, "  result=%.6f reference=%.6f\n", host_out,
                     static_cast<float>(reference));
        std::exit(2);
    }

    const double throughput = static_cast<double>(eff_iterations) / (k_ms * 1e-3);
    const double gflops = 2.0 * static_cast<double>(eff_iterations) / (k_ms * 1e-3) / 1e9;

    std::fprintf(stderr, "[gpu] %s %s: kernel=%.6f ms  e2e=%.6f ms  %.3f GFLOPS(kernel)\n",
                workload.c_str(), variant.c_str(), k_ms, e2e_ms, gflops);

    emit(csv_file,
         {"GPU", workload, variant, "1", std::to_string(eff_iterations), fmt(k_ms),
          fmt(throughput, 3), fmt(gflops, 3),
          "NA", "NA", "NA", "NA", "NA", "NA", fmt(k_ms), fmt(e2e_ms)});

    CUDA_CHECK(cudaFree(d_out));
}

void run_branch_case(int argc, char** argv) {
    const long N = get_arg_long(argc, argv, "--iterations", 10000000L);
    const std::string data_mode = get_arg(argc, argv, "--data", "predictable");
    const int warmup = get_arg_int(argc, argv, "--warmup", 3);
    const int iters = get_arg_int(argc, argv, "--iters", 10);
    const std::string csv_file = get_arg(argc, argv, "--csv-file", "");

    std::vector<float> data(static_cast<size_t>(N));
    if (data_mode == "predictable") {
        std::fill(data.begin(), data.end(), 1.0f);
    } else {
        std::mt19937 rng = make_rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (long i = 0; i < N; ++i) data[static_cast<size_t>(i)] = dist(rng);
    }
    const double reference = branch_ref(data);

    float* d_data = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    double k_ms = 0.0, e2e_ms = 0.0;
    auto pre = [&] { CUDA_CHECK(cudaMemcpy(d_data, data.data(), sizeof(float) * N, cudaMemcpyHostToDevice)); };
    auto launch = [&] { branch_kernel<<<1, 1>>>(d_data, d_out, N); };
    run_gpu(launch, pre, warmup, iters, &k_ms, &e2e_ms);

    float host_out = 0.0f;
    CUDA_CHECK(cudaMemcpy(&host_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    const bool ok = nearly_equal(host_out, static_cast<float>(reference), 5e-3f);
    report_check("gpu branch (vs double reference)", ok);
    if (!ok) {
        std::fprintf(stderr, "  result=%.6f reference=%.6f\n", host_out,
                     static_cast<float>(reference));
        std::exit(2);
    }

    const double throughput = static_cast<double>(N) / (k_ms * 1e-3);
    const double gflops = 2.0 * static_cast<double>(N) * BRANCH_INNER / (k_ms * 1e-3) / 1e9;

    std::fprintf(stderr, "[gpu] branch(%s): kernel=%.6f ms  e2e=%.6f ms  %.3f GFLOPS(kernel)\n",
                data_mode.c_str(), k_ms, e2e_ms, gflops);

    emit(csv_file,
         {"GPU", "branch", data_mode, "1", std::to_string(N), fmt(k_ms), fmt(throughput, 3),
          fmt(gflops, 3),
          "NA", "NA", "NA", "NA", "NA", "NA", fmt(k_ms), fmt(e2e_ms)});

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}

}  // namespace

int main(int argc, char** argv) {
    const std::string cs = get_arg(argc, argv, "--case", "dependent");
    if (cs == "branch")
        run_branch_case(argc, argv);
    else
        run_chain_case(argc, argv);
    return 0;
}
