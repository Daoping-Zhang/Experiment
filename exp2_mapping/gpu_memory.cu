// ============================================================================
// Experiment 2A — GPU memory mapping (coalescing)
//
// C[i] = A[i] + B[i] with a FIXED logical thread count. Only the access order
// changes:
//   stride 1      : coalesced grid-stride loop (warp reads 32 floats
//                   contiguously -> few memory transactions)
//   stride s >= 2 : every element still processed exactly once, but consecutive
//                   threads touch addresses `s` apart -> uncoalesced
//
// The stride-s access order is an ARITHMETIC permutation computed on the fly:
//     idx = (p % M) * s + (p / M),  where M = N / s
// For consecutive logical positions p this steps by s, and it is a bijection
// over [0, N). It uses mask/shift (N and s are powers of two), so there is NO
// permutation table in global memory: every stride reads exactly A + B and
// writes C — total FLOPs and total bytes are identical across strides.
//
// Usage:
//   ./exp2_gpu_memory --stride 1  --threads 65536 --size 16777216
//   ./exp2_gpu_memory --stride 32 --threads 65536 --size 16777216
// ============================================================================
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "common.h"
#include "../common/cuda_utils.h"

namespace {

__global__ void vec_add_coalesced(const float* A, const float* B, float* C, long N) {
    long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long stride = static_cast<long>(gridDim.x) * blockDim.x;
    for (; i < N; i += stride) C[i] = A[i] + B[i];
}

// Arithmetic strided permutation (no global-memory table).
//   mask = M - 1, logM = log2(M), M = N / stride (powers of two)
//   idx  = (p & mask) * stride + (p >> logM) == (p % M) * stride + (p / M)
__global__ void vec_add_permuted(const float* A, const float* B, float* C, long N,
                                 long stride, long mask, int logM) {
    long p = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long G = static_cast<long>(gridDim.x) * blockDim.x;
    for (; p < N; p += G) {
        const long idx = (p & mask) * stride + (p >> logM);
        C[idx] = A[idx] + B[idx];
    }
}

const std::vector<std::string> HEADER = {
    "platform", "stride", "threads", "size", "latency_ms",
    "elements_per_s", "effective_bandwidth_gbs"};

void emit(const std::string& csv_file, const std::vector<std::string>& row) {
    print_csv_row(row);
    if (!csv_file.empty()) CsvSink(csv_file, HEADER).write(row);
}

}  // namespace

int main(int argc, char** argv) {
    const long stride = get_arg_long(argc, argv, "--stride", 1);
    long threads = get_arg_long(argc, argv, "--threads", 65536);
    const long N = get_arg_long(argc, argv, "--size", 16777216L);  // 2^24
    const int warmup = get_arg_int(argc, argv, "--warmup", 3);
    const int iters = get_arg_int(argc, argv, "--iters", 10);
    const std::string csv_file = get_arg(argc, argv, "--csv-file", "");
    const int block = get_arg_int(argc, argv, "--block", 256);

    const int grid = static_cast<int>((threads + block - 1) / block);
    threads = grid * block;  // actual launched logical threads
    const std::string variant = (stride == 1) ? "1" : std::to_string(stride);

    // Arithmetic permutation requires N divisible by stride and M = N/stride a
    // power of two (mask/shift). Defaults satisfy this; reject otherwise.
    if (N % stride != 0) {
        std::fprintf(stderr, "size must be divisible by stride (N=%ld, stride=%ld)\n", N, stride);
        std::exit(1);
    }
    const long M = N / stride;
    if (M < 2 || (M & (M - 1)) != 0) {
        std::fprintf(stderr, "M=N/stride must be a power of two (N=%ld, stride=%ld)\n", N, stride);
        std::exit(1);
    }
    int logM = 0;
    for (long m = M; m > 1; m >>= 1) ++logM;
    const long mask = M - 1;

    std::vector<float> A = gen_lcg_floats(N, 12345u);
    std::vector<float> B = gen_lcg_floats(N, 67890u);
    std::vector<float> C(static_cast<size_t>(N), 0.0f);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * N));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), sizeof(float) * N, cudaMemcpyHostToDevice));

    auto launch = [&] {
        if (stride == 1)
            vec_add_coalesced<<<grid, block>>>(d_A, d_B, d_C, N);
        else
            vec_add_permuted<<<grid, block>>>(d_A, d_B, d_C, N, stride, mask, logM);
    };

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

    // Correctness: full check C[i] == A[i]+B[i] (also proves the arithmetic
    // permutation is a bijection — any missed/duplicated index would fail).
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, sizeof(float) * N, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long i = 0; i < N && ok; ++i)
        ok = nearly_equal(C[static_cast<size_t>(i)], A[static_cast<size_t>(i)] + B[static_cast<size_t>(i)], 1e-4f);
    report_check("gpu vector add (stride " + variant + ")", ok);
    if (!ok) {
        std::fprintf(stderr, "  C[i] != A[i]+B[i] at some index\n");
        std::exit(2);
    }

    const double elements_per_s = static_cast<double>(N) / (k_ms * 1e-3);
    const double bw_gbs = 12.0 * static_cast<double>(N) / (k_ms * 1e-3) / 1e9;

    std::fprintf(stderr, "[gpu] stride=%s: threads=%ld kernel=%.3f ms  %.3e elem/s  %.3f GB/s\n",
                variant.c_str(), threads, k_ms, elements_per_s, bw_gbs);

    if (human_format(argc, argv)) {
        HumanReport r;
        r.title = "Experiment 2A (GPU): stride " + variant + " access";
        r.add("Platform", "GPU");
        r.add("Stride", variant);
        r.add("Logical threads", std::to_string(threads));
        r.add("Elements", std::to_string(N));
        r.add("Kernel latency", fmt(k_ms) + " ms");
        r.add("Elements/s", fmt(elements_per_s, 3));
        r.add("Effective bandwidth", fmt(bw_gbs, 3) + " GB/s");
        r.print();
    } else {
        emit(csv_file,
             {"GPU", variant, std::to_string(threads), std::to_string(N), fmt(k_ms), fmt(elements_per_s, 3),
              fmt(bw_gbs, 3)});
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
