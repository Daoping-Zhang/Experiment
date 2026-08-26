// ============================================================================
// Experiment 2A — GPU memory mapping (coalescing)
//
// C[i] = A[i] + B[i] with a FIXED logical thread count. Only the access order
// changes:
//   stride 1      : coalesced grid-stride loop (good — warp reads 32 floats
//                   contiguously)
//   stride s >= 2 : every element still processed exactly once, but through a
//                   permutation so consecutive threads touch addresses `s`
//                   apart (bad — many memory transactions per warp)
//
// The permutation is a true bijection (built by make_stride_permutation), so
// total FLOPs and total bytes are identical to the coalesced case.
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

__global__ void vec_add_permuted(const float* A, const float* B, float* C, const int* perm,
                                 long N) {
    long p = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long stride = static_cast<long>(gridDim.x) * blockDim.x;
    for (; p < N; p += stride) {
        const int idx = perm[p];
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

    std::vector<float> A = gen_lcg_floats(N, 12345u);
    std::vector<float> B = gen_lcg_floats(N, 67890u);
    std::vector<float> C(static_cast<size_t>(N), 0.0f);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    int* d_perm = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * N));

    std::vector<int> perm;
    if (stride > 1) {
        perm = make_stride_permutation(N, stride);
        // verify bijection once on host
        std::vector<char> seen(static_cast<size_t>(N), 0);
        bool bijective = true;
        for (int idx : perm) {
            if (idx < 0 || idx >= N || seen[static_cast<size_t>(idx)]) {
                bijective = false;
                break;
            }
            seen[static_cast<size_t>(idx)] = 1;
        }
        report_check("stride permutation bijection", bijective);
        if (!bijective) std::exit(2);
        CUDA_CHECK(cudaMalloc(&d_perm, sizeof(int) * N));
        CUDA_CHECK(cudaMemcpy(d_perm, perm.data(), sizeof(int) * N, cudaMemcpyHostToDevice));
    }

    // Upload A, B (and the permutation) once — inputs are identical every
    // iteration, so only the kernel is re-run in the timing loop.
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), sizeof(float) * N, cudaMemcpyHostToDevice));

    auto launch = [&] {
        if (stride == 1)
            vec_add_coalesced<<<grid, block>>>(d_A, d_B, d_C, N);
        else
            vec_add_permuted<<<grid, block>>>(d_A, d_B, d_C, d_perm, N);
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

    // Correctness: copy C back and verify C[i] == A[i]+B[i].
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

    emit(csv_file,
         {"GPU", variant, std::to_string(threads), std::to_string(N), fmt(k_ms), fmt(elements_per_s, 3),
          fmt(bw_gbs, 3)});

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    if (d_perm) CUDA_CHECK(cudaFree(d_perm));
    return 0;
}
