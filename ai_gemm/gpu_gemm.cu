// ============================================================================
// AI validation — GPU GEMM (cuBLAS)
//
// C = A * B (float32), matching the CPU side exactly. Uses cuBLAS SGEMM.
//
// cuBLAS is column-major, so to compute ROW-major C = A*B we pass the operands
// swapped with CUBLAS_OP_N (see derivation in README):
//   cublasSgemm(handle, OP_N, OP_N, n,n,n, &alpha, B, n, A, n, &beta, C, n)
//
// Usage:
//   ./ai_gemm_gpu --sizes 128,256,512,1024,2048,4096
// ============================================================================
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include "common.h"
#include "../common/cuda_utils.h"

namespace {

std::vector<int> parse_sizes(const std::string& s) {
    std::vector<int> sizes;
    std::string cur;
    for (char c : s + ",") {
        if (c == ',') {
            if (!cur.empty()) sizes.push_back(std::stoi(cur));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    return sizes;
}

bool gemm_check_host(int n, const float* A, const float* B, const float* C) {
    std::vector<float> v(n), w(n), Cv(n), Aw(n);
    std::mt19937 rng = make_rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) v[i] = dist(rng);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) w[i] += B[i * n + j] * v[j];
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) Aw[i] += A[i * n + k] * w[k];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) Cv[i] += C[i * n + j] * v[j];
    for (int i = 0; i < n; ++i)
        if (!nearly_equal(Cv[i], Aw[i], 1e-2f)) return false;
    return true;
}

const std::vector<std::string> HEADER = {"platform", "matrix_size", "latency_ms", "gflops"};

void emit(const std::string& csv_file, const std::vector<std::string>& row) {
    print_csv_row(row);
    if (!csv_file.empty()) CsvSink(csv_file, HEADER).write(row);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string sizes_arg = get_arg(argc, argv, "--sizes", "128,256,512,1024,2048,4096");
    const int warmup = get_arg_int(argc, argv, "--warmup", 3);
    const int iters = get_arg_int(argc, argv, "--iters", 10);
    const std::string csv_file = get_arg(argc, argv, "--csv-file", "");

    const std::vector<int> sizes = parse_sizes(sizes_arg);

    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasCreate failed\n");
        return 1;
    }

    for (int n : sizes) {
        const size_t nelem = static_cast<size_t>(n) * static_cast<size_t>(n);
        std::vector<float> A = gen_lcg_floats(static_cast<long>(nelem), 1000u);
        std::vector<float> B = gen_lcg_floats(static_cast<long>(nelem), 2000u);
        std::vector<float> C(nelem, 0.0f);

        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * nelem));
        CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * nelem));
        CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * nelem));

        const float alpha = 1.0f, beta = 0.0f;

        auto pre = [&] {
            CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(float) * nelem, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B, B.data(), sizeof(float) * nelem, cudaMemcpyHostToDevice));
        };
        auto launch = [&] {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_B, n, d_A, n, &beta,
                        d_C, n);
        };

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

        CUDA_CHECK(cudaMemcpy(C.data(), d_C, sizeof(float) * nelem, cudaMemcpyDeviceToHost));
        const bool ok = gemm_check_host(n, A.data(), B.data(), C.data());
        report_check("gpu gemm correctness n=" + std::to_string(n), ok);
        if (!ok) {
            std::fprintf(stderr, "  C != A*B for n=%d\n", n);
            std::exit(2);
        }

        const double gflops = 2.0 * std::pow(static_cast<double>(n), 3) / (k_ms * 1e-3) / 1e9;
        std::fprintf(stderr, "[gemm] n=%d: kernel=%.4f ms  %.3f GFLOPS\n", n, k_ms, gflops);

        emit(csv_file, {"GPU", std::to_string(n), fmt(k_ms), fmt(gflops, 3)});

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    cublasDestroy(handle);
    return 0;
}
