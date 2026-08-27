// ============================================================================
// AI validation — CPU GEMM (C = A * B, row-major, float32)
//
// Uses an optimized BLAS when available (Apple Accelerate on macOS, OpenBLAS /
// any cblas on Linux). Falls back to a tiled reference implementation if no
// BLAS is present (correct but much slower — a real run should use BLAS).
//
// Usage:
//   ./ai_gemm_cpu --sizes 128,256,512,1024,2048,4096
// ============================================================================
#include "common.h"

#include <cstdio>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#define HAVE_CBLAS 1
#elif defined(USE_CBLAS)
#include <cblas.h>
#define HAVE_CBLAS 1
#endif

namespace {

#ifndef HAVE_CBLAS
// Simple tiled fallback (used only when no BLAS library is linked).
void naive_sgemm(int n, const float* A, const float* B, float* C) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) C[i * n + j] = 0.0f;
    const int BS = 64;
    for (int ii = 0; ii < n; ii += BS)
        for (int kk = 0; kk < n; kk += BS)
            for (int jj = 0; jj < n; jj += BS)
                for (int i = ii; i < ii + BS && i < n; ++i) {
                    const float* Ap = A + i * n;
                    for (int k = kk; k < kk + BS && k < n; ++k) {
                        const float a = Ap[k];
                        for (int j = jj; j < jj + BS && j < n; ++j)
                            C[i * n + j] += a * B[k * n + j];
                    }
                }
}
#endif

void gemm(int n, const float* A, const float* B, float* C) {
#ifdef HAVE_CBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A, n, B, n, 0.0f, C,
                n);
#else
    naive_sgemm(n, A, B, C);
#endif
}

// Randomized projection check: verify C == A*B by comparing C*v with A*(B*v)
// for a random vector v. O(n^2), cheap for every size.
bool gemm_check(int n, const float* A, const float* B, const float* C) {
    std::vector<float> v(static_cast<size_t>(n)), w(static_cast<size_t>(n)),
        Cv(static_cast<size_t>(n)), Aw(static_cast<size_t>(n));
    std::mt19937 rng = make_rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) v[static_cast<size_t>(i)] = dist(rng);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            w[static_cast<size_t>(i)] += B[i * n + j] * v[static_cast<size_t>(j)];
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            Aw[static_cast<size_t>(i)] += A[i * n + k] * w[static_cast<size_t>(k)];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            Cv[static_cast<size_t>(i)] += C[i * n + j] * v[static_cast<size_t>(j)];
    for (int i = 0; i < n; ++i)
        if (!nearly_equal(Cv[static_cast<size_t>(i)], Aw[static_cast<size_t>(i)], 1e-2f))
            return false;
    return true;
}

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
#ifdef HAVE_CBLAS
    std::fprintf(stderr, "[gemm] using optimized CBLAS\n");
#else
    if (has_arg(argc, argv, "--require-blas")) {
        std::fprintf(stderr,
                     "ERROR: --require-blas was set but no optimized CBLAS is available.\n"
                     "       The fallback GEMM must not be used for CPU-vs-GPU teaching.\n"
                     "       Install OpenBLAS (e.g. 'sudo apt-get install libopenblas-dev') and rebuild.\n");
        std::exit(1);
    }
    std::fprintf(stderr, "[gemm] WARNING: no BLAS found — using slow tiled fallback\n");
#endif

    for (int n : sizes) {
        const size_t nelem = static_cast<size_t>(n) * static_cast<size_t>(n);
        std::vector<float> A = gen_lcg_floats(static_cast<long>(nelem), 1000u);
        std::vector<float> B = gen_lcg_floats(static_cast<long>(nelem), 2000u);
        std::vector<float> C(nelem, 0.0f);

        auto run = [&] { gemm(n, A.data(), B.data(), C.data()); };

        run();
        const bool ok = gemm_check(n, A.data(), B.data(), C.data());
        report_check("gemm correctness n=" + std::to_string(n), ok);
        if (!ok) {
            std::fprintf(stderr, "  C != A*B for n=%d\n", n);
            std::exit(2);
        }
        sink(C[0]);

        const Stats st = run_benchmark(run, warmup, iters);
        const double t = st.median;
        const double gflops = 2.0 * std::pow(static_cast<double>(n), 3) / t / 1e9;
        std::fprintf(stderr, "[gemm] n=%d: median=%.4f ms  %.3f GFLOPS\n", n, t * 1e3, gflops);

        if (human_format(argc, argv)) {
            HumanReport r;
            r.title = "GEMM (CPU): n=" + std::to_string(n);
            r.add("Platform", "CPU");
            r.add("Matrix size", std::to_string(n) + " x " + std::to_string(n));
            r.add("Latency", fmt(t * 1e3) + " ms");
            r.add("GFLOPS", fmt(gflops, 3));
            r.print();
        } else {
            emit(csv_file, {"CPU", std::to_string(n), fmt(t * 1e3), fmt(gflops, 3)});
        }
    }
    return 0;
}
