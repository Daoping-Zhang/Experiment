// ============================================================================
// Experiment 2A — CPU memory mapping
//
// C[i] = A[i] + B[i] with a FIXED thread count, only the work partition changes:
//   --mapping block   : contiguous [begin, end) per thread (good locality)
//   --mapping cyclic  : stride-T loop per thread     (poor locality)
//
// Hardware counters (cycles / cache / LLC) are collected by perf on Linux and
// merged by scripts/perf_merge.py; they stay "NA" elsewhere.
//
// Usage:
//   ./exp2_cpu_memory --mapping block --threads 8 --size 100000000
// ============================================================================
#include "common.h"

#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

void vec_add_block(const float* A, const float* B, float* C, long N, int T) {
    std::vector<std::thread> pool;
    pool.reserve(T);
    for (int t = 0; t < T; ++t) {
        const long begin = t * N / T;
        const long end = (t + 1) * N / T;
        pool.emplace_back([=] {
            for (long i = begin; i < end; ++i) C[i] = A[i] + B[i];
        });
    }
    for (auto& th : pool) th.join();
}

void vec_add_cyclic(const float* A, const float* B, float* C, long N, int T) {
    std::vector<std::thread> pool;
    pool.reserve(T);
    for (int t = 0; t < T; ++t) {
        pool.emplace_back([=] {
            for (long i = t; i < N; i += T) C[i] = A[i] + B[i];
        });
    }
    for (auto& th : pool) th.join();
}

const std::vector<std::string> HEADER = {
    "platform",          "mapping",          "threads", "size",
    "latency_ms",        "elements_per_s",   "effective_bandwidth_gbs",
    "cycles",            "instructions",     "cache_references",
    "cache_misses",      "llc_loads",        "llc_load_misses"};

void emit(const std::string& csv_file, const std::vector<std::string>& row) {
    print_csv_row(row);
    if (!csv_file.empty()) CsvSink(csv_file, HEADER).write(row);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mapping = get_arg(argc, argv, "--mapping", "block");
    const long N = get_arg_long(argc, argv, "--size", 100000000L);
    int T = get_arg_int(argc, argv, "--threads", -1);
    const int warmup = get_arg_int(argc, argv, "--warmup", 3);
    const int iters = get_arg_int(argc, argv, "--iters", 10);
    const std::string csv_file = get_arg(argc, argv, "--csv-file", "");

    const unsigned hw = std::thread::hardware_concurrency();
    if (T <= 0) T = static_cast<int>(std::min<unsigned>(8u, hw == 0 ? 1u : hw));

    // Runtime-generated data so the compiler cannot fold C[i]=const and drop
    // the two input streams.
    std::vector<float> A = gen_lcg_floats(N, 12345u);
    std::vector<float> B = gen_lcg_floats(N, 67890u);
    std::vector<float> C(static_cast<size_t>(N), 0.0f);

    auto run = [&] {
        if (mapping == "block")
            vec_add_block(A.data(), B.data(), C.data(), N, T);
        else if (mapping == "cyclic")
            vec_add_cyclic(A.data(), B.data(), C.data(), N, T);
        else {
            std::fprintf(stderr, "unknown --mapping: %s\n", mapping.c_str());
            std::exit(1);
        }
    };

    // Correctness (full check, done once, untimed).
    run();
    bool ok = true;
    for (long i = 0; i < N && ok; ++i) {
        ok = nearly_equal(C[static_cast<size_t>(i)], A[static_cast<size_t>(i)] + B[static_cast<size_t>(i)], 1e-4f);
    }
    report_check("cpu vector add (" + mapping + ")", ok);
    if (!ok) {
        std::fprintf(stderr, "  C[i] != A[i]+B[i] at some index\n");
        std::exit(2);
    }
    sink(C[N / 2]);

    const Stats st = run_benchmark(run, warmup, iters);
    const double t = st.median;
    const double elements_per_s = static_cast<double>(N) / t;
    // 12 bytes per element: read A(4B) + read B(4B) + write C(4B)
    const double bw_gbs = 12.0 * static_cast<double>(N) / t / 1e9;

    std::fprintf(stderr, "[cpu] %s: threads=%d median=%.3f ms  %.3e elem/s  %.3f GB/s\n", mapping.c_str(),
                T, t * 1e3, elements_per_s, bw_gbs);

    emit(csv_file,
         {"CPU", mapping, std::to_string(T), std::to_string(N), fmt(t * 1e3), fmt(elements_per_s, 3),
          fmt(bw_gbs, 3),
          "NA", "NA", "NA", "NA", "NA", "NA"});
    return 0;
}
