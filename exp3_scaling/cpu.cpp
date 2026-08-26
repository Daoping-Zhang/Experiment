// ============================================================================
// Experiment 3 — CPU thread scaling
//
// Fully independent, compute-heavy element kernel with the most regular
// mapping possible (block partitioning, no branches):
//   output[i] = F(input[i]),  F: x = x*a + b repeated K times
//
// A single invocation measures ONE thread count; scripts/run_exp3.sh sweeps
// 1,2,4,...,logical_hw_threads and appends each row.
//
// Usage:
//   ./exp3_cpu --threads 8 --size 1000000 --compute-iterations 500
// ============================================================================
#include "common.h"

#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

__attribute__((noinline)) float element_kernel(float v, int K, float a, float b) {
    float x = v;
    for (int k = 0; k < K; ++k) x = x * a + b;
    return x;
}

void run_worker(const float* in, float* out, long begin, long end, int K, float a, float b) {
    for (long i = begin; i < end; ++i)
        out[i] = element_kernel(in[i], K, a, b);
}

const std::vector<std::string> HEADER = {
    "platform", "threads", "size", "k", "latency_ms", "throughput_elem_s", "gflops"};

void emit(const std::string& csv_file, const std::vector<std::string>& row) {
    print_csv_row(row);
    if (!csv_file.empty()) CsvSink(csv_file, HEADER).write(row);
}

}  // namespace

int main(int argc, char** argv) {
    const int T = get_arg_int(argc, argv, "--threads", 1);
    const long N = get_arg_long(argc, argv, "--size", 1000000L);
    const int K = get_arg_int(argc, argv, "--compute-iterations", 500);
    const float a = static_cast<float>(get_arg_double(argc, argv, "--a", 0.5));
    const float b = static_cast<float>(get_arg_double(argc, argv, "--b", 1.0));
    const int warmup = get_arg_int(argc, argv, "--warmup", 3);
    const int iters = get_arg_int(argc, argv, "--iters", 10);
    const std::string csv_file = get_arg(argc, argv, "--csv-file", "");

    std::vector<float> in = gen_lcg_floats(N, 4242u);
    std::vector<float> out(static_cast<size_t>(N), 0.0f);

    auto run = [&] {
        std::vector<std::thread> pool;
        pool.reserve(T);
        for (int t = 0; t < T; ++t) {
            const long begin = t * N / T;
            const long end = (t + 1) * N / T;
            pool.emplace_back([&, begin, end] { run_worker(in.data(), out.data(), begin, end, K, a, b); });
        }
        for (auto& th : pool) th.join();
    };

    // Correctness: full check against a double-precision reference.
    run();
    bool ok = true;
    for (long i = 0; i < N && ok; ++i) {
        double r = in[static_cast<size_t>(i)];
        for (int k = 0; k < K; ++k) r = r * static_cast<double>(a) + static_cast<double>(b);
        ok = nearly_equal(out[static_cast<size_t>(i)], static_cast<float>(r), 1e-2f);
    }
    report_check("cpu element kernel (threads=" + std::to_string(T) + ")", ok);
    if (!ok) {
        std::fprintf(stderr, "  out[i] != reference at some index\n");
        std::exit(2);
    }
    sink(out[N / 2]);

    const Stats st = run_benchmark(run, warmup, iters);
    const double t = st.median;
    const double throughput = static_cast<double>(N) / t;
    const double gflops = 2.0 * static_cast<double>(N) * static_cast<double>(K) / t / 1e9;

    std::fprintf(stderr, "[cpu] threads=%d: median=%.4f ms  %.3e elem/s  %.3f GFLOPS\n", T, t * 1e3,
                throughput, gflops);

    if (human_format(argc, argv)) {
        HumanReport r;
        r.title = "Experiment 3 (CPU): " + std::to_string(T) + " threads";
        r.add("Platform", "CPU");
        r.add("Threads", std::to_string(T));
        r.add("Elements", std::to_string(N));
        r.add("Compute iterations/elem", std::to_string(K));
        r.add("Latency", fmt(t * 1e3) + " ms");
        r.add("Throughput", fmt(throughput, 3) + " elem/s");
        r.add("GFLOPS", fmt(gflops, 3));
        r.print();
    } else {
        emit(csv_file,
             {"CPU", std::to_string(T), std::to_string(N), std::to_string(K), fmt(t * 1e3),
              fmt(throughput, 3), fmt(gflops, 3)});
    }
    return 0;
}
