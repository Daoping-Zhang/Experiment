// ============================================================================
// Experiment 2B — CPU work/control mapping (load imbalance)
//
// Fixed thread count, fixed total work (50% heavy elements + 50% light
// elements). Only the spatial distribution of heavy/light elements changes:
//   --distribution grouped : first half heavy, second half light
//   --distribution mixed   : heavy/light alternate element by element
//
// With block partitioning, "grouped" hands some threads all-heavy blocks and
// others all-light blocks → load imbalance; "mixed" balances every thread.
//
// Usage:
//   ./exp2_cpu_control --distribution grouped --threads 8 --tasks 1048576
// ============================================================================
#include "common.h"

#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int HEAVY = 200;  // inner FMA iterations for a heavy element
constexpr int LIGHT = 20;   // inner FMA iterations for a light element

__attribute__((noinline)) float element_work(float v, int k) {
    float r = v;
    for (int i = 0; i < k; ++i) r = r * 0.99f + 0.01f;
    return r;
}

const std::vector<std::string> HEADER = {
    "platform",        "distribution",    "threads",       "tasks",
    "heavy_iters",     "light_iters",     "latency_ms",    "throughput_tasks_s",
    "max_thread_ms",   "min_thread_ms",   "avg_thread_ms", "load_imbalance_ratio"};

void emit(const std::string& csv_file, const std::vector<std::string>& row) {
    print_csv_row(row);
    if (!csv_file.empty()) CsvSink(csv_file, HEADER).write(row);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dist = get_arg(argc, argv, "--distribution", "grouped");
    const long N = get_arg_long(argc, argv, "--tasks", 1048576L);  // 2^20
    int T = get_arg_int(argc, argv, "--threads", -1);
    const int warmup = get_arg_int(argc, argv, "--warmup", 3);
    const int iters = get_arg_int(argc, argv, "--iters", 10);
    const std::string csv_file = get_arg(argc, argv, "--csv-file", "");

    const unsigned hw = std::thread::hardware_concurrency();
    if (T <= 0) T = static_cast<int>(std::min<unsigned>(8u, hw == 0 ? 1u : hw));

    std::vector<float> vals = gen_lcg_floats(N, 999u);
    std::vector<int> work(static_cast<size_t>(N));
    for (long i = 0; i < N; ++i) {
        const bool heavy = (dist == "grouped") ? (i < N / 2) : (i % 2 == 0);
        work[static_cast<size_t>(i)] = heavy ? HEAVY : LIGHT;
    }

    // Single-thread double-precision reference for correctness.
    double reference = 0.0;
    for (long i = 0; i < N; ++i) {
        double r = vals[static_cast<size_t>(i)];
        for (int k = 0; k < work[static_cast<size_t>(i)]; ++k) r = r * 0.99 + 0.01;
        reference += r;
    }

    std::vector<double> thread_times(static_cast<size_t>(T), 0.0);
    std::vector<float> partial(static_cast<size_t>(T), 0.0f);

    auto run = [&] {
        std::vector<std::thread> pool;
        pool.reserve(T);
        for (int t = 0; t < T; ++t) {
            const long begin = t * N / T;
            const long end = (t + 1) * N / T;
            pool.emplace_back([&, t, begin, end] {
                const double t0 = now_seconds();
                float acc = 0.0f;
                for (long i = begin; i < end; ++i)
                    acc += element_work(vals[static_cast<size_t>(i)], work[static_cast<size_t>(i)]);
                const double t1 = now_seconds();
                thread_times[static_cast<size_t>(t)] = (t1 - t0) * 1e3;
                partial[static_cast<size_t>(t)] = acc;
            });
        }
        for (auto& th : pool) th.join();
    };

    run();
    float sum = 0.0f;
    for (float p : partial) sum += p;
    const bool ok = nearly_equal(sum, static_cast<float>(reference), 5e-3f);
    report_check("cpu control (" + dist + ") vs reference", ok);
    if (!ok) {
        std::fprintf(stderr, "  sum=%.6f reference=%.6f\n", sum, static_cast<float>(reference));
        std::exit(2);
    }
    sink(sum);

    const Stats st = run_benchmark(run, warmup, iters);
    const double latency_ms = st.median * 1e3;
    const double throughput = static_cast<double>(N) / st.median;

    // Per-thread imbalance from the last (steady-state) timed run.
    const double mx = *std::max_element(thread_times.begin(), thread_times.end());
    const double mn = *std::min_element(thread_times.begin(), thread_times.end());
    const double avg = std::accumulate(thread_times.begin(), thread_times.end(), 0.0) / T;
    const double ratio = avg > 0.0 ? mx / avg : 0.0;

    std::fprintf(stderr, "[cpu] %s: latency=%.3f ms  %.3e tasks/s  max/avg/min=%.2f/%.2f/%.2f ms  "
                "imbalance=%.3f\n",
                dist.c_str(), latency_ms, throughput, mx, avg, mn, ratio);

    emit(csv_file,
         {"CPU", dist, std::to_string(T), std::to_string(N), std::to_string(HEAVY),
          std::to_string(LIGHT), fmt(latency_ms), fmt(throughput, 3), fmt(mx), fmt(mn), fmt(avg),
          fmt(ratio)});
    return 0;
}
