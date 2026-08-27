// ============================================================================
// Experiment 1 — CPU single thread
//
// One software thread. Three workloads:
//   --case dependent     : one long data-dependent chain  x = x*A + B
//   --case independent   : C independent chains (ILP)     --chains C
//   --case branch        : branchy loop over a dataset    --data predictable|random
//
// Application-level metrics (latency / throughput / GFLOPS) are measured
// here. Hardware counters (cycles / instructions / branches / misses) are
// collected by `perf stat` on Linux and merged by scripts/perf_merge.py;
// without perf they remain "NA".
//
// Usage:
//   ./exp1_cpu --case dependent --iterations 100000000
//   ./exp1_cpu --case independent --chains 4 --iterations 100000000
//   ./exp1_cpu --case branch --data random --iterations 10000000
// ============================================================================
#include "common.h"

#include <cstdio>
#include <functional>
#include <string>

namespace {

constexpr int BRANCH_INNER = 8;  // inner dependent FMA iterations per element

// x = x*A + B repeated N times (single chain)
float chain_f32(long N, float A, float B) {
    float x = 1.0f;
    for (long i = 0; i < N; ++i) x = x * A + B;
    return x;
}

double chain_f64(long N, double A, double B) {
    double x = 1.0;
    for (long i = 0; i < N; ++i) x = x * A + B;
    return x;
}

// C independent chains, `per` iterations each; returns the sum of the C tails.
// Templated so C is a compile-time constant: the compiler unrolls the inner
// loop and keeps the accumulators in registers, exposing ILP (the whole point
// of Experiment 1B). A runtime-sized array would stay in memory and hide ILP.
template <int C>
float independent_f32(long per, float A, float B) {
    float x[C];
    for (int c = 0; c < C; ++c) x[c] = 1.0f;
    for (long i = 0; i < per; ++i) {
        for (int c = 0; c < C; ++c) x[c] = x[c] * A + B;
    }
    float s = 0.0f;
    for (int c = 0; c < C; ++c) s += x[c];
    return s;
}

float independent_f32_dispatch(long per, int C, float A, float B) {
    switch (C) {
        case 1: return independent_f32<1>(per, A, B);
        case 2: return independent_f32<2>(per, A, B);
        case 4: return independent_f32<4>(per, A, B);
        case 8: return independent_f32<8>(per, A, B);
        default:
            std::fprintf(stderr, "chains must be 1/2/4/8\n");
            std::exit(1);
    }
}

double independent_f64(long per, int C, double A, double B) {
    std::vector<double> x(static_cast<size_t>(C), 1.0);
    for (long i = 0; i < per; ++i) {
        for (int c = 0; c < C; ++c) x[static_cast<size_t>(c)] = x[static_cast<size_t>(c)] * A + B;
    }
    double s = 0.0;
    for (int c = 0; c < C; ++c) s += x[static_cast<size_t>(c)];
    return s;
}

// Two branches of EQUAL cost (same number of FMA iterations) but DIFFERENT
// arithmetic constants. The bodies must differ: otherwise the compiler's
// identical-code-folding merges them into one function and the `if/else`
// collapses into an unconditional computation, removing the branch entirely.
// `noinline` keeps them as real calls so a conditional branch survives.
__attribute__((noinline)) float branch_taken(float v) {
    float r = v;
    for (int k = 0; k < BRANCH_INNER; ++k) r = r * 0.99f + 0.01f;
    return r;
}

__attribute__((noinline)) float branch_not_taken(float v) {
    float r = v;
    for (int k = 0; k < BRANCH_INNER; ++k) r = r * 0.98f + 0.02f;
    return r;
}

// Double-precision reference mirroring the branch semantics exactly.
double branch_ref(const std::vector<float>& data) {
    double x = 0.0;
    for (float v : data) {
        double r = v;
        const double a = (v > 0.0f) ? 0.99 : 0.98;
        const double b = (v > 0.0f) ? 0.01 : 0.02;
        for (int k = 0; k < BRANCH_INNER; ++k) r = r * a + b;
        x += r;
    }
    return x;
}

float branch_sum(const float* data, long N) {
    float x = 0.0f;
    for (long i = 0; i < N; ++i) {
        if (data[i] > 0.0f)
            x += branch_taken(data[i]);
        else
            x += branch_not_taken(data[i]);
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

// Common report for the chain workloads.
void run_chain_case(int argc, char** argv) {
    const std::string workload = get_arg(argc, argv, "--case", "dependent");
    const long N = get_arg_long(argc, argv, "--iterations", 100000000L);
    const int chains = get_arg_int(argc, argv, "--chains", 1);
    const float A = static_cast<float>(get_arg_double(argc, argv, "--a", 0.5));
    const float B = static_cast<float>(get_arg_double(argc, argv, "--b", 1.0));
    const int warmup = get_arg_int(argc, argv, "--warmup", 3);
    const int iters = get_arg_int(argc, argv, "--iters", 10);
    const std::string csv_file = get_arg(argc, argv, "--csv-file", "");

    long eff_iterations = N;
    float result = 0.0f;
    double reference = 0.0;
    std::string variant;

    if (workload == "dependent") {
        variant = "dependent";
        eff_iterations = N;
        reference = chain_f64(N, A, B);
        result = chain_f32(N, A, B);
    } else if (workload == "independent") {
        variant = "chains=" + std::to_string(chains);
        const long per = N / chains;
        eff_iterations = per * chains;
        reference = independent_f64(per, chains, A, B);
        result = independent_f32_dispatch(per, chains, A, B);
    } else {
        std::fprintf(stderr, "unknown --case for chain workload: %s\n", workload.c_str());
        std::exit(1);
    }

    const bool ok = nearly_equal(result, static_cast<float>(reference), 5e-3f);
    report_check(workload + " (float vs double reference)", ok);
    if (!ok) {
        std::fprintf(stderr, "  result=%.6f reference=%.6f\n", result, static_cast<float>(reference));
        std::exit(2);
    }
    sink(result);

    std::function<float()> fn;
    if (workload == "dependent")
        fn = [&] { return chain_f32(N, A, B); };
    else {
        const long per = N / chains;
        fn = [&] { return independent_f32_dispatch(per, chains, A, B); };
    }

    const Stats st = run_benchmark(fn, warmup, iters);
    const double t = st.median;  // seconds
    const double throughput = static_cast<double>(eff_iterations) / t;
    const double gflops = 2.0 * static_cast<double>(eff_iterations) / t / 1e9;

    std::fprintf(stderr, "[cpu] %s %s: median=%.6f ms  throughput=%.3e iter/s  %.3f GFLOPS\n",
                workload.c_str(), variant.c_str(), t * 1e3, throughput, gflops);

    if (human_format(argc, argv)) {
        HumanReport r;
        r.title = (workload == "dependent")
                      ? "Experiment 1A: Dependent Chain"
                      : "Experiment 1B: Independent Chains (" + variant + ")";
        r.add("Platform", "CPU");
        r.add("Threads", "1");
        r.add("Iterations", std::to_string(eff_iterations));
        r.add("Latency", fmt(t * 1e3) + " ms");
        r.add("Throughput", fmt(throughput, 3) + " iter/s");
        r.add("GFLOPS", fmt(gflops, 3));
        r.print();
    } else {
        emit(csv_file,
             {"CPU", workload, variant, "1", std::to_string(eff_iterations), fmt(t * 1e3),
              fmt(throughput, 3), fmt(gflops, 3),
              "NA", "NA", "NA", "NA", "NA", "NA",
              "NA", fmt(t * 1e3)});
    }
}

void run_branch_case(int argc, char** argv) {
    const long N = get_arg_long(argc, argv, "--iterations", 10000000L);
    const std::string data_mode = get_arg(argc, argv, "--data", "predictable");
    const int warmup = get_arg_int(argc, argv, "--warmup", 3);
    const int iters = get_arg_int(argc, argv, "--iters", 10);
    const std::string csv_file = get_arg(argc, argv, "--csv-file", "");

    std::vector<float> data(static_cast<size_t>(N));
    if (data_mode == "predictable") {
        std::fill(data.begin(), data.end(), 1.0f);  // all > 0 → always taken
    } else if (data_mode == "random") {
        std::mt19937 rng = make_rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (long i = 0; i < N; ++i) data[static_cast<size_t>(i)] = dist(rng);
    } else {
        std::fprintf(stderr, "unknown --data: %s\n", data_mode.c_str());
        std::exit(1);
    }

    const float result = branch_sum(data.data(), N);
    sink(result);
    const double reference = branch_ref(data);
    const bool ok = nearly_equal(result, static_cast<float>(reference), 5e-3f);
    report_check("cpu branch (" + data_mode + ") vs double reference", ok);
    if (!ok) {
        std::fprintf(stderr, "  result=%.6f reference=%.6f\n", result, static_cast<float>(reference));
        std::exit(2);
    }
    std::fprintf(stderr, "[cpu] branch(%s): result=%.3f (vs reference %.3f)\n", data_mode.c_str(),
                 result, static_cast<float>(reference));

    std::function<float()> fn = [&] { return branch_sum(data.data(), N); };
    const Stats st = run_benchmark(fn, warmup, iters);
    const double t = st.median;
    const double throughput = static_cast<double>(N) / t;
    const double gflops = 2.0 * static_cast<double>(N) * BRANCH_INNER / t / 1e9;

    std::fprintf(stderr, "[cpu] branch(%s): median=%.6f ms  throughput=%.3e elem/s  %.3f GFLOPS\n",
                data_mode.c_str(), t * 1e3, throughput, gflops);

    if (human_format(argc, argv)) {
        HumanReport r;
        r.title = "Experiment 1C: Branch Prediction (" + data_mode + ")";
        r.add("Platform", "CPU");
        r.add("Threads", "1");
        r.add("Elements", std::to_string(N));
        r.add("Latency", fmt(t * 1e3) + " ms");
        r.add("Throughput", fmt(throughput, 3) + " elem/s");
        r.add("GFLOPS", fmt(gflops, 3));
        r.print();
    } else {
        emit(csv_file,
             {"CPU", "branch", data_mode, "1", std::to_string(N), fmt(t * 1e3),
              fmt(throughput, 3), fmt(gflops, 3),
              "NA", "NA", "NA", "NA", "NA", "NA",
              "NA", fmt(t * 1e3)});
    }
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
