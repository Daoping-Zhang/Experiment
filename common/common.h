// ============================================================================
// Lecture 01 Tutorial — shared CPU utilities
//
// Timing, statistics (median/mean/stddev), CLI argument parsing, CSV output,
// a volatile sink to defeat dead-code elimination, deterministic RNG and
// common correctness helpers.
//
// This header is CPU-only and must compile under both g++ (Linux) and clang++
// (macOS). CUDA-only helpers live in common/cuda_utils.h.
// ============================================================================
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
inline double now_seconds() {
    using clock_t = std::chrono::steady_clock;
    static const auto t0 = clock_t::now();
    return std::chrono::duration<double>(clock_t::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------
struct Stats {
    double median = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
    std::vector<double> samples;
};

inline Stats compute_stats(std::vector<double> s) {
    Stats st;
    if (s.empty()) return st;
    std::sort(s.begin(), s.end());
    st.min = s.front();
    st.max = s.back();
    st.mean = std::accumulate(s.begin(), s.end(), 0.0) / static_cast<double>(s.size());
    double var = 0.0;
    for (double v : s) var += (v - st.mean) * (v - st.mean);
    st.stddev = std::sqrt(var / static_cast<double>(s.size()));
    const size_t n = s.size();
    st.median = (n % 2 == 1) ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
    st.samples = std::move(s);
    return st;
}

// Run `fn` `warmup` times untimed, then `iters` times timed; return statistics
// over the timed samples. Median is the figure of merit used in the report.
template <typename F>
Stats run_benchmark(F&& fn, int warmup, int iters) {
    std::vector<double> ts;
    ts.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < warmup; ++i) fn();
    for (int i = 0; i < iters; ++i) {
        const double t0 = now_seconds();
        fn();
        const double t1 = now_seconds();
        ts.push_back(t1 - t0);
    }
    return compute_stats(std::move(ts));
}

// ---------------------------------------------------------------------------
// Volatile sink — force the compiler to treat a value as observable so the
// benchmark body cannot be removed by dead-code elimination.
// ---------------------------------------------------------------------------
template <typename T>
inline void sink(const T& v) {
    static volatile T s;
    s = v;
    (void)s;  // volatile read: observable side effect, also silences -Wunused-but-set-variable
}

// ---------------------------------------------------------------------------
// CLI argument parsing (--key value)
// ---------------------------------------------------------------------------
inline std::string get_arg(int argc, char** argv, const std::string& key,
                           const std::string& def = "") {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == key) return argv[i + 1];
    }
    return def;
}

inline bool has_arg(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == key) return true;
    }
    return false;
}

inline long get_arg_long(int argc, char** argv, const std::string& key, long def) {
    const std::string v = get_arg(argc, argv, key);
    if (v.empty()) return def;
    return std::stoll(v);
}

inline int get_arg_int(int argc, char** argv, const std::string& key, int def) {
    const std::string v = get_arg(argc, argv, key);
    if (v.empty()) return def;
    return std::stoi(v);
}

inline double get_arg_double(int argc, char** argv, const std::string& key, double def) {
    const std::string v = get_arg(argc, argv, key);
    if (v.empty()) return def;
    return std::stod(v);
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------
class CsvSink {
public:
    std::string path;
    std::vector<std::string> header;
    bool header_written = false;

    CsvSink() = default;
    CsvSink(const std::string& p, const std::vector<std::string>& h)
        : path(p), header(h) {
        if (p.empty()) return;
        std::ifstream f(p);
        header_written = f.good() && f.peek() != std::ifstream::traits_type::eof();
    }

    void write(const std::vector<std::string>& row) {
        if (path.empty()) return;
        std::ofstream f(path, std::ios::app);
        if (!f.is_open()) {
            std::fprintf(stderr, "[error] cannot open CSV file: %s\n", path.c_str());
            return;
        }
        if (!header_written) {
            for (size_t i = 0; i < header.size(); ++i) {
                if (i) f << ",";
                f << header[i];
            }
            f << "\n";
            header_written = true;
        }
        for (size_t i = 0; i < row.size(); ++i) {
            if (i) f << ",";
            f << row[i];
        }
        f << "\n";
    }
};

// Join a vector of strings into a single CSV line and print to stdout. Every
// benchmark binary prints exactly one such line so run scripts / perf wrappers
// can capture it.
inline void print_csv_row(const std::vector<std::string>& row) {
    for (size_t i = 0; i < row.size(); ++i) {
        if (i) std::printf(",");
        std::printf("%s", row[i].c_str());
    }
    std::printf("\n");
    std::fflush(stdout);
}

inline std::string fmt(double v, int precision = 6) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

inline std::string fmt(long v) {
    return std::to_string(v);
}

// ---------------------------------------------------------------------------
// Human-readable report (used with `--format human` for live classroom demos).
// `--format csv` (default) keeps the single-line CSV contract used by the
// run scripts and perf wrapper.
// ---------------------------------------------------------------------------
class HumanReport {
public:
    std::string title;
    std::vector<std::pair<std::string, std::string>> fields;

    void add(const std::string& k, const std::string& v) { fields.emplace_back(k, v); }

    void print(FILE* out = stdout) const {
        size_t w = 0;
        for (const auto& kv : fields) w = std::max(w, kv.first.size());
        std::fprintf(out, "\n=== %s ===\n\n", title.c_str());
        for (const auto& kv : fields)
            std::fprintf(out, "  %-*s : %s\n", static_cast<int>(w), kv.first.c_str(),
                         kv.second.c_str());
        std::fprintf(out, "\n");
    }
};

inline std::string get_format(int argc, char** argv) {
    return get_arg(argc, argv, "--format", "csv");
}

inline bool human_format(int argc, char** argv) {
    return get_format(argc, argv) == "human";
}

// ---------------------------------------------------------------------------
// Deterministic RNG
// ---------------------------------------------------------------------------
inline std::mt19937 make_rng(uint32_t seed) {
    return std::mt19937(seed);
}

// Generate `n` floats uniformly distributed in [lo, hi) with a fixed seed so
// repeated runs produce identical datasets.
inline std::vector<float> gen_random_floats(long n, uint32_t seed,
                                            float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(static_cast<size_t>(n));
    for (long i = 0; i < n; ++i) v[static_cast<size_t>(i)] = dist(rng);
    return v;
}

// Fast deterministic fill for very large arrays (LCG). Much cheaper than
// std::mt19937 when N is in the tens of millions, and still reproducible.
inline std::vector<float> gen_lcg_floats(long n, uint32_t seed) {
    std::vector<float> v(static_cast<size_t>(n));
    uint32_t x = seed;
    for (long i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        v[static_cast<size_t>(i)] = static_cast<float>(x & 0xFFFF) * (1.0f / 65536.0f);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Correctness helpers
// ---------------------------------------------------------------------------
inline bool nearly_equal(float a, float b, float eps = 1e-3f) {
    const float diff = std::fabs(a - b);
    const float scale = std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
    return diff <= eps * scale;
}

// Print a pass/fail correctness line and return the boolean result.
inline bool report_check(const std::string& name, bool ok) {
    std::fprintf(stderr, "[check] %-40s %s\n", name.c_str(), ok ? "PASS" : "FAILED");
    return ok;
}
