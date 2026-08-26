// ============================================================================
// Lecture 01 Tutorial — shared CUDA utilities
//
// Included only by .cu files. Provides CUDA error checking, CUDA-event based
// kernel timing, and a helper to build a permutation index array for the
// strided (uncoalesced) memory-access experiment.
// ============================================================================
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d -> %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(_e));                   \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA-event timer: measures pure kernel time on the GPU (launch/sync cost is
// intentionally excluded so kernel latency can be reported separately).
// ---------------------------------------------------------------------------
struct CudaEventTimer {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    bool ok = false;

    CudaEventTimer() {
        if (cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess)
            ok = true;
    }
    ~CudaEventTimer() {
        if (ok) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }
    void begin() { cudaEventRecord(start, 0); }
    void end() { cudaEventRecord(stop, 0); }
    float elapsed_ms() {
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ---------------------------------------------------------------------------
// Build a permutation `perm[0..N-1]` that maps every logical position to a
// unique element index (a bijection), so total FLOPs and bytes are identical
// to the coalesced case, but consecutive logical positions (i.e. consecutive
// threads inside a warp) land `stride` elements apart in memory.
//
// Guaranteed a full permutation for ANY (N, stride): it walks the additive
// cycle `addr -> (addr + stride) mod N` and restarts at the next unused index
// whenever a cycle closes early (which happens when gcd(stride, N) > 1).
// ---------------------------------------------------------------------------
inline std::vector<int> make_stride_permutation(long N, long stride) {
    std::vector<int> perm(static_cast<size_t>(N));
    std::vector<char> used(static_cast<size_t>(N), 0);
    long filled = 0;
    long start = 0;
    while (filled < N) {
        while (start < N && used[static_cast<size_t>(start)]) ++start;
        if (start >= N) break;
        long addr = start;
        for (;;) {
            if (used[static_cast<size_t>(addr)]) break;
            used[static_cast<size_t>(addr)] = 1;
            perm[static_cast<size_t>(filled++)] = static_cast<int>(addr);
            addr = (addr + stride) % N;
        }
    }
    if (filled != N) {
        std::fprintf(stderr, "make_stride_permutation failed (filled=%ld, N=%ld)\n", filled, N);
        std::exit(1);
    }
    return perm;
}
