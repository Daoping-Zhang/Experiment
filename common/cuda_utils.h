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
