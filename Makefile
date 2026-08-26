# ============================================================================
# Lecture 01 Tutorial — Makefile
#
#   make          -> build everything available (CPU always; GPU if nvcc found)
#   make cpu      -> build CPU binaries only
#   make gpu      -> build GPU binaries only (requires CUDA toolkit)
#   make clean    -> remove bin/
#
# Binaries are placed in ./bin . Run scripts (scripts/run_*.sh) expect them
# there. Override CUDA_ARCH (e.g. `make gpu CUDA_ARCH=-arch=sm_86`) if
# `-arch=native` is not supported by your nvcc.
# ============================================================================
BIN := bin
CXX := g++
CXXFLAGS := -O3 -std=c++17 -pthread -Wall
# Detect nvcc: PATH first, then the standard CUDA install location.
NVCC := $(shell command -v nvcc 2>/dev/null || { [ -x /usr/local/cuda/bin/nvcc ] && echo /usr/local/cuda/bin/nvcc; })
NVCCFLAGS := -O3 -std=c++17
CUDA_ARCH ?= -arch=native
BLAS_FLAGS := $(shell bash scripts/detect_blas.sh)

CPU_BINS := $(BIN)/exp1_cpu $(BIN)/exp2_cpu_memory $(BIN)/exp2_cpu_control \
            $(BIN)/exp3_cpu $(BIN)/ai_gemm_cpu
GPU_BINS := $(BIN)/exp1_gpu $(BIN)/exp2_gpu_memory $(BIN)/exp2_gpu_control \
            $(BIN)/exp3_gpu $(BIN)/ai_gemm_gpu

.PHONY: all cpu gpu clean

all: cpu $(if $(NVCC),gpu,)
	@echo "Built CPU binaries: $(CPU_BINS)"
	@if [ -z "$(NVCC)" ]; then \
		echo "NOTE: nvcc not found -> GPU binaries not built (install CUDA toolkit)."; \
	else \
		echo "Built GPU binaries: $(GPU_BINS)"; \
	fi

cpu: $(CPU_BINS)

gpu:
	@if [ -z "$(NVCC)" ]; then \
		echo "NOTE: nvcc not found -> GPU binaries not built (install CUDA toolkit)."; \
	else \
		$(MAKE) $(GPU_BINS); \
	fi

$(BIN):
	mkdir -p $(BIN)

$(BIN)/exp1_cpu: exp1_single_thread/cpu.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN)/exp2_cpu_memory: exp2_mapping/cpu_memory.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN)/exp2_cpu_control: exp2_mapping/cpu_control.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN)/exp3_cpu: exp3_scaling/cpu.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN)/ai_gemm_cpu: ai_gemm/cpu_gemm.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(BLAS_FLAGS) $< -o $@

$(BIN)/exp1_gpu: exp1_single_thread/gpu.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $< -o $@

$(BIN)/exp2_gpu_memory: exp2_mapping/gpu_memory.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $< -o $@

$(BIN)/exp2_gpu_control: exp2_mapping/gpu_control.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $< -o $@

$(BIN)/exp3_gpu: exp3_scaling/gpu.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $< -o $@

$(BIN)/ai_gemm_gpu: ai_gemm/gpu_gemm.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $< -lcublas -o $@

clean:
	rm -rf $(BIN)
