cmake_minimum_required(VERSION 3.18)
project(lecture01_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- Optional CUDA (GPU binaries are only built when nvcc is available) -----
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
endif()

find_package(Threads REQUIRED)

# --- Optional BLAS for ai_gemm_cpu ------------------------------------------
set(GEMM_BLAS_DEFS "")
set(GEMM_BLAS_LIBS "")
if(APPLE)
  set(GEMM_BLAS_DEFS USE_CBLAS)
  set(GEMM_BLAS_LIBS "-framework Accelerate")
else()
  find_path(CBLAS_H_DIR cblas.h)
  find_package(BLAS QUIET)
  if(CBLAS_H_DIR AND BLAS_FOUND)
    set(GEMM_BLAS_DEFS USE_CBLAS)
    set(GEMM_BLAS_LIBS ${BLAS_LIBRARIES})
  endif()
endif()

# --- CPU binaries ------------------------------------------------------------
foreach(tgt
    exp1_cpu:exp1_single_thread/cpu.cpp
    exp2_cpu_memory:exp2_mapping/cpu_memory.cpp
    exp2_cpu_control:exp2_mapping/cpu_control.cpp
    exp3_cpu:exp3_scaling/cpu.cpp)
  string(REPLACE ":" ";" tgt_parts "${tgt}")
  list(GET tgt_parts 0 name)
  list(GET tgt_parts 1 src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE Threads::Threads)
endforeach()

add_executable(ai_gemm_cpu ai_gemm/cpu_gemm.cpp)
target_link_libraries(ai_gemm_cpu PRIVATE Threads::Threads ${GEMM_BLAS_LIBS})
if(GEMM_BLAS_DEFS)
  target_compile_definitions(ai_gemm_cpu PRIVATE ${GEMM_BLAS_DEFS})
endif()

# --- GPU binaries (only if CUDA is available) --------------------------------
if(CMAKE_CUDA_COMPILER)
  add_executable(exp1_gpu exp1_single_thread/gpu.cu)
  add_executable(exp2_gpu_memory exp2_mapping/gpu_memory.cu)
  add_executable(exp2_gpu_control exp2_mapping/gpu_control.cu)
  add_executable(exp3_gpu exp3_scaling/gpu.cu)
  add_executable(ai_gemm_gpu ai_gemm/gpu_gemm.cu)
  target_link_libraries(ai_gemm_gpu PRIVATE cublas)
endif()

# Put binaries in build/bin (matches the run scripts' default path when built
# with -DCMAKE_RUNTIME_OUTPUT_DIRECTORY).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
