# VerseNext 自研算子 —— CMake 构建（CUDA / ROCm）
#
# 适用场景：需要把算子作为独立 .so 集成进更大的 CMake 工程，或用
# find_package(Torch) 而非 Python cpp_extension 时。
# 日常开发直接用 `bash build_cuda.sh` / `build_rocm.sh` 更省事（自动探测 arch）。
#
# 用法：
#   cmake -B build -DCMAKE_PREFIX_PATH=$(python -c "import torch;print(torch.utils.cmake_prefix_path)")
#   cmake --build build -j
#
# ROCm 下 CMAKE 会走 hipcc，源码仍为同一份 csrc/cuda/*.cu（由 hip 工具链处理）。

cmake_minimum_required(VERSION 3.18)
project(verse_nn_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Torch REQUIRED)

# 判定 CUDA 还是 ROCm：ROCm 版 torch 会定义 USE_ROCM
if(DEFINED USE_ROCM OR TORCH_HIP_VERSION)
  set(VERSE_BACKEND "rocm")
else()
  set(VERSE_BACKEND "cuda")
endif()
message(STATUS "VerseNext kernels backend: ${VERSE_BACKEND}")

set(VERSE_SOURCES
  cuda/fused_rmsnorm.cu
  cuda/fused_swiglu.cu
  cuda/fused_rope.cu
  cuda/flash_attn.cu
  cuda/chunked_ce.cu
  cuda/kda_chunk.cu
  cuda/bindings.cpp
)

add_library(verse_nn_kernels SHARED ${VERSE_SOURCES})
target_include_directories(verse_nn_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(verse_nn_kernels PRIVATE ${TORCH_LIBRARIES})

if(VERSE_BACKEND STREQUAL "cuda")
  set_target_properties(verse_nn_kernels PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
  )
  # 常见架构；如需精简可只留目标卡
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
  target_compile_options(verse_nn_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math -lineinfo>)
else()
  target_compile_definitions(verse_nn_kernels PRIVATE __HIP_PLATFORM_AMD__)
  target_compile_options(verse_nn_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx90a --offload-arch=gfx942 -O3>)
endif()

# 产物直接放到 _build/<backend>/，供 verse_nn.kernels._ext 探测
set_target_properties(verse_nn_kernels PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../_build/${VERSE_BACKEND}"
  PREFIX ""
  OUTPUT_NAME "verse_nn_kernels"
)
