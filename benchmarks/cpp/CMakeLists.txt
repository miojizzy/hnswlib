# CMakeLists.txt
cmake_minimum_required(VERSION 3.11)

include(FetchContent)


# close benchmark-test
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
# Google Benchmark
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.9.4
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(benchmark)


# master branch
FetchContent_Declare(
  hnswlib_std
  GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
  GIT_TAG        v0.8.0
  GIT_SHALLOW    TRUE
)

# 手动处理版本
FetchContent_GetProperties(hnswlib_std)
if(NOT hnswlib_std_POPULATED)
  FetchContent_Populate(hnswlib_std)
  
  # 创建一个带有不同前缀的目标
  add_library(hnswlib_std INTERFACE)
  add_library(hnswlib_std::hnswlib ALIAS hnswlib_std)
  
  target_include_directories(hnswlib_std INTERFACE 
    $<BUILD_INTERFACE:${hnswlib_std_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)
endif()

# 创建你的可执行文件
add_executable(benchmark_standard main.cpp
  bm_basic.cpp
)
target_link_libraries(benchmark_standard benchmark::benchmark)
target_link_libraries(benchmark_standard hnswlib_std::hnswlib)

add_executable(benchmark_current main.cpp
  bm_basic.cpp
)
target_link_libraries(benchmark_current benchmark::benchmark)
target_link_libraries(benchmark_current hnswlib)