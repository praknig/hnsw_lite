cmake_minimum_required(VERSION 3.20)
project(hnsw-lite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HNSW_LITE_SANITIZE "Build with AddressSanitizer + UBSan (GCC/Clang, Linux/macOS)" OFF)

# ---------------------------------------------------------------------------
# Library: Layer 1 headers, Layer 2 kernels, Layer 3 indexes
# ---------------------------------------------------------------------------
file(GLOB HNSW_LITE_HEADERS CONFIGURE_DEPENDS include/*.h src/*.h tests/*.h)
add_library(hnsw_lite STATIC
        src/dispatch.cpp src/distance_scalar.cpp src/distance_neon.cpp   # Layer 2
        src/flat_index.cpp src/hnsw_index.cpp                            # Layer 3
        ${HNSW_LITE_HEADERS})
target_include_directories(hnsw_lite PUBLIC include)

# Each SIMD file gets only its own CPU flags; dispatch.cpp checks the CPU at runtime.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  target_sources(hnsw_lite PRIVATE src/distance_avx2.cpp src/distance_avx512.cpp)
  target_compile_definitions(hnsw_lite PRIVATE HNSW_LITE_HAVE_AVX2 HNSW_LITE_HAVE_AVX512)
  if(MSVC)
    set_source_files_properties(src/distance_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/distance_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  else()
    set_source_files_properties(src/distance_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    set_source_files_properties(src/distance_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f")
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  target_compile_definitions(hnsw_lite PRIVATE HNSW_LITE_HAVE_NEON)
endif()

# Warnings and optional sanitizers for every target we build
function(hnsw_lite_setup target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  if(HNSW_LITE_SANITIZE AND NOT MSVC)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()
endfunction()
hnsw_lite_setup(hnsw_lite)

# ---------------------------------------------------------------------------
# Tests and benchmark
# ---------------------------------------------------------------------------
enable_testing()

# Comprehensive suite: one program containing every scenario. Each group is also
# registered as its own CTest entry, e.g. "ctest -R comprehensive.hnsw".
find_package(Threads REQUIRED)
add_executable(test_comprehensive tests/test_comprehensive.cpp)
target_link_libraries(test_comprehensive PRIVATE hnsw_lite Threads::Threads)
hnsw_lite_setup(test_comprehensive)
foreach(group layer1 layer2 helpers flat hnsw robustness concurrency e2e)
  add_test(NAME comprehensive.${group} COMMAND test_comprehensive --group ${group})
endforeach()

foreach(name bench_distance bench_search)
  add_executable(${name} bench/${name}.cpp)
  target_link_libraries(${name} PRIVATE hnsw_lite)
  hnsw_lite_setup(${name})
endforeach()