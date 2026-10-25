cmake_minimum_required(VERSION 3.20)
project(hnsw-lite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Tests, benchmarks and examples are built only when hnsw-lite is the main
# project, not when another project includes it with add_subdirectory or FetchContent.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(HNSW_LITE_IS_MAIN_PROJECT ON)
else()
  set(HNSW_LITE_IS_MAIN_PROJECT OFF)
endif()
option(HNSW_LITE_BUILD_TESTS "Build the test suite" ${HNSW_LITE_IS_MAIN_PROJECT})
option(HNSW_LITE_BUILD_BENCHMARKS "Build the benchmarks" ${HNSW_LITE_IS_MAIN_PROJECT})
option(HNSW_LITE_BUILD_EXAMPLES "Build the example programs" ${HNSW_LITE_IS_MAIN_PROJECT})
option(HNSW_LITE_SANITIZE "Build with AddressSanitizer + UBSan (GCC/Clang, Linux/macOS)" OFF)
option(HNSW_LITE_BUILD_PYTHON "Build the Python extension module (hnsw_lite_lib._core)" OFF)

find_package(Threads REQUIRED)  # batch search runs on a thread pool

# ---------------------------------------------------------------------------
# Library: Layer 1 headers, Layer 2 kernels, Layer 3 indexes
# ---------------------------------------------------------------------------
file(GLOB HNSW_LITE_HEADERS CONFIGURE_DEPENDS include/*.h src/*.h)
add_library(hnsw_lite STATIC
        src/dispatch.cpp src/distance_scalar.cpp src/distance_neon.cpp   # Layer 2
        src/flat_index.cpp src/hnsw_index.cpp src/query_planner.cpp src/persistence.cpp      # Layer 3
        src/kmeans.cpp src/codec.cpp src/vamana.cpp src/disk_index.cpp src/health.cpp src/upkeep.cpp src/rebuild.cpp src/wal.cpp src/wal_ops.cpp src/wal_replay.cpp src/durable.cpp src/sparse.cpp src/collection.cpp src/scoring.cpp  # ..., durability, sparse and hybrid search, collections
        ${HNSW_LITE_HEADERS})
target_include_directories(hnsw_lite PUBLIC include)
target_link_libraries(hnsw_lite PUBLIC Threads::Threads)
add_library(hnsw_lite::hnsw_lite ALIAS hnsw_lite)

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

# Training and codecs must give bit-identical results on every CPU (their output
# is saved in index files), so a*b + c is never fused into one instruction:
# GCC does that by default on ARM, changing rounding there. MSVC's default
# (/fp:precise) does not fuse.
if(NOT MSVC)
  set_source_files_properties(src/kmeans.cpp src/codec.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

# Warnings and optional sanitizers for every target we build
function(hnsw_lite_setup target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  if(HNSW_LITE_SANITIZE AND NOT MSVC)
    # -O1 as the sanitizers' documentation recommends: the same checks, many times
    # faster than an unoptimized build (frame pointers keep the reports' stacks).
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -O1 -g -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()
endfunction()
hnsw_lite_setup(hnsw_lite)

# ---------------------------------------------------------------------------
# Tests, benchmarks and examples
# ---------------------------------------------------------------------------
if(HNSW_LITE_BUILD_TESTS)
  enable_testing()
  # One test program containing every test. Each group is registered as its own
  # CTest entry (its own process), e.g. "ctest -R comprehensive.hnsw".
  add_executable(test_comprehensive tests/test_comprehensive.cpp)
  target_compile_definitions(test_comprehensive PRIVATE HNSW_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  target_link_libraries(test_comprehensive PRIVATE hnsw_lite)
  hnsw_lite_setup(test_comprehensive)
  foreach(group layer1 layer2 helpers flat hnsw robustness deletion concurrency
          metadata filter planner payload batch range search_e2e update hardening persistence add_batch bulk records control codec sq8 matryoshka lean review codes_only binary pq ivf edges vamana disk health upkeep rebuild wal durable sparse collection maintenance presets acorn reorder scoring features stress e2e)
    # The heaviest groups run as several CTest entries (shards), so that a
    # parallel run (ctest -j) is not held up by one long group; they start first.
    set(shards 1)
    if(group STREQUAL "disk")
      set(shards 4)
    elseif(group MATCHES "^(pq|binary|codes_only|sq8|add_batch|deletion|control|upkeep)$")
      set(shards 2)
    endif()
    if(shards EQUAL 1)
      add_test(NAME comprehensive.${group} COMMAND test_comprehensive --group ${group})
    else()
      foreach(k RANGE 1 ${shards})
        add_test(NAME comprehensive.${group}.${k}of${shards} COMMAND test_comprehensive --group ${group} --shard ${k}/${shards})
        set_tests_properties(comprehensive.${group}.${k}of${shards} PROPERTIES COST 100)
      endforeach()
    endif()
  endforeach()

  # A separate program that tries to break the library: absurd sizes, leaks and
  # unbounded growth, hostile numbers, structural abuse, corrupt files, concurrent
  # misuse (tests/test_adversarial.cpp). Most telling under the sanitizers.
  add_executable(test_adversarial tests/test_adversarial.cpp)
  target_link_libraries(test_adversarial PRIVATE hnsw_lite)
  hnsw_lite_setup(test_adversarial)
  add_test(NAME adversarial COMMAND test_adversarial)

  # Race-condition tests: threads deliberately colliding on every kind of shared
  # state (tests/test_races.cpp). Written for ThreadSanitizer (CI's thread-sanitizer job).
  add_executable(test_races tests/test_races.cpp)
  target_link_libraries(test_races PRIVATE hnsw_lite)
  hnsw_lite_setup(test_races)
  add_test(NAME races COMMAND test_races)
endif()

if(HNSW_LITE_BUILD_BENCHMARKS)
  foreach(name bench_distance bench_search bench_baseline bench_codecs bench_durability)
    add_executable(${name} bench/${name}.cpp)
    target_link_libraries(${name} PRIVATE hnsw_lite)
    hnsw_lite_setup(${name})
  endforeach()
endif()

if(HNSW_LITE_BUILD_EXAMPLES)
  foreach(name example_basics example_filters example_batch_range)
    add_executable(${name} examples/${name}.cpp)
    target_link_libraries(${name} PRIVATE hnsw_lite)
    hnsw_lite_setup(${name})
  endforeach()
endif()

# The Python extension module, built by scikit-build-core (pip install .): the
# library compiled position-independent and linked into hnsw_lite_lib._core.
if(HNSW_LITE_BUILD_PYTHON)
  set_target_properties(hnsw_lite PROPERTIES POSITION_INDEPENDENT_CODE ON)
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core python/src/bindings.cpp)
  target_link_libraries(_core PRIVATE hnsw_lite)
  install(TARGETS _core LIBRARY DESTINATION hnsw_lite_lib)
endif()