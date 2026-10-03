cmake_minimum_required(VERSION 3.20)
project(hnsw-lite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HNSW_LITE_SANITIZE "Build tests with AddressSanitizer + UBSan (GCC/Clang, Linux/macOS)" OFF)

# Header-only library
add_library(hnsw_lite INTERFACE)
target_include_directories(hnsw_lite INTERFACE include)

# Tests (headers listed so IDEs treat them as part of a target)
enable_testing()
file(GLOB HNSW_LITE_HEADERS CONFIGURE_DEPENDS include/*.h)
add_executable(test_layer1 tests/test_layer1.cpp ${HNSW_LITE_HEADERS})
target_link_libraries(test_layer1 PRIVATE hnsw_lite)

if(MSVC)
    target_compile_options(test_layer1 PRIVATE /W4)
else()
    target_compile_options(test_layer1 PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(HNSW_LITE_SANITIZE AND NOT MSVC)
    target_compile_options(test_layer1 PRIVATE -fsanitize=address,undefined)
    target_link_options(test_layer1 PRIVATE -fsanitize=address,undefined)
endif()

add_test(NAME test_layer1 COMMAND test_layer1)