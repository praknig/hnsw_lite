cmake_minimum_required(VERSION 4.3)
project(hnsw_lite)

set(CMAKE_CXX_STANDARD 20)

file(GLOB HEADERS include/*.h)
add_executable(test_layer1 tests/test_layer1.cpp ${HEADERS})
