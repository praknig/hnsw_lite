/**
* @file check.h
 * @brief Tiny shared test helpers, so test files need no external framework.
 */
#pragma once
#include <cstdio>
#include <cstdlib>

/// Stops the test program with the line number if `cond` is false.
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

/// True if calling `f` throws any exception.
template <class F>
bool throws(F f) {
    try {
        f();
    } catch (...) {
        return true;
    }
    return false;
}