/**
* @file check.h
 * @brief Tiny shared test helpers, so test files need no external framework.
 */
#pragma once
#include <cstdio>
#include <cstdlib>

/// Number of CHECKs executed so far (tests may print it at the end).
inline long g_check_count = 0;

/// Stops the test program with the file and line if `cond` is false.
#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_check_count;                                                     \
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

/// True only if calling `f` throws an exception of type E (or derived from E).
template <class E, class F>
bool throws_as(F f) {
    try {
        f();
    } catch (const E&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}