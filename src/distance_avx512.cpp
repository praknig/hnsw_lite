/**
 * @file distance_avx512.cpp
 * @brief AVX-512F kernels: 16 floats per instruction (x86-64 only).
 *
 * Compiled with -mavx512f (or /arch:AVX512). Only called after dispatch.cpp has
 * confirmed the CPU and operating system support AVX-512.
 *
 * Same structure as the AVX2 version, with registers twice as wide:
 *  1. Main loop: 64 floats per step in 4 independent running sums.
 *  2. Leftover loop: 16 floats per step.
 *  3. Final sum: _mm512_reduce_add_ps adds the 16 values inside a register.
 *
 * Requires n to be a multiple of 16 (the padded stride always is).
 */
#include <immintrin.h>

#include "kernels.h"

namespace vecdb::kernels {
namespace {

/// Sum of (a[i] - b[i])^2.
inline float l2(const float* a, const float* b, std::size_t n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    std::size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
        __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32));
        __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48));
        s0 = _mm512_fmadd_ps(d0, d0, s0);
        s1 = _mm512_fmadd_ps(d1, d1, s1);
        s2 = _mm512_fmadd_ps(d2, d2, s2);
        s3 = _mm512_fmadd_ps(d3, d3, s3);
    }
    for (; i < n; i += 16) {
        __m512 d = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        s0 = _mm512_fmadd_ps(d, d, s0);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
}

/// Sum of a[i] * b[i].
inline float dot(const float* a, const float* b, std::size_t n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    std::size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), s1);
        s2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), s2);
        s3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), s3);
    }
    for (; i < n; i += 16)
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
}

}  // namespace

/// Squared L2 distance.
float l2_avx512(const float* a, const float* b, std::size_t n) { return l2(a, b, n); }

/// Inner-product distance: -(a . b).
float ip_avx512(const float* a, const float* b, std::size_t n) { return -dot(a, b, n); }

/// Cosine distance for normalized vectors: 1 - (a . b).
float cos_avx512(const float* a, const float* b, std::size_t n) { return 1.0f - dot(a, b, n); }

}  // namespace vecdb::kernels