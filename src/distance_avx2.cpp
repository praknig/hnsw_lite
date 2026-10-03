/**
 * @file distance_avx2.cpp
 * @brief AVX2 + FMA kernels: 8 floats per instruction (x86-64 only).
 *
 * Compiled with -mavx2 -mfma (or /arch:AVX2). Only called after dispatch.cpp has
 * confirmed the CPU supports AVX2 and FMA.
 *
 * How each kernel works:
 *  1. Main loop: 32 floats per step, split into 4 independent running sums,
 *     so the CPU can work on 4 additions at once instead of waiting on one.
 *  2. Leftover loop: 8 floats per step for the last 16 floats, if any.
 *  3. Final sum: add the 4 running sums, then add the 8 values inside it.
 *
 * Requires n to be a multiple of 8 (the padded stride, a multiple of 16, always is).
 * Uses unaligned loads, which are just as fast on aligned data and never crash.
 */
#include <immintrin.h>

#include "kernels.h"

namespace vecdb::kernels {
namespace {

/// Adds the 8 floats inside one register into a single float.
inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));  // 8 -> 4
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));                                         // 4 -> 2
    s = _mm_add_ss(s, _mm_movehdup_ps(s));                                          // 2 -> 1
    return _mm_cvtss_f32(s);
}

/// Sum of (a[i] - b[i])^2.
inline float l2(const float* a, const float* b, std::size_t n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        s0 = _mm256_fmadd_ps(d0, d0, s0);  // s0 += d0 * d0
        s1 = _mm256_fmadd_ps(d1, d1, s1);
        s2 = _mm256_fmadd_ps(d2, d2, s2);
        s3 = _mm256_fmadd_ps(d3, d3, s3);
    }
    for (; i < n; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        s0 = _mm256_fmadd_ps(d, d, s0);
    }
    return hsum(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
}

/// Sum of a[i] * b[i].
inline float dot(const float* a, const float* b, std::size_t n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for (; i < n; i += 8)
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    return hsum(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
}

}  // namespace

/// Squared L2 distance.
float l2_avx2(const float* a, const float* b, std::size_t n) { return l2(a, b, n); }

/// Inner-product distance: -(a . b).
float ip_avx2(const float* a, const float* b, std::size_t n) { return -dot(a, b, n); }

/// Cosine distance for normalized vectors: 1 - (a . b).
float cos_avx2(const float* a, const float* b, std::size_t n) { return 1.0f - dot(a, b, n); }

}  // namespace vecdb::kernels