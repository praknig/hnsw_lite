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

#include <cstdint>

#include "half.h"
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


namespace {
inline __m512 bytes16(const std::uint8_t* c) {  // 16 bytes, widened to 16 floats
    return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c))));
}
}  // namespace

float sq8_l2_avx512(const float* t, const float* s, const std::uint8_t* c, std::size_t n) {
    __m512 acc = _mm512_setzero_ps();
    std::size_t j = 0;
    for (; j + 16 <= n; j += 16) {
        const __m512 diff = _mm512_fnmadd_ps(bytes16(c + j), _mm512_loadu_ps(s + j), _mm512_loadu_ps(t + j));
        acc = _mm512_fmadd_ps(diff, diff, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; j < n; ++j) {
        const float diff = t[j] - float(c[j]) * s[j];
        sum += diff * diff;
    }
    return sum;
}

float sq8_dot_avx512(const float* w, const std::uint8_t* c, std::size_t n) {
    __m512 acc = _mm512_setzero_ps();
    std::size_t j = 0;
    for (; j + 16 <= n; j += 16) acc = _mm512_fmadd_ps(bytes16(c + j), _mm512_loadu_ps(w + j), acc);
    float sum = _mm512_reduce_add_ps(acc);
    for (; j < n; ++j) sum += w[j] * float(c[j]);
    return sum;
}

float bits_sum_avx512(const float* w, const std::uint8_t* c, std::size_t n) {
    __m512 acc = _mm512_setzero_ps();
    std::size_t j = 0;
    for (; j + 16 <= n; j += 16) {  // two code bytes are exactly a 16-lane mask
        const __mmask16 m = static_cast<__mmask16>(c[j / 8] | (c[j / 8 + 1] << 8));
        acc = _mm512_mask_add_ps(acc, m, acc, _mm512_loadu_ps(w + j));
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; j < n; ++j)
        if ((c[j / 8] >> (j % 8)) & 1u) sum += w[j];
    return sum;
}

// Half precision: 16 values per step (vcvtph2ps zmm is part of AVX-512F).
namespace {
inline __m512 f16x16(const std::uint16_t* c) { return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(c))); }
inline __m512 bf16x16(const std::uint16_t* c) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(c))), 16));
}
}  // namespace

float f16_l2_avx512(const float* q, const std::uint16_t* c, std::size_t n) {
    __m512 acc = _mm512_setzero_ps();
    std::size_t j = 0;
    for (; j + 16 <= n; j += 16) {
        const __m512 d = _mm512_sub_ps(_mm512_loadu_ps(q + j), f16x16(c + j));
        acc = _mm512_fmadd_ps(d, d, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; j < n; ++j) {
        const float d = q[j] - f16_to_float(c[j]);
        sum += d * d;
    }
    return sum;
}
float f16_dot_avx512(const float* q, const std::uint16_t* c, std::size_t n) {
    __m512 acc = _mm512_setzero_ps();
    std::size_t j = 0;
    for (; j + 16 <= n; j += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(q + j), f16x16(c + j), acc);
    float sum = _mm512_reduce_add_ps(acc);
    for (; j < n; ++j) sum += q[j] * f16_to_float(c[j]);
    return sum;
}
float bf16_l2_avx512(const float* q, const std::uint16_t* c, std::size_t n) {
    __m512 acc = _mm512_setzero_ps();
    std::size_t j = 0;
    for (; j + 16 <= n; j += 16) {
        const __m512 d = _mm512_sub_ps(_mm512_loadu_ps(q + j), bf16x16(c + j));
        acc = _mm512_fmadd_ps(d, d, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; j < n; ++j) {
        const float d = q[j] - bf16_to_float(c[j]);
        sum += d * d;
    }
    return sum;
}
float bf16_dot_avx512(const float* q, const std::uint16_t* c, std::size_t n) {
    __m512 acc = _mm512_setzero_ps();
    std::size_t j = 0;
    for (; j + 16 <= n; j += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(q + j), bf16x16(c + j), acc);
    float sum = _mm512_reduce_add_ps(acc);
    for (; j < n; ++j) sum += q[j] * bf16_to_float(c[j]);
    return sum;
}

void f16_decode_avx512(const std::uint16_t* c, float* out, std::size_t n) {
    std::size_t j = 0;
    for (; j + 16 <= n; j += 16) _mm512_storeu_ps(out + j, f16x16(c + j));
    for (; j < n; ++j) out[j] = f16_to_float(c[j]);
}
void bf16_decode_avx512(const std::uint16_t* c, float* out, std::size_t n) {
    std::size_t j = 0;
    for (; j + 16 <= n; j += 16) _mm512_storeu_ps(out + j, bf16x16(c + j));
    for (; j < n; ++j) out[j] = bf16_to_float(c[j]);
}

}  // namespace vecdb::kernels