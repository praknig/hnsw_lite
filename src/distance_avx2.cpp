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

#include <cstdint>

#include "half.h"
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


namespace {
inline __m256 bytes8(const std::uint8_t* c) {  // 8 bytes, widened to 8 floats
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(c))));
}
}  // namespace

float sq8_l2_avx2(const float* t, const float* s, const std::uint8_t* c, std::size_t n) {
    __m256 acc = _mm256_setzero_ps();
    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) {
        const __m256 diff = _mm256_fnmadd_ps(bytes8(c + j), _mm256_loadu_ps(s + j), _mm256_loadu_ps(t + j));
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    float sum = hsum(acc);
    for (; j < n; ++j) {
        const float diff = t[j] - float(c[j]) * s[j];
        sum += diff * diff;
    }
    return sum;
}

float sq8_dot_avx2(const float* w, const std::uint8_t* c, std::size_t n) {
    __m256 acc = _mm256_setzero_ps();
    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) acc = _mm256_fmadd_ps(bytes8(c + j), _mm256_loadu_ps(w + j), acc);
    float sum = hsum(acc);
    for (; j < n; ++j) sum += w[j] * float(c[j]);
    return sum;
}

float bits_sum_avx2(const float* w, const std::uint8_t* c, std::size_t n) {
    const __m256i bit = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    __m256 acc = _mm256_setzero_ps();
    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) {  // one code byte: 8 dimensions
        const __m256i spread = _mm256_and_si256(_mm256_set1_epi32(c[j / 8]), bit);
        const __m256 mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(spread, bit));
        acc = _mm256_add_ps(acc, _mm256_and_ps(mask, _mm256_loadu_ps(w + j)));
    }
    float sum = hsum(acc);
    for (; j < n; ++j)
        if ((c[j / 8] >> (j % 8)) & 1u) sum += w[j];
    return sum;
}

// Half precision: 8 values per step. float16 widens with F16C (vcvtph2ps); bfloat16
// is the top half of a float32, so widening is a zero-extension and a 16-bit shift.
namespace {
inline __m256 f16x8(const std::uint16_t* c) { return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c))); }
inline __m256 bf16x8(const std::uint16_t* c) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c))), 16));
}
}  // namespace

float f16_l2_avx2(const float* q, const std::uint16_t* c, std::size_t n) {
    __m256 acc = _mm256_setzero_ps();
    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) {
        const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(q + j), f16x8(c + j));
        acc = _mm256_fmadd_ps(d, d, acc);
    }
    float sum = hsum(acc);
    for (; j < n; ++j) {
        const float d = q[j] - f16_to_float(c[j]);
        sum += d * d;
    }
    return sum;
}
float f16_dot_avx2(const float* q, const std::uint16_t* c, std::size_t n) {
    __m256 acc = _mm256_setzero_ps();
    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + j), f16x8(c + j), acc);
    float sum = hsum(acc);
    for (; j < n; ++j) sum += q[j] * f16_to_float(c[j]);
    return sum;
}
float bf16_l2_avx2(const float* q, const std::uint16_t* c, std::size_t n) {
    __m256 acc = _mm256_setzero_ps();
    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) {
        const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(q + j), bf16x8(c + j));
        acc = _mm256_fmadd_ps(d, d, acc);
    }
    float sum = hsum(acc);
    for (; j < n; ++j) {
        const float d = q[j] - bf16_to_float(c[j]);
        sum += d * d;
    }
    return sum;
}
float bf16_dot_avx2(const float* q, const std::uint16_t* c, std::size_t n) {
    __m256 acc = _mm256_setzero_ps();
    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + j), bf16x8(c + j), acc);
    float sum = hsum(acc);
    for (; j < n; ++j) sum += q[j] * bf16_to_float(c[j]);
    return sum;
}

void f16_decode_avx2(const std::uint16_t* c, float* out, std::size_t n) {
    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) _mm256_storeu_ps(out + j, f16x8(c + j));
    for (; j < n; ++j) out[j] = f16_to_float(c[j]);
}
void bf16_decode_avx2(const std::uint16_t* c, float* out, std::size_t n) {
    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) _mm256_storeu_ps(out + j, bf16x8(c + j));
    for (; j < n; ++j) out[j] = bf16_to_float(c[j]);
}

// SQ8 in integers: 32 code bytes times 32 signed weights per step. vpmaddubsw adds
// pairs of products into 16 bits: with weights in [-63, 63], at most 2 * 255 * 63 =
// 32130, so never saturated; vpmaddwd then widens to 32-bit sums.
std::int32_t sq8_dot_i8_avx2(const std::int8_t* w, const std::uint8_t* c, std::size_t n) {
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    std::size_t j = 0;
    for (; j + 32 <= n; j += 32) {
        const __m256i pairs = _mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + j)),
                                                   _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + j)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, ones));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    std::int32_t sum = _mm_cvtsi128_si32(s);
    for (; j < n; ++j) sum += std::int32_t(c[j]) * std::int32_t(w[j]);
    return sum;
}

}  // namespace vecdb::kernels