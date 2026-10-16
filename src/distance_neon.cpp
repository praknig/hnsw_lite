/**
 * @file distance_neon.cpp
 * @brief NEON kernels: 4 floats per instruction (64-bit ARM only).
 *
 * NEON is always present on 64-bit ARM (Apple Silicon, AWS Graviton, phones),
 * so no special compiler flags or CPU check are needed.
 *
 * Same structure as the x86 versions:
 *  1. Main loop: 16 floats per step in 4 independent running sums.
 *  2. Leftover loop: 4 floats per step (normally unused, since n is a multiple of 16).
 *  3. Final sum: vaddvq_f32 adds the 4 values inside a register.
 *
 * Requires n to be a multiple of 4 (the padded stride always is).
 *
 * This file is part of every build, but its code only exists on 64-bit ARM.
 * On x86 it compiles to nothing, so IDEs on x86 show no errors for it.
 */
#include "kernels.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

namespace vecdb::kernels {
namespace {

/// Sum of (a[i] - b[i])^2.
inline float l2(const float* a, const float* b, std::size_t n) {
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        s0 = vfmaq_f32(s0, d0, d0);  // s0 += d0 * d0
        s1 = vfmaq_f32(s1, d1, d1);
        s2 = vfmaq_f32(s2, d2, d2);
        s3 = vfmaq_f32(s3, d3, d3);
    }
    for (; i < n; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        s0 = vfmaq_f32(s0, d, d);
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
}

/// Sum of a[i] * b[i].
inline float dot(const float* a, const float* b, std::size_t n) {
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i < n; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    return vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
}

}  // namespace

/// Squared L2 distance.
float l2_neon(const float* a, const float* b, std::size_t n) { return l2(a, b, n); }

/// Inner-product distance: -(a . b).
float ip_neon(const float* a, const float* b, std::size_t n) { return -dot(a, b, n); }

/// Cosine distance for normalized vectors: 1 - (a . b).
float cos_neon(const float* a, const float* b, std::size_t n) { return 1.0f - dot(a, b, n); }


float sq8_l2_neon(const float* t, const float* s, const std::uint8_t* c, std::size_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) {
        const uint16x8_t w16 = vmovl_u8(vld1_u8(c + j));
        const float32x4_t lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16))), hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16)));
        const float32x4_t d0 = vfmsq_f32(vld1q_f32(t + j), lo, vld1q_f32(s + j));
        const float32x4_t d1 = vfmsq_f32(vld1q_f32(t + j + 4), hi, vld1q_f32(s + j + 4));
        acc = vfmaq_f32(vfmaq_f32(acc, d0, d0), d1, d1);
    }
    float sum = vaddvq_f32(acc);
    for (; j < n; ++j) {
        const float diff = t[j] - float(c[j]) * s[j];
        sum += diff * diff;
    }
    return sum;
}

float sq8_dot_neon(const float* w, const std::uint8_t* c, std::size_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) {
        const uint16x8_t w16 = vmovl_u8(vld1_u8(c + j));
        acc = vfmaq_f32(acc, vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16))), vld1q_f32(w + j));
        acc = vfmaq_f32(acc, vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16))), vld1q_f32(w + j + 4));
    }
    float sum = vaddvq_f32(acc);
    for (; j < n; ++j) sum += w[j] * float(c[j]);
    return sum;
}

float bits_sum_neon(const float* w, const std::uint8_t* c, std::size_t n) {
    const uint32x4_t lo_bits = {1, 2, 4, 8}, hi_bits = {16, 32, 64, 128};
    float32x4_t acc = vdupq_n_f32(0.0f);
    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) {  // one code byte: 8 dimensions
        const uint32x4_t b = vdupq_n_u32(c[j / 8]);
        acc = vaddq_f32(acc, vreinterpretq_f32_u32(vandq_u32(vtstq_u32(b, lo_bits), vreinterpretq_u32_f32(vld1q_f32(w + j)))));
        acc = vaddq_f32(acc, vreinterpretq_f32_u32(vandq_u32(vtstq_u32(b, hi_bits), vreinterpretq_u32_f32(vld1q_f32(w + j + 4)))));
    }
    float sum = vaddvq_f32(acc);
    for (; j < n; ++j)
        if ((c[j / 8] >> (j % 8)) & 1u) sum += w[j];
    return sum;
}
}  // namespace vecdb::kernels

#endif  // 64-bit ARM