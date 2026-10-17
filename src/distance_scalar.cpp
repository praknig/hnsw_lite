/**
 * @file distance_scalar.cpp
 * @brief Scalar kernels: one float per step. Works on every CPU.
 *
 * This is the simplest, most obviously correct version. It is the fallback when
 * no SIMD is available and the speed baseline in benchmarks. The compiler does
 * not turn these loops into SIMD on its own, because that would change the order
 * of float additions, which strict floating-point rules forbid.
 *
 * Works for any n (no multiple-of-16 requirement).
 */
#include <cstdint>

#include "kernels.h"

namespace vecdb::kernels {
namespace {

/// Sum of (a[i] - b[i])^2.
float l2(const float* a, const float* b, std::size_t n) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

/// Sum of a[i] * b[i].
float dot(const float* a, const float* b, std::size_t n) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

}  // namespace

/// Squared L2 distance.
float l2_scalar(const float* a, const float* b, std::size_t n) { return l2(a, b, n); }

/// Inner-product distance: -(a . b), so smaller means closer.
float ip_scalar(const float* a, const float* b, std::size_t n) { return -dot(a, b, n); }

/// Cosine distance for normalized vectors: 1 - (a . b).
float cos_scalar(const float* a, const float* b, std::size_t n) { return 1.0f - dot(a, b, n); }


float sq8_l2_scalar(const float* t, const float* s, const std::uint8_t* c, std::size_t n) {
    float sum = 0;
    for (std::size_t j = 0; j < n; ++j) {
        const float diff = t[j] - float(c[j]) * s[j];
        sum += diff * diff;
    }
    return sum;
}

float sq8_dot_scalar(const float* w, const std::uint8_t* c, std::size_t n) {
    float sum = 0;
    for (std::size_t j = 0; j < n; ++j) sum += w[j] * float(c[j]);
    return sum;
}

float bits_sum_scalar(const float* w, const std::uint8_t* c, std::size_t n) {
    float sum = 0;
    for (std::size_t j = 0; j < n; ++j)
        if ((c[j / 8] >> (j % 8)) & 1u) sum += w[j];
    return sum;
}
}  // namespace vecdb::kernels