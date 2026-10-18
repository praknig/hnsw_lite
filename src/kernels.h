/**
 * @file kernels.h
 * @brief Internal declarations of every distance kernel (not part of the public API).
 *
 * Each kernel file (scalar, AVX2, AVX-512, NEON) defines three functions with the
 * DistanceFn shape: l2_*, ip_* and cos_*. dispatch.cpp picks among them.
 *
 * IMPORTANT: this header contains declarations only, no inline code. Kernel files
 * are compiled with special CPU flags (for example -mavx2). If they included a
 * header with inline functions, those functions could be compiled with AVX2
 * instructions and then used on CPUs without AVX2, causing a crash.
 */
#pragma once
#include <cstddef>
#include <cstdint>

namespace vecdb::kernels {

// Scalar: plain loops, works on every CPU. Also the reference for tests.
float l2_scalar(const float* a, const float* b, std::size_t n);
float ip_scalar(const float* a, const float* b, std::size_t n);
float cos_scalar(const float* a, const float* b, std::size_t n);

// AVX2 + FMA: 8 floats per instruction (x86-64 only).
float l2_avx2(const float* a, const float* b, std::size_t n);
float ip_avx2(const float* a, const float* b, std::size_t n);
float cos_avx2(const float* a, const float* b, std::size_t n);

// AVX-512F: 16 floats per instruction (x86-64 only).
float l2_avx512(const float* a, const float* b, std::size_t n);
float ip_avx512(const float* a, const float* b, std::size_t n);
float cos_avx512(const float* a, const float* b, std::size_t n);

// NEON: 4 floats per instruction (64-bit ARM only).
float l2_neon(const float* a, const float* b, std::size_t n);
float ip_neon(const float* a, const float* b, std::size_t n);
float cos_neon(const float* a, const float* b, std::size_t n);


// Code distances (codec.h), each with the same four versions:
//   sq8_l2:   sum over j of (t[j] - c[j] * s[j])^2   (SQ8, L2)
//   sq8_dot:  sum over j of w[j] * c[j]              (SQ8, inner product and cosine)
//   bits_sum: sum of w[j] over the set bits of c     (binary: bit j is c[j / 8] >> (j % 8))
float sq8_l2_scalar(const float* t, const float* s, const std::uint8_t* c, std::size_t n);
float sq8_dot_scalar(const float* w, const std::uint8_t* c, std::size_t n);
float bits_sum_scalar(const float* w, const std::uint8_t* c, std::size_t n);
float sq8_l2_avx2(const float* t, const float* s, const std::uint8_t* c, std::size_t n);
float sq8_dot_avx2(const float* w, const std::uint8_t* c, std::size_t n);
float bits_sum_avx2(const float* w, const std::uint8_t* c, std::size_t n);
float sq8_l2_avx512(const float* t, const float* s, const std::uint8_t* c, std::size_t n);
float sq8_dot_avx512(const float* w, const std::uint8_t* c, std::size_t n);
float bits_sum_avx512(const float* w, const std::uint8_t* c, std::size_t n);
float sq8_l2_neon(const float* t, const float* s, const std::uint8_t* c, std::size_t n);
float sq8_dot_neon(const float* w, const std::uint8_t* c, std::size_t n);
float bits_sum_neon(const float* w, const std::uint8_t* c, std::size_t n);
}  // namespace vecdb::kernels