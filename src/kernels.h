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

}  // namespace vecdb::kernels