/**
 * @file distance.h
 * @brief Public API of Layer 2: fast distance functions between two vectors.
 *
 * Every distance follows one rule: a SMALLER value means the vectors are CLOSER.
 *  - L2:           sum of squared differences (no square root; ranking is the same).
 *  - InnerProduct: -(a . b), so a larger dot product gives a smaller distance.
 *  - Cosine:       1 - (a . b). Vectors MUST be normalized first (see normalize()).
 *
 * Each distance exists in several versions (Isa): Scalar, AVX2, AVX-512, NEON.
 * get_distance(metric) picks the fastest version this CPU supports, once.
 *
 * Input rules for every DistanceFn:
 *  - n must be a multiple of 16 (use VectorStore::stride(), not dim()).
 *  - Values after the real dimension must be zero (VectorStore guarantees this).
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>

namespace vecdb {

/// Which distance to compute.
enum class Metric { L2, InnerProduct, Cosine };

/// Which instruction set a kernel uses.
enum class Isa { Scalar, Avx2, Avx512, Neon };

/// Shape of every distance kernel: two vectors of n floats in, one distance out.
using DistanceFn = float (*)(const float* a, const float* b, std::size_t n);

/// Returns the fastest kernel for `metric` on this CPU. Never returns nullptr.
DistanceFn get_distance(Metric metric);

/// Returns the kernel for `metric` in a specific version, or nullptr if this
/// build or this CPU does not support it. Used by tests and benchmarks.
DistanceFn get_distance(Metric metric, Isa isa);

/// True if `isa` was compiled into this build AND this CPU can run it.
bool isa_supported(Isa isa);

/// The version get_distance(metric) uses: the fastest supported one.
Isa active_isa();

/// Human-readable name: "scalar", "avx2", "avx512" or "neon".
const char* isa_name(Isa isa);

/// Scales `v` to length 1, in place. A zero vector is left unchanged.
/// Call this before storing vectors (and on queries) when using Metric::Cosine.
void normalize(std::span<float> v);

/// The kernels behind codec distances (codec.h): for SQ8 the sum over j of
/// (t[j] - c[j] * s[j])^2 and of w[j] * c[j]; for binary codes the sum of w[j]
/// over the set bits of c (bit j is c[j / 8] >> (j % 8)).
struct CodeKernels {
    float (*sq8_l2)(const float* t, const float* s, const std::uint8_t* c, std::size_t n);
    float (*sq8_dot)(const float* w, const std::uint8_t* c, std::size_t n);
    float (*bits_sum)(const float* w, const std::uint8_t* c, std::size_t n);
    /// Half precision (HalfCodec): squared L2 and dot product between a float32 query
    /// and float16 or bfloat16 codes.
    float (*f16_l2)(const float* q, const std::uint16_t* c, std::size_t n);
    float (*f16_dot)(const float* q, const std::uint16_t* c, std::size_t n);
    float (*bf16_l2)(const float* q, const std::uint16_t* c, std::size_t n);
    float (*bf16_dot)(const float* q, const std::uint16_t* c, std::size_t n);
    /// Half precision: n codes widened to float32 (decoding, for comparing two stored vectors).
    void (*f16_decode)(const std::uint16_t* c, float* out, std::size_t n);
    void (*bf16_decode)(const std::uint16_t* c, float* out, std::size_t n);
};

/// The fastest code kernels this CPU supports (the same choice as get_distance).
CodeKernels get_code_kernels();

/// The code kernels of one instruction set (for tests and benchmarks).
/// Throws std::invalid_argument if this CPU does not support it.
CodeKernels get_code_kernels(Isa isa);

}  // namespace vecdb