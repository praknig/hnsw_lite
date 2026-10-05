/**
 * @file dispatch.cpp
 * @brief Implements distance.h: detects CPU features once and hands out the
 *        fastest kernel this machine supports.
 *
 * Two checks decide whether a version can be used:
 *  1. Compile time: CMake defines HNSW_LITE_HAVE_AVX2 / _AVX512 / _NEON only when
 *     that kernel file is part of the build (x86 files on x86, NEON on ARM).
 *  2. Run time (x86 only): the cpuid instruction says what the CPU supports, and
 *     xgetbv says whether the operating system saves the wide registers.
 *     Both are needed: a CPU may support AVX-512 while the OS has it disabled.
 *
 * The result is cached in function-local statics, which C++ initializes exactly
 * once, even when several threads call at the same time.
 */
#include <cmath>

#include "distance.h"
#include "kernels.h"

#if defined(HNSW_LITE_HAVE_AVX2) || defined(HNSW_LITE_HAVE_AVX512)
#define HNSW_LITE_X86 1
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace vecdb {
namespace {

#ifdef HNSW_LITE_X86
/// The four registers returned by the cpuid instruction.
struct CpuidRegs { unsigned a, b, c, d; };

/// Runs cpuid for a leaf (topic) and subleaf. Works on MSVC, GCC and Clang.
CpuidRegs cpuid(unsigned leaf, unsigned subleaf) {
    CpuidRegs r{};
#if defined(_MSC_VER) && !defined(__clang__)
    int v[4];
    __cpuidex(v, static_cast<int>(leaf), static_cast<int>(subleaf));
    r = {unsigned(v[0]), unsigned(v[1]), unsigned(v[2]), unsigned(v[3])};
#else
    __cpuid_count(leaf, subleaf, r.a, r.b, r.c, r.d);
#endif
    return r;
}

/// Reads XCR0: which register groups the operating system saves and restores.
unsigned long long read_xcr0() {
#if defined(_MSC_VER) && !defined(__clang__)
    return _xgetbv(0);
#else
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
}

/// What this x86 CPU and operating system can run.
struct X86Features {
    bool avx2_fma = false;
    bool avx512f = false;
};

/// Asks the CPU and OS which instruction sets are usable.
X86Features detect_x86() {
    X86Features f;
    const unsigned max_leaf = cpuid(0, 0).a;
    const CpuidRegs l1 = cpuid(1, 0);
    const bool fma = l1.c & (1u << 12);
    const bool osxsave = l1.c & (1u << 27);  // OS lets us call xgetbv
    const bool avx = l1.c & (1u << 28);
    if (!osxsave || !avx || max_leaf < 7) return f;  // GCOVR_EXCL_BR_LINE: CPU-dependent (emulated CPU runs)

    const unsigned long long xcr0 = read_xcr0();
    const bool os_saves_ymm = (xcr0 & 0x06) == 0x06;  // 128- and 256-bit registers
    const bool os_saves_zmm = (xcr0 & 0xE6) == 0xE6;  // plus all 512-bit state
    const CpuidRegs l7 = cpuid(7, 0);
    f.avx2_fma = os_saves_ymm && fma && (l7.b & (1u << 5));  // GCOVR_EXCL_BR_LINE: CPU-dependent
    f.avx512f = os_saves_zmm && (l7.b & (1u << 16));         // GCOVR_EXCL_BR_LINE: CPU-dependent
    return f;
}

/// Cached result of detect_x86(); runs the detection only once.
const X86Features& x86_features() {
    static const X86Features f = detect_x86();  // GCOVR_EXCL_BR_LINE: one-time initialization guard
    return f;
}
#endif

/// The three kernels of one version.
struct KernelSet { DistanceFn l2, ip, cos; };

/// Returns the kernels of `isa`. Caller must check isa_supported() first.
KernelSet kernels_for(Isa isa) {
    switch (isa) {
#ifdef HNSW_LITE_HAVE_AVX2
        case Isa::Avx2: return {kernels::l2_avx2, kernels::ip_avx2, kernels::cos_avx2};
#endif
#ifdef HNSW_LITE_HAVE_AVX512
        case Isa::Avx512: return {kernels::l2_avx512, kernels::ip_avx512, kernels::cos_avx512};
#endif
#ifdef HNSW_LITE_HAVE_NEON
        case Isa::Neon: return {kernels::l2_neon, kernels::ip_neon, kernels::cos_neon};
#endif
        default: return {kernels::l2_scalar, kernels::ip_scalar, kernels::cos_scalar};
    }
}

}  // namespace

bool isa_supported(Isa isa) {
    switch (isa) {
        case Isa::Scalar: return true;
#ifdef HNSW_LITE_HAVE_AVX2
        case Isa::Avx2: return x86_features().avx2_fma;
#endif
#ifdef HNSW_LITE_HAVE_AVX512
        case Isa::Avx512: return x86_features().avx512f;
#endif
#ifdef HNSW_LITE_HAVE_NEON
        case Isa::Neon: return true;  // always present on 64-bit ARM
#endif
        default: return false;
    }
}

Isa active_isa() {
    // Fastest first. Detected once, then reused.
    static const Isa best = [] {
        for (Isa isa : {Isa::Avx512, Isa::Avx2, Isa::Neon})  // GCOVR_EXCL_BR_LINE: CPU-dependent
            if (isa_supported(isa)) return isa;              // GCOVR_EXCL_BR_LINE: CPU-dependent
        return Isa::Scalar;                                  // GCOVR_EXCL_LINE: only on CPUs without SIMD
    }();                                                     // GCOVR_EXCL_BR_LINE: initialization guard
    return best;
}

DistanceFn get_distance(Metric metric, Isa isa) {
    if (!isa_supported(isa)) return nullptr;
    const KernelSet k = kernels_for(isa);
    switch (metric) {
        case Metric::InnerProduct: return k.ip;
        case Metric::Cosine: return k.cos;
        default: return k.l2;
    }
}

DistanceFn get_distance(Metric metric) { return get_distance(metric, active_isa()); }

const char* isa_name(Isa isa) {
    switch (isa) {
        case Isa::Avx2: return "avx2";
        case Isa::Avx512: return "avx512";
        case Isa::Neon: return "neon";
        default: return "scalar";
    }
}

void normalize(std::span<float> v) {
    double sum = 0.0;  // double avoids rounding error on long vectors
    for (float x : v) sum += static_cast<double>(x) * x;
    if (sum == 0.0) return;
    // Scale in double: for tiny vectors 1/length exceeds the float range.
    const double inv_len = 1.0 / std::sqrt(sum);
    for (float& x : v) x = static_cast<float>(x * inv_len);
}

}  // namespace vecdb