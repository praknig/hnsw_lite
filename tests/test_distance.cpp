/**
 * @file test_distance.cpp
 * @brief Tests for Layer 2. Prints "All Layer 2 tests passed." on success.
 *
 * Checks that:
 *  - Every version this CPU supports matches a double-precision reference
 *    on random vectors of many lengths (unsupported versions are skipped).
 *  - Exact cases hold (identical vectors, orthogonal vectors, known values).
 *  - Zero padding from VectorStore does not change results.
 *  - normalize() produces length 1 and leaves a zero vector unchanged.
 *  - Dispatch always returns a usable kernel.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "distance.h"
#include "vector_store.h"

#include "check.h"

using namespace vecdb;

static const Isa kAllIsas[] = {Isa::Scalar, Isa::Avx2, Isa::Avx512, Isa::Neon};
static const Metric kAllMetrics[] = {Metric::L2, Metric::InnerProduct, Metric::Cosine};

/// Random values in [-1, 1].
static std::vector<float> random_vector(std::size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (float& x : v) x = dist(rng);
    return v;
}

/// Reference distance in double precision, plus the size of the terms summed,
/// which sets how much float rounding error is acceptable.
static double reference(Metric m, const std::vector<float>& a, const std::vector<float>& b,
                        double& magnitude) {
    double sum = 0.0;
    magnitude = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double term = (m == Metric::L2) ? (double(a[i]) - b[i]) * (double(a[i]) - b[i])
                                        : double(a[i]) * b[i];
        sum += term;
        magnitude += std::fabs(term);
    }
    if (m == Metric::InnerProduct) return -sum;
    if (m == Metric::Cosine) return 1.0 - sum;
    return sum;
}

static void test_matches_reference() {
    std::mt19937 rng(42);
    for (Isa isa : kAllIsas) {
        if (!isa_supported(isa)) {
            std::printf("  %-7s skipped (not supported here)\n", isa_name(isa));
            continue;
        }
        // Lengths chosen to exercise both the main loop and the leftover loop.
        for (std::size_t n : {16, 32, 48, 64, 80, 112, 128, 768, 1536}) {
            auto a = random_vector(n, rng), b = random_vector(n, rng);
            for (Metric m : kAllMetrics) {
                double magnitude = 0.0;
                double expected = reference(m, a, b, magnitude);
                float got = get_distance(m, isa)(a.data(), b.data(), n);
                CHECK(std::fabs(got - expected) <= 1e-5 * magnitude + 1e-6);
            }
        }
        std::printf("  %-7s matches reference\n", isa_name(isa));
    }
}

static void test_exact_cases() {
    std::vector<float> a(16, 0.0f), b(16, 0.0f);
    a[0] = 1; a[1] = 2;  // a = [1, 2, 0, ...]
    b[0] = 4; b[1] = 6;  // b = [4, 6, 0, ...]
    std::vector<float> x(16, 0.0f), y(16, 0.0f);
    x[0] = 1; y[1] = 1;  // orthogonal unit vectors

    for (Isa isa : kAllIsas) {
        if (!isa_supported(isa)) continue;
        auto l2 = get_distance(Metric::L2, isa);
        auto ip = get_distance(Metric::InnerProduct, isa);
        auto cos = get_distance(Metric::Cosine, isa);
        CHECK(l2(a.data(), b.data(), 16) == 25.0f);   // 3^2 + 4^2
        CHECK(ip(a.data(), b.data(), 16) == -16.0f);  // -(1*4 + 2*6)
        CHECK(l2(a.data(), a.data(), 16) == 0.0f);    // same vector
        CHECK(cos(x.data(), x.data(), 16) == 0.0f);   // same direction
        CHECK(cos(x.data(), y.data(), 16) == 1.0f);   // perpendicular
    }
}

static void test_padding_is_harmless() {
    std::mt19937 rng(7);
    VectorStore store(100);  // stride 112: 12 zeros after each vector
    auto a = random_vector(100, rng), b = random_vector(100, rng);
    NodeId ia = store.add(a), ib = store.add(b);

    double magnitude = 0.0;
    double expected = reference(Metric::L2, a, b, magnitude);
    auto l2 = get_distance(Metric::L2);
    float got = l2(store.get_padded(ia).data(), store.get_padded(ib).data(), store.stride());
    CHECK(std::fabs(got - expected) <= 1e-5 * magnitude + 1e-6);
}

static void test_normalize() {
    std::vector<float> v{3.0f, 4.0f};  // length 5
    normalize(v);
    CHECK(std::fabs(v[0] - 0.6f) < 1e-6f && std::fabs(v[1] - 0.8f) < 1e-6f);

    std::mt19937 rng(1);
    auto w = random_vector(768, rng);
    normalize(w);
    double len2 = 0.0;
    for (float x : w) len2 += double(x) * x;
    CHECK(std::fabs(len2 - 1.0) < 1e-5);

    std::vector<float> zero(16, 0.0f);
    normalize(zero);  // must not divide by zero
    for (float x : zero) CHECK(x == 0.0f);
}

static void test_dispatch() {
    CHECK(isa_supported(Isa::Scalar));
    CHECK(isa_supported(active_isa()));
    for (Metric m : kAllMetrics) CHECK(get_distance(m) != nullptr);
    for (Isa isa : kAllIsas)
        CHECK((get_distance(Metric::L2, isa) != nullptr) == isa_supported(isa));
    std::printf("  active version: %s\n", isa_name(active_isa()));
}

int main() {
    test_matches_reference();
    test_exact_cases();
    test_padding_is_harmless();
    test_normalize();
    test_dispatch();
    std::printf("All Layer 2 tests passed.\n");
}