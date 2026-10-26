/**
 * @file codec.cpp
 * @brief Vector codecs; see codec.h. Built without floating-point contraction
 *        (see CMakeLists.txt), so codes are identical on every CPU.
 */
#include <cstring>
#include <random>
#include "codec.h"
#include "half.h"
#include "kmeans.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vecdb {

void SQ8Codec::train(std::span<const float> sample) {
    const std::size_t d = dim();
    if (sample.empty() || sample.size() % d != 0) throw std::invalid_argument("SQ8: sample must hold whole vectors, at least one");
    std::vector<float> lo(sample.begin(), sample.begin() + std::ptrdiff_t(d)), hi = lo;
    for (std::size_t i = 0; i < sample.size(); ++i) {
        const float x = sample[i];
        if (!std::isfinite(x)) throw std::invalid_argument("SQ8: sample contains NaN or infinity");
        lo[i % d] = std::min(lo[i % d], x);
        hi[i % d] = std::max(hi[i % d], x);
    }
    std::vector<float> step(d);
    // In double: hi - lo overflows a float when the values span more than its range.
    for (std::size_t j = 0; j < d; ++j) step[j] = static_cast<float>((double(hi[j]) - double(lo[j])) / 255.0);
    set_range(lo, step);
}

void SQ8Codec::set_range(std::span<const float> minimum, std::span<const float> step) {
    if (minimum.size() != dim() || step.size() != dim()) throw std::invalid_argument("SQ8: range has wrong size");
    for (std::size_t j = 0; j < dim(); ++j)
        if (!std::isfinite(minimum[j]) || !std::isfinite(step[j]) || step[j] < 0)
            throw std::invalid_argument("SQ8: range values must be finite, steps not negative");
    min_.assign(minimum.begin(), minimum.end());
    step_.assign(step.begin(), step.end());
}

void SQ8Codec::encode(std::span<const float> v, std::uint8_t* code) const {
    check_trained();
    check_size(v);
    for (std::size_t j = 0; j < dim(); ++j) {
        float level = 0;  // a constant dimension (step 0) is stored exactly as its minimum
        // In double (deterministic everywhere): v - min overflows a float for huge ranges.
        if (step_[j] > 0) level = static_cast<float>(std::floor((double(v[j]) - double(min_[j])) / double(step_[j]) + 0.5));
        code[j] = static_cast<std::uint8_t>(std::clamp(level, 0.0f, 255.0f));     // outside the range: clamped
    }
}

void SQ8Codec::decode(const std::uint8_t* code, std::span<float> out) const {
    check_trained();
    check_size(out);
    for (std::size_t j = 0; j < dim(); ++j) {  // in double, kept within the float range
        const double x = double(min_[j]) + double(code[j]) * double(step_[j]);
        out[j] = static_cast<float>(std::clamp(x, -double(std::numeric_limits<float>::max()), double(std::numeric_limits<float>::max())));
    }
}

void SQ8Codec::prepare(std::span<const float> query, std::vector<float>& table) const {
    check_trained();
    check_size(query);
    const std::size_t d = dim();
    if (metric() == Metric::L2) {
        // |q - (min + c * step)|^2 = sum ((q - min) - c * step)^2
        table.resize(d);
        for (std::size_t j = 0; j < d; ++j) table[j] = query[j] - min_[j];
    } else {
        // q . (min + c * step) = sum q * min + sum (q * step) * c. The weights q * step
        // are quantized to integers in [-63, 63] with one scale per query, so the sum
        // runs in integer SIMD (2.5x to 2.8x faster than floats on AVX2): the table holds
        // the constant, the scale, then the weights as bytes. Each weight is off by at most
        // half a scale step, so a distance by at most scale / 2 * sum(c).
        //
        // In double: with values near float's limit, q * step and sum(q * min) exceed float's
        // range, and infinity minus infinity would make distances NaN (it did before integer
        // kernels too). The constant and the scale are kept as doubles, their 8 bytes copied
        // into two table slots each, and the distance is summed in double: at worst it
        // overflows to infinity when converted, never NaN.
        std::vector<double> weight(d);
        double constant = 0, largest = 0;
        for (std::size_t j = 0; j < d; ++j) {
            weight[j] = double(query[j]) * double(step_[j]);
            constant += double(query[j]) * double(min_[j]);
            largest = std::max(largest, std::abs(weight[j]));
        }
        const double scale = largest > 0 ? largest / 63.0 : 0.0;
        table.assign(4 + (d + 3) / 4, 0.0f);
        const double numbers[2] = {constant, scale};
        std::memcpy(table.data(), numbers, sizeof numbers);  // slots 0 to 3
        auto* w8 = reinterpret_cast<std::int8_t*>(table.data() + 4);
        for (std::size_t j = 0; j < d; ++j)
            w8[j] = scale > 0 ? static_cast<std::int8_t>(std::clamp(std::llround(weight[j] / scale), -63LL, 63LL)) : std::int8_t(0);
    }
}

float SQ8Codec::distance(const std::vector<float>& table, const std::uint8_t* code) const {
    const std::size_t d = dim();
    if (metric() == Metric::L2) {
        // With values near float's limit, q - min overflows to infinity; a kernel rounding
        // c * step to infinity too (the plain one: no fused multiply-subtract) then gets
        // infinity - infinity. NaN arises only there, where the true distance is beyond
        // float's range: infinity (found by a4 in a build using the plain kernel).
        const float l2 = kernels_.sq8_l2(table.data(), step_.data(), code, d);  // SIMD where available
        return std::isnan(l2) ? std::numeric_limits<float>::infinity() : l2;
    }
    const auto* w8 = reinterpret_cast<const std::int8_t*>(table.data() + 4);
    double numbers[2];  // the constant and the scale (see prepare)
    std::memcpy(numbers, table.data(), sizeof numbers);
    const double dot = numbers[0] + numbers[1] * double(kernels_.sq8_dot_i8(w8, code, d));  // exact integer sum
    return static_cast<float>(metric() == Metric::Cosine ? 1.0 - dot : -dot);
}

void BinaryCodec::train(std::span<const float> sample) {
    const std::size_t d = dim();
    if (sample.empty() || sample.size() % d != 0) throw std::invalid_argument("Binary: sample must hold whole vectors, at least one");
    const std::size_t n = sample.size() / d;
    std::vector<double> mean(d, 0.0), deviation(d, 0.0);
    for (std::size_t i = 0; i < sample.size(); ++i) {
        if (!std::isfinite(sample[i])) throw std::invalid_argument("Binary: sample contains NaN or infinity");
        mean[i % d] += sample[i];
    }
    std::vector<float> threshold(d), spread(d);
    for (std::size_t j = 0; j < d; ++j) threshold[j] = static_cast<float>(mean[j] / double(n));
    for (std::size_t i = 0; i < sample.size(); ++i) deviation[i % d] += std::abs(double(sample[i]) - double(threshold[i % d]));
    for (std::size_t j = 0; j < d; ++j) spread[j] = static_cast<float>(deviation[j] / double(n));
    std::vector<float> state(threshold);
    state.insert(state.end(), spread.begin(), spread.end());
    restore(state);
}

void BinaryCodec::restore(std::span<const float> state) {
    const std::size_t d = dim();
    if (state.size() != 2 * d) throw std::invalid_argument("Binary: saved state has wrong size");
    for (std::size_t j = 0; j < d; ++j)
        if (!std::isfinite(state[j]) || !std::isfinite(state[d + j]) || state[d + j] < 0 || !std::isfinite(state[j] + state[d + j]) ||
            !std::isfinite(state[j] - state[d + j]))
            throw std::invalid_argument("Binary: thresholds and spreads must be finite, spreads not negative");
    threshold_.assign(state.begin(), state.begin() + std::ptrdiff_t(d));
    spread_.assign(state.begin() + std::ptrdiff_t(d), state.end());
}

float BinaryCodec::high(std::size_t j) const {
    // Above the threshold even when the spread is 0, so it encodes back to 1.
    return spread_[j] > 0 ? threshold_[j] + spread_[j] : std::nextafter(threshold_[j], std::numeric_limits<float>::infinity());
}

void BinaryCodec::encode(std::span<const float> v, std::uint8_t* code) const {
    check_trained();
    check_size(v);
    std::fill(code, code + code_size(), std::uint8_t(0));
    for (std::size_t j = 0; j < dim(); ++j)
        if (v[j] > threshold_[j]) code[j / 8] = static_cast<std::uint8_t>(code[j / 8] | (1u << (j % 8)));  // a comparison: same everywhere
}

void BinaryCodec::decode(const std::uint8_t* code, std::span<float> out) const {
    check_trained();
    check_size(out);
    for (std::size_t j = 0; j < dim(); ++j) out[j] = (code[j / 8] >> (j % 8)) & 1u ? high(j) : threshold_[j] - spread_[j];
}

void BinaryCodec::prepare(std::span<const float> query, std::vector<float>& table) const {
    check_trained();
    check_size(query);
    // Each dimension contributes one of two values, by its bit: the table holds
    // the contribution of a 0 bit, and how much a 1 bit adds, plus their sum.
    const std::size_t d = dim();
    table.resize(2 * d + 1);
    float base = 0;
    for (std::size_t j = 0; j < d; ++j) {
        const float low = threshold_[j] - spread_[j];
        float c0, c1;
        if (metric() == Metric::L2) {
            c0 = (query[j] - low) * (query[j] - low);
            c1 = (query[j] - high(j)) * (query[j] - high(j));
        } else {
            c0 = -query[j] * low;  // negative dot product
            c1 = -query[j] * high(j);
        }
        table[j] = c0;
        table[d + j] = c1 - c0;
        base += c0;
    }
    table[2 * d] = base;
}

float BinaryCodec::distance(const std::vector<float>& table, const std::uint8_t* code) const {
    const std::size_t d = dim();
    const float sum = table[2 * d] + kernels_.bits_sum(table.data() + d, code, d);  // SIMD where available
    return metric() == Metric::Cosine ? 1.0f + sum : sum;  // cosine: 1 - dot, and sum is -dot
}

PQCodec::PQCodec(std::size_t dim, Metric metric, std::size_t parts)
    : VectorCodec(dim, metric), parts_(parts == 0 ? std::max<std::size_t>(1, dim / 8) : parts) {
    if (parts_ > dim) throw std::invalid_argument("PQ: more parts than dimensions");
}

std::size_t PQCodec::part_begin(std::size_t p) const {
    const std::size_t base = dim() / parts_, extra = dim() % parts_;  // the first `extra` parts get one more
    return p * base + std::min(p, extra);
}

void PQCodec::train(std::span<const float> sample) {
    const std::size_t d = dim();
    if (sample.empty() || sample.size() % d != 0) throw std::invalid_argument("PQ: sample must hold whole vectors, at least one");
    for (float x : sample)
        if (!std::isfinite(x)) throw std::invalid_argument("PQ: sample contains NaN or infinity");
    const std::size_t n = sample.size() / d, k = std::min<std::size_t>(256, n);
    std::vector<float> state{float(k)};
    std::vector<float> part;
    for (std::size_t p = 0; p < parts_; ++p) {
        const std::size_t b = part_begin(p), e = part_begin(p + 1), pd = e - b;
        part.resize(n * pd);
        for (std::size_t i = 0; i < n; ++i) std::copy_n(sample.begin() + std::ptrdiff_t(i * d + b), pd, part.begin() + std::ptrdiff_t(i * pd));
        const KMeansResult r = kmeans(part, pd, KMeansParams{k, 25, 0x5051 + p, 0});  // the thread count never changes the result
        state.insert(state.end(), r.centroids.begin(), r.centroids.end());
    }
    restore(state);
}

std::vector<float> PQCodec::state() const {
    if (!trained()) return {};
    std::vector<float> s{float(centroids_count_)};
    s.insert(s.end(), centroids_.begin(), centroids_.end());
    return s;
}

void PQCodec::restore(std::span<const float> state) {
    if (state.empty() || !(state[0] >= 1 && state[0] <= 256) || state[0] != std::floor(state[0]))
        throw std::invalid_argument("PQ: saved state must start with 1 to 256 centroids per part");
    const std::size_t k = std::size_t(state[0]);
    if (state.size() != 1 + k * dim()) throw std::invalid_argument("PQ: saved state has wrong size");
    for (std::size_t i = 1; i < state.size(); ++i)
        if (!std::isfinite(state[i])) throw std::invalid_argument("PQ: centroids must be finite");
    centroids_.assign(state.begin() + 1, state.end());
    centroids_count_ = k;
}

void PQCodec::encode(std::span<const float> v, std::uint8_t* code) const {
    check_trained();
    check_size(v);
    for (std::size_t p = 0; p < parts_; ++p) {
        const std::size_t b = part_begin(p), pd = part_begin(p + 1) - b;
        const std::span<const float> cs(centroids_.data() + b * centroids_count_, centroids_count_ * pd);
        code[p] = static_cast<std::uint8_t>(nearest_centroid(cs, pd, v.data() + b));  // ties: the lower centroid
    }
}

void PQCodec::decode(const std::uint8_t* code, std::span<float> out) const {
    check_trained();
    check_size(out);
    for (std::size_t p = 0; p < parts_; ++p) {
        const std::size_t b = part_begin(p), pd = part_begin(p + 1) - b;
        if (code[p] >= centroids_count_) throw std::invalid_argument("PQ: code refers to a centroid that does not exist");
        std::copy_n(centroids_.data() + b * centroids_count_ + std::size_t(code[p]) * pd, pd, out.begin() + std::ptrdiff_t(b));
    }
}

void PQCodec::prepare(std::span<const float> query, std::vector<float>& table) const {
    check_trained();
    check_size(query);
    // For every part and centroid: its share of the distance (squared distance,
    // or negative dot product). 256 entries per part, unused ones never read.
    table.assign(parts_ * 256, 0.0f);
    for (std::size_t p = 0; p < parts_; ++p) {
        const std::size_t b = part_begin(p), pd = part_begin(p + 1) - b;
        for (std::size_t c = 0; c < centroids_count_; ++c) {
            const float* x = centroids_.data() + b * centroids_count_ + c * pd;
            float share = 0;
            for (std::size_t j = 0; j < pd; ++j)
                share += metric() == Metric::L2 ? (query[b + j] - x[j]) * (query[b + j] - x[j]) : -query[b + j] * x[j];
            table[p * 256 + c] = share;
        }
    }
}

float PQCodec::distance(const std::vector<float>& table, const std::uint8_t* code) const {
    float sum = 0;
    for (std::size_t p = 0; p < parts_; ++p) sum += table[p * 256 + code[p]];  // one lookup per part
    return metric() == Metric::Cosine ? 1.0f + sum : sum;  // cosine: 1 - dot, and sum is -dot
}

// ---------------------------------------------------------------- half precision

HalfCodec::HalfCodec(CodecKind kind, std::size_t dim, Metric metric) : VectorCodec(dim, metric), kind_(kind) {
    if (kind != CodecKind::F16 && kind != CodecKind::BF16) throw std::invalid_argument("HalfCodec: kind must be F16 or BF16");
}

void HalfCodec::restore(std::span<const float> state) {
    if (!state.empty()) throw std::invalid_argument("HalfCodec: has no state to restore");
}

void HalfCodec::train(std::span<const float> sample) {
    if (sample.size() % dim() != 0) throw std::invalid_argument("HalfCodec: sample must hold whole vectors");
}

void HalfCodec::check(std::span<const float> v) const {
    if (kind_ != CodecKind::F16) return;  // bfloat16 holds every finite float32 magnitude
    for (const float x : v)
        if (std::fabs(x) > kF16Max)
            throw std::invalid_argument("float16 holds values up to 65504 in magnitude; use bfloat16 (CodecKind::BF16) for larger values");
}

void HalfCodec::encode(std::span<const float> v, std::uint8_t* code) const {
    if (v.size() != dim()) throw std::invalid_argument("HalfCodec: vector has the wrong dimension");
    for (std::size_t j = 0; j < v.size(); ++j) {
        const std::uint16_t h = kind_ == CodecKind::F16 ? float_to_f16(v[j]) : float_to_bf16(v[j]);
        std::memcpy(code + 2 * j, &h, 2);
    }
}

void HalfCodec::decode(const std::uint8_t* code, std::span<float> out) const {
    const std::size_t n = std::min(out.size(), dim());
    const auto* c = reinterpret_cast<const std::uint16_t*>(code);  // 2-byte aligned (see distance_direct)
    if (kind_ == CodecKind::F16) kernels_.f16_decode(c, out.data(), n);  // SIMD where available
    else kernels_.bf16_decode(c, out.data(), n);
}

void HalfCodec::prepare(std::span<const float> query, std::vector<float>& table) const {
    table.assign(query.begin(), query.end());  // distances compare the float32 query directly
}

float HalfCodec::distance(const std::vector<float>& table, const std::uint8_t* code) const {
    return distance_direct(table.data(), code);  // the prepared query is the query itself
}

float HalfCodec::distance_direct(const float* query, const std::uint8_t* code) const {
    // Codes start at multiples of code_size() in byte arrays from operator new, so they
    // are 2-byte aligned.
    const auto* c = reinterpret_cast<const std::uint16_t*>(code);
    const bool f16 = kind_ == CodecKind::F16;
    if (metric() == Metric::L2) return f16 ? kernels_.f16_l2(query, c, dim()) : kernels_.bf16_l2(query, c, dim());
    const float dot = f16 ? kernels_.f16_dot(query, c, dim()) : kernels_.bf16_dot(query, c, dim());
    return metric() == Metric::Cosine ? 1.0f - dot : -dot;
}

bool HalfCodec::finite(const std::uint8_t* code) const {
    const std::uint16_t exponent = kind_ == CodecKind::F16 ? 0x7C00 : 0x7F80;  // all ones: infinity or NaN
    for (std::size_t j = 0; j < dim(); ++j) {
        std::uint16_t h;
        std::memcpy(&h, code + 2 * j, 2);
        if ((h & exponent) == exponent) return false;
    }
    return true;
}

// ---------------------------------------------------------------- RaBitQ

namespace {
std::size_t largest_power_of_two(std::size_t n) {
    std::size_t p = 1;
    while (p * 2 <= n) p *= 2;
    return p;
}
// The fast Walsh-Hadamard transform, unnormalized: a fixed order of additions.
void walsh_hadamard(float* x, std::size_t n) {
    for (std::size_t h = 1; h < n; h *= 2)
        for (std::size_t i = 0; i < n; i += 2 * h)
            for (std::size_t j = i; j < i + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
}
}  // namespace

RaBitQCodec::RaBitQCodec(std::size_t dim, Metric metric) : VectorCodec(dim, metric), block_(largest_power_of_two(dim)) {}

std::vector<float> RaBitQCodec::state() const {
    std::vector<float> s(centroid_);
    s.insert(s.end(), signs_.begin(), signs_.end());
    return s;
}

void RaBitQCodec::restore(std::span<const float> state) {
    if (state.size() != 4 * dim()) throw std::invalid_argument("RaBitQ: state of the wrong size");
    for (std::size_t j = 0; j < dim(); ++j)
        if (!std::isfinite(state[j])) throw std::invalid_argument("RaBitQ: centroid not finite");
    for (std::size_t j = dim(); j < state.size(); ++j)
        if (state[j] != 1.0f && state[j] != -1.0f) throw std::invalid_argument("RaBitQ: rotation signs must be +1 or -1");
    centroid_.assign(state.begin(), state.begin() + std::ptrdiff_t(dim()));
    signs_.assign(state.begin() + std::ptrdiff_t(dim()), state.end());
}

void RaBitQCodec::train(std::span<const float> sample) {
    const std::size_t d = dim();
    if (sample.empty() || sample.size() % d != 0) throw std::invalid_argument("RaBitQ: sample must hold whole vectors, at least one");
    const std::size_t n = sample.size() / d;
    std::vector<double> mean(d, 0.0);
    for (std::size_t i = 0; i < sample.size(); ++i) {
        if (!std::isfinite(sample[i])) throw std::invalid_argument("RaBitQ: sample contains NaN or infinity");
        mean[i % d] += sample[i];
    }
    std::vector<float> centroid(d);
    for (std::size_t j = 0; j < d; ++j) centroid[j] = static_cast<float>(mean[j] / double(n));
    // Signs from the raw 32-bit output, which the C++ standard fixes for mt19937.
    std::mt19937 rng(0x5AB17u + static_cast<unsigned>(d));
    std::vector<float> signs(3 * d);
    for (float& s : signs) s = (rng() & 1u) ? 1.0f : -1.0f;
    centroid_ = std::move(centroid);
    signs_ = std::move(signs);
}

void RaBitQCodec::rotate(std::vector<float>& x) const {
    const std::size_t d = dim(), p = block_;
    const float scale = 1.0f / std::sqrt(static_cast<float>(p));  // makes each transform orthonormal
    for (std::size_t round = 0; round < 3; ++round) {
        const float* s = signs_.data() + round * d;
        for (std::size_t j = 0; j < d; ++j) x[j] *= s[j];
        walsh_hadamard(x.data(), p);
        for (std::size_t j = 0; j < p; ++j) x[j] *= scale;
        if (p < d) {  // the last P coordinates too: together they cover every dimension
            walsh_hadamard(x.data() + (d - p), p);
            for (std::size_t j = d - p; j < d; ++j) x[j] *= scale;
        }
    }
}

void RaBitQCodec::unrotate(std::vector<float>& x) const {
    const std::size_t d = dim(), p = block_;
    const float scale = 1.0f / std::sqrt(static_cast<float>(p));
    for (std::size_t round = 3; round-- > 0;) {  // every step undone in reverse (H / sqrt(P) is its own inverse)
        if (p < d) {
            walsh_hadamard(x.data() + (d - p), p);
            for (std::size_t j = d - p; j < d; ++j) x[j] *= scale;
        }
        walsh_hadamard(x.data(), p);
        for (std::size_t j = 0; j < p; ++j) x[j] *= scale;
        const float* s = signs_.data() + round * d;
        for (std::size_t j = 0; j < d; ++j) x[j] *= s[j];
    }
}

void RaBitQCodec::encode(std::span<const float> v, std::uint8_t* code) const {
    check_trained();
    check_size(v);
    // The residual in double: in float, values near float's limit overflow (3e38 - -3e38),
    // and infinity / infinity would make the code NaN. The unit residual (every value in
    // [-1, 1]) is rotated; the length is stored capped at float's maximum.
    std::vector<double> r(dim());
    double norm2 = 0;
    for (std::size_t j = 0; j < dim(); ++j) {
        r[j] = double(v[j]) - double(centroid_[j]);
        norm2 += r[j] * r[j];
    }
    const double norm = std::sqrt(norm2);
    std::vector<float> u(dim(), 0.0f);
    if (norm > 0)
        for (std::size_t j = 0; j < dim(); ++j) u[j] = static_cast<float>(r[j] / norm);
    rotate(u);
    std::fill(code, code + bit_bytes(), std::uint8_t(0));
    double ip = 0;  // <x, u>: the bits as a unit vector against the unit residual
    if (norm > 0) {
        for (std::size_t j = 0; j < dim(); ++j) {
            if (u[j] > 0) code[j / 8] = static_cast<std::uint8_t>(code[j / 8] | (1u << (j % 8)));
            ip += std::abs(double(u[j]));
        }
        ip /= std::sqrt(double(dim()));
    }
    const float length = static_cast<float>(std::min(norm, double(std::numeric_limits<float>::max())));
    const float factors[2] = {length, norm > 0 ? static_cast<float>(std::min(ip, 1.0)) : 1.0f};
    std::memcpy(code + bit_bytes(), factors, sizeof factors);
}

void RaBitQCodec::decode(const std::uint8_t* code, std::span<float> out) const {
    check_trained();
    if (out.size() != dim()) throw std::invalid_argument("RaBitQ: decode needs dim values");
    float factors[2];
    std::memcpy(factors, code + bit_bytes(), sizeof factors);
    // The least-squares reconstruction along the bits' direction: c + |r| <x, u> P^-1 x.
    std::vector<float> x(dim());
    const float unit = 1.0f / std::sqrt(static_cast<float>(dim()));
    for (std::size_t j = 0; j < dim(); ++j) x[j] = ((code[j / 8] >> (j % 8)) & 1u) ? unit : -unit;
    unrotate(x);
    for (std::size_t j = 0; j < dim(); ++j) out[j] = centroid_[j] + factors[0] * factors[1] * x[j];
}

void RaBitQCodec::prepare(std::span<const float> query, std::vector<float>& table) const {
    check_trained();
    check_size(query);
    // L2 compares residuals from the centroid; inner product and cosine the query itself,
    // plus <c, q> for the centroid's part. Table: the rotated unit query (D values), then
    // its sum, the query's length, and <c, q>.
    // In double, then the unit vector rotated (as in encode): no overflow to NaN.
    std::vector<double> base(dim());
    double norm2 = 0, cq = 0;
    for (std::size_t j = 0; j < dim(); ++j) {
        base[j] = metric() == Metric::L2 ? double(query[j]) - double(centroid_[j]) : double(query[j]);
        norm2 += base[j] * base[j];
        cq += double(centroid_[j]) * double(query[j]);
    }
    const double norm = std::sqrt(norm2);
    table.assign(dim(), 0.0f);
    if (norm > 0)
        for (std::size_t j = 0; j < dim(); ++j) table[j] = static_cast<float>(base[j] / norm);
    rotate(table);
    double sum = 0;
    for (const float t : table) sum += t;
    const double kMax = double(std::numeric_limits<float>::max());  // beyond it: saturated, never infinity in the table
    table.push_back(static_cast<float>(sum));
    table.push_back(static_cast<float>(std::min(norm, kMax)));
    table.push_back(static_cast<float>(std::clamp(cq, -kMax, kMax)));
}

bool RaBitQCodec::valid_code(const std::uint8_t* code) const {
    float factors[2];
    std::memcpy(factors, code + bit_bytes(), sizeof factors);
    return std::isfinite(factors[0]) && factors[0] >= 0.0f && std::isfinite(factors[1]) && factors[1] >= 0.0f && factors[1] <= 1.001f;
}

float RaBitQCodec::distance(const std::vector<float>& table, const std::uint8_t* code) const {
    float factors[2];
    std::memcpy(factors, code + bit_bytes(), sizeof factors);
    const std::size_t d = dim();
    const float sum = table[d], qnorm = table[d + 1], cq = table[d + 2];
    // <x, u_q> from the set bits: (2 * sum over set bits - sum over all) / sqrt(d).
    const float xq = (2.0f * kernels_.bits_sum(table.data(), code, d) - sum) / std::sqrt(static_cast<float>(d));
    // <u_o, u_q>, a cosine: clamped to its range. encode() never writes <x, u> = 0, but a
    // zeroed code could hold it: 0, not a division by zero (std::clamp passes NaN through).
    const float estimate = factors[1] > 0 ? std::clamp(xq / factors[1], -1.0f, 1.0f) : 0.0f;
    // In double: lengths up to float's range square past it, and in float inf - inf is NaN;
    // in double the result is finite, and converting it gives at worst infinity, never NaN.
    const double no = factors[0], nq = qnorm, e = estimate;
    if (metric() == Metric::L2) return static_cast<float>(std::max(0.0, no * no + nq * nq - 2.0 * no * nq * e));
    const double dot = double(cq) + no * nq * e;
    return static_cast<float>(metric() == Metric::Cosine ? 1.0 - dot : -dot);
}

std::unique_ptr<VectorCodec> make_codec(CodecKind kind, std::size_t dim, Metric metric, std::size_t pq_subspaces) {
    switch (kind) {
        case CodecKind::None: return nullptr;
        case CodecKind::SQ8: return std::make_unique<SQ8Codec>(dim, metric);
        case CodecKind::Binary: return std::make_unique<BinaryCodec>(dim, metric);
        case CodecKind::PQ: return std::make_unique<PQCodec>(dim, metric, pq_subspaces);
        case CodecKind::F16:
        case CodecKind::BF16: return std::make_unique<HalfCodec>(kind, dim, metric);
        case CodecKind::RaBitQ: return std::make_unique<RaBitQCodec>(dim, metric);
    }
    throw std::invalid_argument("unknown codec kind");  // GCOVR_EXCL_LINE: every enumerator is handled above
}

}  // namespace vecdb