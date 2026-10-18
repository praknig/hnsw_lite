/**
 * @file codec.cpp
 * @brief Vector codecs; see codec.h. Built without floating-point contraction
 *        (see CMakeLists.txt), so codes are identical on every CPU.
 */
#include "codec.h"
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
        // q . (min + c * step) = sum q * min + sum (q * step) * c: d weights, then the constant
        table.resize(d + 1);
        float constant = 0;
        for (std::size_t j = 0; j < d; ++j) {
            table[j] = query[j] * step_[j];
            constant += query[j] * min_[j];
        }
        table[d] = constant;
    }
}

float SQ8Codec::distance(const std::vector<float>& table, const std::uint8_t* code) const {
    const std::size_t d = dim();
    if (metric() == Metric::L2) return kernels_.sq8_l2(table.data(), step_.data(), code, d);  // SIMD where available
    const float dot = table[d] + kernels_.sq8_dot(table.data(), code, d);
    return metric() == Metric::Cosine ? 1.0f - dot : -dot;
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

std::unique_ptr<VectorCodec> make_codec(CodecKind kind, std::size_t dim, Metric metric, std::size_t pq_subspaces) {
    switch (kind) {
        case CodecKind::None: return nullptr;
        case CodecKind::SQ8: return std::make_unique<SQ8Codec>(dim, metric);
        case CodecKind::Binary: return std::make_unique<BinaryCodec>(dim, metric);
        case CodecKind::PQ: return std::make_unique<PQCodec>(dim, metric, pq_subspaces);
    }
    throw std::invalid_argument("unknown codec kind");  // GCOVR_EXCL_LINE: every enumerator is handled above
}

}  // namespace vecdb