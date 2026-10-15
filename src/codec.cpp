/**
 * @file codec.cpp
 * @brief Vector codecs; see codec.h. Built without floating-point contraction
 *        (see CMakeLists.txt), so codes are identical on every CPU.
 */
#include "codec.h"

#include <algorithm>
#include <cmath>

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
    for (std::size_t j = 0; j < d; ++j) step[j] = (hi[j] - lo[j]) / 255.0f;
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
        if (step_[j] > 0) level = std::floor((v[j] - min_[j]) / step_[j] + 0.5f);  // round half up: same everywhere
        code[j] = static_cast<std::uint8_t>(std::clamp(level, 0.0f, 255.0f));     // outside the range: clamped
    }
}

void SQ8Codec::decode(const std::uint8_t* code, std::span<float> out) const {
    check_trained();
    check_size(out);
    for (std::size_t j = 0; j < dim(); ++j) out[j] = min_[j] + float(code[j]) * step_[j];
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
    if (metric() == Metric::L2) {
        float sum = 0;
        for (std::size_t j = 0; j < d; ++j) {
            const float diff = table[j] - float(code[j]) * step_[j];
            sum += diff * diff;
        }
        return sum;
    }
    float dot = table[d];
    for (std::size_t j = 0; j < d; ++j) dot += table[j] * float(code[j]);
    return metric() == Metric::Cosine ? 1.0f - dot : -dot;
}

std::unique_ptr<VectorCodec> make_codec(CodecKind kind, std::size_t dim, Metric metric) {
    switch (kind) {
        case CodecKind::None: return nullptr;
        case CodecKind::SQ8: return std::make_unique<SQ8Codec>(dim, metric);
    }
    throw std::invalid_argument("unknown codec kind");  // GCOVR_EXCL_LINE: every enumerator is handled above
}

}  // namespace vecdb