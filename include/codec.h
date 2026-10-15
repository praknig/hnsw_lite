/**
 * @file codec.h
 * @brief Vector codecs: compact encodings of vectors, with distances computed
 *        directly from the encoded form.
 *
 * A codec is trained once on sample vectors, then encodes each vector into
 * code_size() bytes. Searching compares a full-precision query with codes
 * (asymmetric distance, more accurate than comparing two codes): prepare()
 * builds a small table from the query once, and distance() compares that table
 * with one code. Distances follow the library's conventions, so they are
 * directly comparable with full-precision ones: squared distance for L2,
 * negative dot product for inner product, and 1 - dot product for cosine (whose
 * vectors are normalized before encoding).
 *
 * Encoding is deterministic: the same training data gives the same codes on
 * every CPU and compiler (codec.cpp is built without floating-point
 * contraction), because codes are saved in index files.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "distance.h"

namespace vecdb {

/// Which encoding an index uses for its vectors.
enum class CodecKind : std::uint8_t {
    None = 0,  ///< full-precision floats (no codec)
    SQ8 = 1,   ///< 8-bit scalar quantization: one byte per dimension
};

/// A trained encoding of vectors into compact codes. See the file comment.
class VectorCodec {
public:
    VectorCodec(std::size_t dim, Metric metric) : dim_(dim), metric_(metric) {
        if (dim == 0) throw std::invalid_argument("codec: dimension must be at least 1");
    }
    virtual ~VectorCodec() = default;

    virtual CodecKind kind() const = 0;
    std::size_t dim() const { return dim_; }
    Metric metric() const { return metric_; }

    /// Bytes per encoded vector.
    virtual std::size_t code_size() const = 0;

    virtual bool trained() const = 0;

    /// Learns the encoding from `sample` (n * dim values, n >= 1, all finite).
    /// Training again replaces the previous encoding. Throws std::invalid_argument.
    virtual void train(std::span<const float> sample) = 0;

    /// Encodes one vector (dim values) into code_size() bytes at `code`.
    /// Throws std::logic_error before training, std::invalid_argument for a wrong size.
    virtual void encode(std::span<const float> v, std::uint8_t* code) const = 0;

    /// The approximate vector a code stands for (dim values).
    virtual void decode(const std::uint8_t* code, std::span<float> out) const = 0;

    /// Prepares `query` (dim values) for distance(): done once per query.
    virtual void prepare(std::span<const float> query, std::vector<float>& table) const = 0;

    /// The distance between a prepared query and one code (see the file comment).
    virtual float distance(const std::vector<float>& table, const std::uint8_t* code) const = 0;

protected:
    void check_trained() const {
        if (!trained()) throw std::logic_error("codec: not trained yet");
    }
    void check_size(std::span<const float> v) const {
        if (v.size() != dim_) throw std::invalid_argument("codec: vector has wrong dimension");
    }

private:
    std::size_t dim_;
    Metric metric_;
};

/**
 * @brief 8-bit scalar quantization (SQ8): each dimension is mapped onto 256
 *        evenly spaced levels between its smallest and largest trained value.
 *
 * One byte per dimension: 4 times smaller than floats. A value outside the
 * trained range is clamped to it. A dimension whose trained values are all
 * equal is stored exactly. The rounding error of a value within the range is
 * at most half a level: (max - min) / 510.
 */
class SQ8Codec final : public VectorCodec {
public:
    SQ8Codec(std::size_t dim, Metric metric) : VectorCodec(dim, metric) {}

    CodecKind kind() const override { return CodecKind::SQ8; }
    std::size_t code_size() const override { return dim(); }
    bool trained() const override { return !min_.empty(); }
    void train(std::span<const float> sample) override;
    void encode(std::span<const float> v, std::uint8_t* code) const override;
    void decode(const std::uint8_t* code, std::span<float> out) const override;
    void prepare(std::span<const float> query, std::vector<float>& table) const override;
    float distance(const std::vector<float>& table, const std::uint8_t* code) const override;

    /// The trained range: value = minimum + code * step, per dimension.
    std::span<const float> minimum() const { return min_; }
    std::span<const float> step() const { return step_; }

    /// Sets a trained range directly (for example when loading a saved index).
    /// Throws std::invalid_argument for wrong sizes, negative steps or non-finite values.
    void set_range(std::span<const float> minimum, std::span<const float> step);

private:
    std::vector<float> min_, step_;
};

/// Creates an untrained codec of the given kind (nullptr for CodecKind::None).
std::unique_ptr<VectorCodec> make_codec(CodecKind kind, std::size_t dim, Metric metric);

}  // namespace vecdb