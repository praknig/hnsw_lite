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
    Binary = 2,  ///< binary quantization: one bit per dimension
    PQ = 3,      ///< product quantization: one byte per part of the vector
    F16 = 4,     ///< half precision (IEEE float16): two bytes per dimension, values up to 65504
    BF16 = 5,    ///< bfloat16: two bytes per dimension, float32's range with 8 bits of precision
};

/// How an index stores its vectors in compact form (see the compression and
/// index types plan). Full-precision vectors are kept as well, for re-ranking
/// and get_vector().
struct QuantizationParams {
    CodecKind codec = CodecKind::None;  ///< None: full precision only, as before
    /// Train the codec automatically once this many vectors are stored (0: only
    /// when train_codec() is called). Until trained, searches use full precision.
    std::size_t train_after = 1000;
    /// Re-rank the best candidates found with codes at full precision: better
    /// recall for a little more work. Without it, distances come from codes.
    bool rerank = true;
    /// Exact searches with re-ranking compare k * rerank_factor candidates by
    /// code before re-ranking them (at least 1).
    std::size_t rerank_factor = 4;
    /// Matryoshka truncation: index and search only the first truncate_to
    /// dimensions of each vector (re-normalized for cosine), for embeddings
    /// trained to work shortened. Vectors and queries are still given in full,
    /// and get_vector() returns them in full. 0, or the full dimension: off.
    std::size_t truncate_to = 0;
    /// Keep each vector exactly as given (cosine indexes, which normalize, and
    /// truncating indexes otherwise store a changed copy), so get_vector() returns
    /// it exactly. false saves that memory: get_vector(), get() and scan() then
    /// return the stored vector (normalized for cosine; the first truncate_to
    /// values when truncating). With a codec it also drops the full-precision
    /// vectors once the codec is trained, keeping only codes: re-ranking is then
    /// off, vectors read back decoded, and that first training is final.
    bool keep_full_vectors = true;
    /// Product quantization: the number of parts each vector is split into (one
    /// byte of code each). 0: one part per 8 dimensions. At most the (indexed)
    /// dimension; dimensions that do not divide evenly go to the first parts.
    std::size_t pq_subspaces = 0;
};

/// Whether an index keeps each vector as given besides the stored one: when it
/// changes them (cosine normalizes, truncation shortens) and keep_full_vectors.
inline std::size_t indexed_dim(const QuantizationParams& q, std::size_t dim);
inline bool keeps_originals(const QuantizationParams& q, Metric metric, std::size_t dim) {
    return q.keep_full_vectors && (metric == Metric::Cosine || indexed_dim(q, dim) < dim);
}

/// The number of dimensions an index stores and searches: `dim`, or the
/// truncation if one is set. Throws std::invalid_argument if the truncation
/// exceeds `dim`.
inline std::size_t indexed_dim(const QuantizationParams& q, std::size_t dim) {
    if (q.truncate_to > dim) throw std::invalid_argument("QuantizationParams::truncate_to exceeds the dimension");
    return q.truncate_to == 0 ? dim : q.truncate_to;
}

/// An inverted file (IVF) for FlatIndex: k-means divides the vectors into
/// `lists`, and a search scans only the `probes` lists nearest to the query
/// (more if they hold fewer than k matching vectors), instead of everything.
/// With a codec this is, for example, IVF-PQ. Range searches stay exact.
struct IvfParams {
    std::size_t lists = 0;          ///< 0: no inverted file (every search scans everything)
    std::size_t probes = 8;         ///< lists searched per query (SearchOptions::probes overrides)
    /// Train the lists automatically once this many vectors (and at least
    /// `lists`) are stored; 0: only when train_lists() is called.
    std::size_t train_after = 10000;
};

/// Throws std::invalid_argument for IVF settings that cannot work.
inline void validate(const IvfParams& p) {
    if (p.lists > 0 && p.probes < 1) throw std::invalid_argument("IvfParams::probes must be at least 1");
    if (p.lists > (std::size_t(1) << 31)) throw std::invalid_argument("IvfParams::lists is too large");
}

/// a * b, or SIZE_MAX if that would overflow (k * rerank_factor must never wrap).
inline std::size_t saturating_mul(std::size_t a, std::size_t b) {
    return (b != 0 && a > SIZE_MAX / b) ? SIZE_MAX : a * b;
}

/// Throws std::invalid_argument for settings that cannot work.
inline void validate(const QuantizationParams& q) {
    if (static_cast<int>(q.codec) > static_cast<int>(CodecKind::BF16)) throw std::invalid_argument("QuantizationParams: unknown codec");
    if (q.rerank_factor < 1) throw std::invalid_argument("QuantizationParams::rerank_factor must be at least 1");
}

/// A trained encoding of vectors into compact codes. See the file comment.
class VectorCodec {
public:
    VectorCodec(std::size_t dim, Metric metric) : dim_(dim), metric_(metric) {
        if (dim == 0) throw std::invalid_argument("codec: dimension must be at least 1");
    }
    virtual ~VectorCodec() = default;

    virtual CodecKind kind() const = 0;
    /// A copy, including the trained state.
    virtual std::unique_ptr<VectorCodec> clone() const = 0;

    /// The trained state as numbers, for saving (empty when untrained).
    virtual std::vector<float> state() const = 0;
    /// Restores a saved state. Throws std::invalid_argument if it is not valid.
    virtual void restore(std::span<const float> state) = 0;
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

    /// Throws std::invalid_argument if `v` (dim values, already validated as finite and,
    /// for cosine, normalized) cannot be encoded faithfully. Called before an insert or an
    /// update changes anything, since encoding itself happens in steps that must not
    /// throw. Only float16 refuses anything: values beyond its range (+-65504).
    virtual void check(std::span<const float> v) const { (void)v; }

    /// Whether distance_direct() is available: a distance between any float32 vector and
    /// a code with no prepare() step and no decoding (half precision: the prepared query
    /// is the vector itself). The index uses it to compare stored vectors quickly.
    virtual bool direct() const { return false; }
    /// The distance between `query` (dim values, prepared as the index prepares vectors)
    /// and a code, the same as distance() with a prepared query. Only when direct().
    virtual float distance_direct(const float* query, const std::uint8_t* code) const {
        (void)query;
        (void)code;
        return 0.0f;
    }

protected:
    /// The fastest code kernels for this CPU, chosen once (see distance.h).
    CodeKernels kernels_ = get_code_kernels();

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
    std::unique_ptr<VectorCodec> clone() const override { return std::make_unique<SQ8Codec>(*this); }
    /// The minimum of each dimension, then the step of each.
    std::vector<float> state() const override {
        std::vector<float> s(min_);
        s.insert(s.end(), step_.begin(), step_.end());
        return s;
    }
    void restore(std::span<const float> state) override {
        if (state.size() != 2 * dim()) throw std::invalid_argument("SQ8: saved state has wrong size");
        set_range(state.first(dim()), state.subspan(dim()));
    }
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

/**
 * @brief Binary quantization: one bit per dimension, 32 times smaller than floats.
 *
 * Training learns, for each dimension, a threshold (the mean of the sample)
 * and a spread (the mean absolute distance from it). A bit records whether a
 * value lies above its threshold, and decodes to threshold + spread or
 * threshold - spread. Distances compare the full-precision query with that
 * reconstruction (asymmetric, more accurate than comparing bits). Coarse:
 * meant to be combined with re-ranking at full precision.
 *
 * Decoding then encoding reproduces a code exactly, even for a dimension that
 * was constant in training (spread 0): a set bit then decodes to the smallest
 * value above the threshold.
 */
class BinaryCodec final : public VectorCodec {
public:
    BinaryCodec(std::size_t dim, Metric metric) : VectorCodec(dim, metric) {}

    CodecKind kind() const override { return CodecKind::Binary; }
    std::unique_ptr<VectorCodec> clone() const override { return std::make_unique<BinaryCodec>(*this); }
    /// The threshold of each dimension, then its spread.
    std::vector<float> state() const override {
        std::vector<float> s(threshold_);
        s.insert(s.end(), spread_.begin(), spread_.end());
        return s;
    }
    void restore(std::span<const float> state) override;
    std::size_t code_size() const override { return (dim() + 7) / 8; }
    bool trained() const override { return !threshold_.empty(); }
    void train(std::span<const float> sample) override;
    void encode(std::span<const float> v, std::uint8_t* code) const override;
    void decode(const std::uint8_t* code, std::span<float> out) const override;
    void prepare(std::span<const float> query, std::vector<float>& table) const override;
    float distance(const std::vector<float>& table, const std::uint8_t* code) const override;

private:
    float high(std::size_t j) const;  // what a set bit decodes to
    std::vector<float> threshold_, spread_;
};

/**
 * @brief Half precision: each value stored in 16 bits, float16 (CodecKind::F16) or
 * bfloat16 (CodecKind::BF16), rounded to the nearest representable value.
 *
 * Needs no training (trained() is always true). With keep_full_vectors = false the
 * float32 vectors are dropped from the first insert, halving vector memory; distances
 * are then computed between the float32 query and the 16-bit values (SIMD kernels:
 * F16C on AVX2, AVX-512, NEON). float16 keeps about 3.3 significant digits and refuses
 * values beyond +-65504 (check); bfloat16 keeps float32's range with about 2.4 digits.
 * Cosine vectors are normalized before encoding, so both suit every metric.
 */
class HalfCodec final : public VectorCodec {
public:
    HalfCodec(CodecKind kind, std::size_t dim, Metric metric);

    CodecKind kind() const override { return kind_; }
    std::unique_ptr<VectorCodec> clone() const override { return std::make_unique<HalfCodec>(*this); }
    std::vector<float> state() const override { return {}; }
    void restore(std::span<const float> state) override;
    std::size_t code_size() const override { return 2 * dim(); }
    bool trained() const override { return true; }
    void train(std::span<const float> sample) override;
    void encode(std::span<const float> v, std::uint8_t* code) const override;
    void decode(const std::uint8_t* code, std::span<float> out) const override;
    void prepare(std::span<const float> query, std::vector<float>& table) const override;
    float distance(const std::vector<float>& table, const std::uint8_t* code) const override;
    void check(std::span<const float> v) const override;
    bool direct() const override { return true; }
    float distance_direct(const float* query, const std::uint8_t* code) const override;

    /// Whether `code` (code_size() bytes) holds only finite values: false for the bit
    /// patterns of infinity and NaN, which a damaged file could contain.
    bool finite(const std::uint8_t* code) const;

private:
    CodecKind kind_;
};

/**
 * @brief Product quantization (PQ): each vector is split into parts, each part
 *        stored as the index of its nearest centroid, one byte.
 *
 * Training runs k-means (kmeans.h) on each part: 256 centroids, or as many as
 * there are training vectors if fewer. A query prepares a table of its
 * distance to every centroid of every part, so a distance is one lookup per
 * part. Encoding is deterministic on every CPU, and breaks ties toward the
 * lower centroid, so decoding then encoding reproduces a code exactly.
 */
class PQCodec final : public VectorCodec {
public:
    /// `parts`: the number of parts, 0 for one per 8 dimensions. Throws
    /// std::invalid_argument if it exceeds the dimension.
    PQCodec(std::size_t dim, Metric metric, std::size_t parts = 0);

    CodecKind kind() const override { return CodecKind::PQ; }
    std::unique_ptr<VectorCodec> clone() const override { return std::make_unique<PQCodec>(*this); }
    /// The number of centroids per part, then every part's centroids in order.
    std::vector<float> state() const override;
    void restore(std::span<const float> state) override;
    std::size_t code_size() const override { return parts_; }
    bool trained() const override { return centroids_count_ > 0; }
    void train(std::span<const float> sample) override;
    void encode(std::span<const float> v, std::uint8_t* code) const override;
    void decode(const std::uint8_t* code, std::span<float> out) const override;
    void prepare(std::span<const float> query, std::vector<float>& table) const override;
    float distance(const std::vector<float>& table, const std::uint8_t* code) const override;

    std::size_t parts() const { return parts_; }
    std::size_t centroids() const { return centroids_count_; }

private:
    std::size_t part_begin(std::size_t p) const;  // first dimension of part p
    std::size_t parts_, centroids_count_ = 0;
    std::vector<float> centroids_;  // part by part: centroids_count_ x part dimension
};

/// Creates an untrained codec of the given kind (nullptr for CodecKind::None);
/// `pq_subspaces` is used by PQ only.
std::unique_ptr<VectorCodec> make_codec(CodecKind kind, std::size_t dim, Metric metric, std::size_t pq_subspaces = 0);

}  // namespace vecdb