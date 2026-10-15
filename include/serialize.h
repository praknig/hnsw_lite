/**
 * @file serialize.h
 * @brief The saved-file format: checksums, a writer and a reader for untrusted files.
 *
 * File layout (all numbers little-endian):
 *
 *   header:   magic "HNSWLITE" (8 bytes), format version (u32: this library writes
 *             2 and reads 1 and 2), byte-order mark
 *             (u32 0x01020304), index kind (u32: 1 Flat, 2 HNSW), reserved (u32, 0)
 *   sections: tag (4 ASCII bytes), payload length (u64), payload, CRC-32 of the payload (u32)
 *   last:     an "END " section with an empty payload; nothing may follow it
 *
 * Sections appear in a fixed order per index kind; see persistence.cpp.
 *
 * The reader treats the file as untrusted: every read is checked against the
 * bytes left in the section and in the file, and every count is checked
 * against those bytes before anything is allocated, so a damaged count cannot
 * trigger a huge allocation. Any problem throws FormatError.
 */
#pragma once
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ios>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace vecdb {

/// A saved file is damaged, truncated, from an unsupported version, or of the
/// wrong index kind. A failed load never produces a partly built index.
class FormatError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace persist {

inline constexpr char kMagic[8] = {'H', 'N', 'S', 'W', 'L', 'I', 'T', 'E'};
/// The version this library writes. Version 2 added, for cosine indexes, each
/// vector as given (before normalization) right after the stored vector.
inline constexpr std::uint32_t kVersion = 2;
/// The oldest version this library still reads.
inline constexpr std::uint32_t kOldestVersion = 1;
inline constexpr std::uint32_t kByteOrderMark = 0x01020304u;

enum class IndexKind : std::uint32_t { Flat = 1, Hnsw = 2 };

/// Section tag from 4 ASCII characters.
constexpr std::uint32_t tag(const char (&s)[5]) {
    return std::uint32_t(std::uint8_t(s[0])) | std::uint32_t(std::uint8_t(s[1])) << 8 |
           std::uint32_t(std::uint8_t(s[2])) << 16 | std::uint32_t(std::uint8_t(s[3])) << 24;
}

/// Updates a CRC-32 (the IEEE polynomial, as in zlib, PNG and Ethernet) with `n` bytes.
inline std::uint32_t crc32(std::uint32_t crc, const void* data, std::size_t n) {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    const auto* p = static_cast<const unsigned char*>(data);
    crc = ~crc;
    for (std::size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    return ~crc;
}

/// An integer or floating-point value in little-endian byte order.
template <class T>
void to_little_endian(T value, unsigned char* out) {
    static_assert(std::is_arithmetic_v<T>);
    using U = std::conditional_t<sizeof(T) == 1, std::uint8_t,
              std::conditional_t<sizeof(T) == 2, std::uint16_t,
              std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>>>;
    U bits;
    std::memcpy(&bits, &value, sizeof(T));
    for (std::size_t i = 0; i < sizeof(T); ++i) out[i] = static_cast<unsigned char>(bits >> (8 * i));
}

template <class T>
T from_little_endian(const unsigned char* in) {
    static_assert(std::is_arithmetic_v<T>);
    using U = std::conditional_t<sizeof(T) == 1, std::uint8_t,
              std::conditional_t<sizeof(T) == 2, std::uint16_t,
              std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>>>;
    U bits = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) bits = U(bits | U(U(in[i]) << (8 * i)));
    T value;
    std::memcpy(&value, &bits, sizeof(T));
    return value;
}

/**
 * @brief Writes one section's payload. With no stream it only counts bytes,
 *        which is how a section's length is found before it is written.
 */
class Writer {
public:
    explicit Writer(std::ostream* out) : out_(out) {}

    void bytes(const void* p, std::size_t n) {
        if (out_) {
            out_->write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
            if (!*out_) throw std::runtime_error("writing the index failed");  // GCOVR_EXCL_BR_LINE: defensive, StreamErrorsThrow throws first
            crc_ = crc32(crc_, p, n);
        }
        count_ += n;
    }

    template <class T>
    void put(T value) {
        unsigned char buf[sizeof(T)];
        to_little_endian(value, buf);
        bytes(buf, sizeof(T));
    }

    void put_string(std::string_view s) {
        if (s.size() > std::numeric_limits<std::uint32_t>::max())  // GCOVR_EXCL_BR_LINE: needs a string over 4 GB
            throw std::length_error("string too long to save");  // GCOVR_EXCL_LINE: needs a string over 4 GB
        put<std::uint32_t>(static_cast<std::uint32_t>(s.size()));
        bytes(s.data(), s.size());
    }

    void put_floats(const float* p, std::size_t n) {
        if constexpr (std::endian::native == std::endian::little) {
            bytes(p, n * sizeof(float));
        } else {
            for (std::size_t i = 0; i < n; ++i) put(p[i]);
        }
    }

    std::uint64_t count() const { return count_; }
    std::uint32_t crc() const { return crc_; }

private:
    std::ostream* out_;
    std::uint64_t count_ = 0;
    std::uint32_t crc_ = 0;
};

/**
 * @brief Makes a stream rethrow errors from inside its buffer (such as running
 *        out of memory) instead of only setting badbit, for the lifetime of this
 *        object; the stream's previous setting is restored afterwards.
 *
 * Without it, a stream that runs out of memory while growing quietly stops
 * writing, and a save would produce a damaged file without reporting an error.
 * A stream already in a failed state throws std::ios_base::failure at once.
 */
class StreamErrorsThrow {
public:
    explicit StreamErrorsThrow(std::ios& s) : s_(s), previous_(s.exceptions()) { s_.exceptions(previous_ | std::ios::badbit); }
    ~StreamErrorsThrow() {
        try {
            s_.exceptions(previous_);
            // GCOVR_EXCL_START: unreachable, restoring a less strict setting cannot throw
        } catch (...) {
        }
        // GCOVR_EXCL_STOP
    }
    StreamErrorsThrow(const StreamErrorsThrow&) = delete;
    StreamErrorsThrow& operator=(const StreamErrorsThrow&) = delete;

private:
    std::ios& s_;
    std::ios::iostate previous_;
};

/// Writes the file header.
inline void write_header(std::ostream& out, IndexKind kind) {
    unsigned char buf[24];
    std::memcpy(buf, kMagic, 8);
    to_little_endian(kVersion, buf + 8);
    to_little_endian(kByteOrderMark, buf + 12);
    to_little_endian(static_cast<std::uint32_t>(kind), buf + 16);
    to_little_endian(std::uint32_t(0), buf + 20);
    out.write(reinterpret_cast<const char*>(buf), sizeof buf);
    if (!out) throw std::runtime_error("writing the index failed");  // GCOVR_EXCL_BR_LINE: defensive, StreamErrorsThrow throws first
}

/// Writes a section: `body(writer)` is called twice, first only to count the
/// payload's bytes, then to write them.
template <class Body>
void write_section(std::ostream& out, std::uint32_t section_tag, Body&& body) {
    Writer counter(nullptr);
    body(counter);
    unsigned char head[12];
    to_little_endian(section_tag, head);
    to_little_endian(counter.count(), head + 4);
    out.write(reinterpret_cast<const char*>(head), sizeof head);
    if (!out) throw std::runtime_error("writing the index failed");  // GCOVR_EXCL_BR_LINE: defensive, StreamErrorsThrow throws first
    Writer w(&out);
    body(w);
    if (w.count() != counter.count()) throw std::logic_error("section length changed while writing");  // GCOVR_EXCL_BR_LINE: defensive
    unsigned char crc[4];
    to_little_endian(w.crc(), crc);
    out.write(reinterpret_cast<const char*>(crc), sizeof crc);
    if (!out) throw std::runtime_error("writing the index failed");  // GCOVR_EXCL_BR_LINE: defensive, StreamErrorsThrow throws first
}

/**
 * @brief Reads a saved file, treating it as untrusted.
 *
 * The stream must be seekable (files and string streams are), so the file's
 * size is known before reading: no count may claim more data than remains.
 */
class Reader {
public:
    explicit Reader(std::istream& in) : in_(in), errors_(in) {
        const auto start = in_.tellg();
        in_.seekg(0, std::ios::end);
        const auto end = in_.tellg();
        in_.seekg(start);
        if (start < 0 || end < start || !in_) throw FormatError("the stream must be seekable");  // GCOVR_EXCL_BR_LINE: only the first condition can fail in practice (tested)
        file_left_ = static_cast<std::uint64_t>(end - start);
    }

    /// Reads and checks the header; throws FormatError for anything unexpected.
    void read_header(IndexKind expected) {
        unsigned char buf[24];
        raw(buf, sizeof buf, "header");
        if (std::memcmp(buf, kMagic, 8) != 0) throw FormatError("not an hnsw-lite index file");
        const auto version = from_little_endian<std::uint32_t>(buf + 8);
        if (version < kOldestVersion || version > kVersion)
            throw FormatError("unsupported format version " + std::to_string(version) + " (this library reads versions " +
                              std::to_string(kOldestVersion) + " to " + std::to_string(kVersion) + ")");
        version_ = version;
        if (from_little_endian<std::uint32_t>(buf + 12) != kByteOrderMark) throw FormatError("unsupported byte order");
        const auto kind = from_little_endian<std::uint32_t>(buf + 16);
        if (kind != static_cast<std::uint32_t>(expected))
            throw FormatError(kind == 1 ? "the file holds a Flat index, not an HNSW index"
                              : kind == 2 ? "the file holds an HNSW index, not a Flat index"
                                          : "unknown index kind");
        if (from_little_endian<std::uint32_t>(buf + 20) != 0) throw FormatError("reserved header field is not zero");
    }

    /// Starts the next section, which must have tag `expected`.
    void begin_section(std::uint32_t expected, const char* name) {
        unsigned char head[12];
        raw(head, sizeof head, name);
        if (from_little_endian<std::uint32_t>(head) != expected) throw FormatError(std::string("expected section ") + name);
        const auto length = from_little_endian<std::uint64_t>(head + 4);
        if (length > file_left_) throw FormatError(std::string("section ") + name + " is longer than the file");
        section_left_ = length;
        crc_ = 0;
        name_ = name;
    }

    /// Ends the section: every byte must have been read, and the checksum must match.
    void end_section() {
        if (section_left_ != 0) throw FormatError(std::string("unexpected data at the end of section ") + name_);
        unsigned char buf[4];
        raw(buf, sizeof buf, name_);
        if (from_little_endian<std::uint32_t>(buf) != crc_)
            throw FormatError(std::string("checksum mismatch in section ") + name_ + " (the file is damaged)");
    }

    /// After the final section: the file must end here.
    void expect_end_of_file() {
        if (file_left_ != 0) throw FormatError("unexpected bytes after the end of the index");
    }

    void bytes(void* p, std::size_t n) {
        if (n > section_left_) throw FormatError(std::string("section ") + name_ + " ends early");
        raw(p, n, name_);
        section_left_ -= n;
        crc_ = crc32(crc_, p, n);
    }

    template <class T>
    T get() {
        unsigned char buf[sizeof(T)];
        bytes(buf, sizeof(T));
        return from_little_endian<T>(buf);
    }

    /// Reads a count of items that each take at least `min_bytes` (at least 1), rejecting it
    /// unless that many bytes remain in the section (checked before allocating).
    std::uint64_t get_count(std::uint64_t min_bytes, const char* what) {
        const auto n = get<std::uint64_t>();
        if (n > section_left_ / min_bytes)
            throw FormatError(std::string("count of ") + what + " exceeds the data in the file");
        return n;
    }

    std::string get_string() {
        const auto n = get<std::uint32_t>();
        if (n > section_left_) throw FormatError(std::string("string in section ") + name_ + " is longer than the section");
        std::string s(n, '\0');
        bytes(s.data(), n);
        return s;
    }  // GCOVR_EXCL_LINE: gcov counts this brace separately (return value optimization)

    /// Reads `n` floats; every value must be finite.
    void get_floats(float* out, std::size_t n) {
        if constexpr (std::endian::native == std::endian::little) {
            if (n > section_left_ / sizeof(float)) throw FormatError(std::string("section ") + name_ + " ends early");
            bytes(out, n * sizeof(float));
            // GCOVR_EXCL_START: big-endian machines only; CI machines are little-endian
        } else {
            for (std::size_t i = 0; i < n; ++i) out[i] = get<float>();
        }
        // GCOVR_EXCL_STOP
        for (std::size_t i = 0; i < n; ++i)
            if (!std::isfinite(out[i])) throw FormatError("a stored vector contains NaN or infinity");
    }

    std::uint64_t section_left() const { return section_left_; }

    /// The file's format version (after read_header).
    std::uint32_t version() const { return version_; }

private:
    void raw(void* p, std::size_t n, const char* where) {
        if (n > file_left_) throw FormatError(std::string("the file ends early (in ") + where + ")");
        in_.read(static_cast<char*>(p), static_cast<std::streamsize>(n));
        if (static_cast<std::size_t>(in_.gcount()) != n) throw FormatError(std::string("the file ends early (in ") + where + ")");  // GCOVR_EXCL_BR_LINE: defensive, the size was checked above
        file_left_ -= n;
    }

    std::istream& in_;
    StreamErrorsThrow errors_;  // running out of memory while reading is reported as such
    std::uint64_t file_left_ = 0;
    std::uint64_t section_left_ = 0;
    std::uint32_t crc_ = 0;
    const char* name_ = "header";
    std::uint32_t version_ = 0;
};

}  // namespace persist
}  // namespace vecdb