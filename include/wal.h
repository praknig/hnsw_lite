/**
 * @file wal.h
 * @brief The write-ahead log: an append-only sequence of records, so changes
 *        since the last snapshot survive a crash (durability, phase D1).
 *
 * A log segment is a header followed by records, all little-endian:
 *
 *   header (40 bytes): magic "HNSWWAL1", version (u32), reserved (u32),
 *                      index identity (u64), first LSN (u64), reserved (u32),
 *                      checksum of the preceding 36 bytes (u32)
 *   record:            payload length (u32), checksum (u32) of everything
 *                      after it, LSN (u64), type (u8), payload
 *
 * LSNs (log sequence numbers) are consecutive within a segment, starting at the
 * header's first LSN. The identity ties a log to one index, so a log can never
 * be replayed onto another.
 *
 * Reading after a crash distinguishes a torn tail (the last write cut short:
 * normal, and discarded) from corruption (an error):
 *  - an incomplete header, or a record extending past the end: torn tail;
 *  - an invalid record that ends exactly at the end of the file: torn tail (a
 *    partial write left garbage in place);
 *  - only zero bytes from the invalid point to the end: torn tail (file systems
 *    can leave zero-filled space after a crash);
 *  - an invalid record followed by more data, or an LSN out of sequence:
 *    corruption.
 */
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace vecdb {

/// One logged operation, as read back.
struct WalRecord {
    std::uint64_t lsn = 0;
    std::uint8_t type = 0;
    std::vector<unsigned char> payload;
};

/// What reading a segment found.
struct WalScan {
    std::uint64_t identity = 0;
    std::uint64_t first_lsn = 0;
    std::vector<WalRecord> records;  ///< every complete, valid record, in order
    std::uint64_t valid_end = 0;     ///< byte offset just after the last valid record
    std::uint64_t file_size = 0;
    /// A torn tail was found (and discarded): bytes from valid_end to file_size.
    bool torn_tail() const { return valid_end < file_size; }
};

namespace wal {
inline constexpr std::size_t kHeaderSize = 40;
inline constexpr std::size_t kRecordHeaderSize = 17;          // length, checksum, LSN, type
inline constexpr std::uint32_t kMaxPayload = 1u << 30;        // 1 GiB: larger lengths are garbage

/// Reads a whole segment. Throws FormatError for a damaged header or for
/// corruption before the end (see the rules above), std::runtime_error if the
/// file cannot be read.
WalScan read_segment(const std::string& path);

/// Makes a written file durable (as WalWriter::sync). Throws std::runtime_error.
void sync_path(const std::string& path);
/// Makes a directory's entries durable (new files, renames, removals): fsync of
/// the directory on POSIX; Windows journals them itself (nothing to do).
void sync_directory(const std::string& dir);
}  // namespace wal

/// An exclusive lock on a file, held until destroyed (flock on POSIX, LockFileEx
/// on Windows); the operating system releases it if the process dies. Throws
/// std::runtime_error if another process (or another FileLock) holds it.
class FileLock {
public:
    explicit FileLock(const std::string& path);
    FileLock(FileLock&& o) noexcept : handle_(o.handle_) { o.handle_ = -1; }
    FileLock& operator=(FileLock&&) = delete;
    FileLock(const FileLock&) = delete;
    ~FileLock();

private:
    std::intptr_t handle_ = -1;
};

/// Appends records to one segment. append() is not thread-safe (the index calls
/// it under its write lock); sync() may run concurrently with it. Each append writes one complete record; if a write fails part
/// way, the file is cut back to the end of the last complete record, so later
/// records never follow garbage. If that cut, or a sync, fails, the writer is
/// broken: every further append or sync throws std::runtime_error, and the
/// segment must be reopened (which discards any torn tail).
class WalWriter {
public:
    /// Creates a new segment (the file must not exist).
    static WalWriter create(const std::string& path, std::uint64_t identity, std::uint64_t first_lsn);
    /// Opens an existing segment for appending: reads it, cuts off a torn tail,
    /// and continues after its last record. Throws FormatError for corruption,
    /// or if `identity` differs from the segment's.
    static WalWriter open(const std::string& path, std::uint64_t identity);

    WalWriter(WalWriter&& other) noexcept;
    WalWriter& operator=(WalWriter&& other) noexcept;
    WalWriter(const WalWriter&) = delete;
    WalWriter& operator=(const WalWriter&) = delete;
    ~WalWriter();

    /// Appends one record and returns its LSN. Not yet durable: see sync().
    /// Throws std::invalid_argument for type 0 (reserved) or a payload over
    /// kMaxPayload, std::runtime_error if writing fails (the file is then cut
    /// back, or the writer broken).
    std::uint64_t append(std::uint8_t type, std::span<const unsigned char> payload);
    /// Makes every appended record durable (fdatasync on Linux, F_FULLFSYNC on
    /// macOS, FlushFileBuffers on Windows). A failure breaks the writer: after a
    /// failed sync the operating system may already have dropped the data.
    void sync();

    std::uint64_t next_lsn() const { return next_lsn_; }
    std::uint64_t size() const { return size_.load(); }
    bool broken() const { return broken_.load(); }
    const std::string& path() const { return path_; }

private:
    WalWriter() = default;
    void close() noexcept;
    std::string path_;
    std::intptr_t handle_ = -1;
    std::uint64_t next_lsn_ = 0;
    std::atomic<std::uint64_t> size_{0};  // atomic: a background flusher reads it while appends run
    std::atomic<bool> broken_{false};
};

/// Fault injection for tests: simulated disk errors in WalWriter. Each switch is
/// a countdown; -1 (the default) never fails. Costs one atomic read per call.
namespace wal_faults {
/// The write that would take the total of bytes written past `n` (from now)
/// writes only up to that point, then fails (like a full disk).
inline std::atomic<long long> fail_write_after_bytes{-1};
/// The next sync fails if this is 0 (counts down per sync).
inline std::atomic<long> fail_sync_after{-1};
/// The next cut-back after a failed write fails if this is 0 (counts down).
inline std::atomic<long> fail_truncate_after{-1};
/// Syncs performed (successful or not), for tests that check when data is flushed.
inline std::atomic<long> syncs{0};
inline void disarm() {
    fail_write_after_bytes = -1;
    fail_sync_after = -1;
    fail_truncate_after = -1;
}
}  // namespace wal_faults

}  // namespace vecdb
