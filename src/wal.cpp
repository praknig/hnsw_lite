/**
 * @file wal.cpp
 * @brief Write-ahead log segments: writing, syncing, reading; see wal.h.
 */
#include "wal.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "serialize.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace vecdb {

namespace {

constexpr char kMagic[8] = {'H', 'N', 'S', 'W', 'W', 'A', 'L', '1'};
constexpr std::uint32_t kVersion = 1;

template <class T>
void put(unsigned char* at, T v) {
    persist::to_little_endian(v, at);
}
template <class T>
T get(const unsigned char* at) {
    return persist::from_little_endian<T>(at);
}

/// Counts down a fault switch: true when it fires (reaches 0 from 0).
bool fires(std::atomic<long>& countdown) {
    long v = countdown.load();
    while (v >= 0) {
        if (countdown.compare_exchange_weak(v, v - 1)) return v == 0;  // GCOVR_EXCL_BR_LINE: a retry needs a concurrent fault switch
    }
    return false;
}

// ---------------------------------------------------------------- platform I/O

std::intptr_t open_file(const std::string& path, bool create) {
#ifdef _WIN32
    HANDLE h = CreateFileW(std::filesystem::path(path).wstring().c_str(), GENERIC_READ | GENERIC_WRITE,
                           FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, create ? CREATE_NEW : OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open log segment " + path);
    return reinterpret_cast<std::intptr_t>(h);
#else
    const int fd = ::open(path.c_str(), O_RDWR | (create ? O_CREAT | O_EXCL : 0), 0644);
    if (fd < 0) throw std::runtime_error("cannot open log segment " + path);
    return fd;
#endif
}

void close_file(std::intptr_t h) noexcept {
#ifdef _WIN32
    CloseHandle(reinterpret_cast<HANDLE>(h));
#else
    ::close(static_cast<int>(h));
#endif
}

/// Writes up to n bytes at `offset`; returns how many were written (fewer on error).
std::size_t write_at(std::intptr_t h, std::uint64_t offset, const unsigned char* data, std::size_t n) {
    // Simulated full disk: write only part, then fail.
    long long budget = wal_faults::fail_write_after_bytes.load();
    std::size_t allowed = n;
    bool fail = false;
    if (budget >= 0) {
        if (static_cast<unsigned long long>(budget) < n) {
            allowed = static_cast<std::size_t>(budget);
            fail = true;
            wal_faults::fail_write_after_bytes = -1;
        } else {
            wal_faults::fail_write_after_bytes = budget - static_cast<long long>(n);
        }
    }
    std::size_t done = 0;
    while (done < allowed) {
#ifdef _WIN32
        OVERLAPPED ov{};
        const std::uint64_t at = offset + done;
        ov.Offset = static_cast<DWORD>(at & 0xFFFFFFFFu);
        ov.OffsetHigh = static_cast<DWORD>(at >> 32);
        DWORD wrote = 0;
        const DWORD want = static_cast<DWORD>(std::min<std::size_t>(allowed - done, 1u << 30));
        if (!WriteFile(reinterpret_cast<HANDLE>(h), data + done, want, &wrote, &ov) || wrote == 0) break;  // GCOVR_EXCL_LINE: a real disk error
        done += wrote;
#else
        const ssize_t wrote = ::pwrite(static_cast<int>(h), data + done, allowed - done, static_cast<off_t>(offset + done));
        if (wrote <= 0) break;  // GCOVR_EXCL_LINE: a real disk error
        done += static_cast<std::size_t>(wrote);
#endif
    }
    return fail ? std::min(done, allowed) : done;
}

bool truncate_to(std::intptr_t h, std::uint64_t size) {
    if (fires(wal_faults::fail_truncate_after)) return false;
#ifdef _WIN32
    LARGE_INTEGER at;
    at.QuadPart = static_cast<LONGLONG>(size);
    return SetFilePointerEx(reinterpret_cast<HANDLE>(h), at, nullptr, FILE_BEGIN) && SetEndOfFile(reinterpret_cast<HANDLE>(h));
#else
    return ::ftruncate(static_cast<int>(h), static_cast<off_t>(size)) == 0;
#endif
}

bool sync_file(std::intptr_t h) {
    ++wal_faults::syncs;
    if (fires(wal_faults::fail_sync_after)) return false;
#ifdef _WIN32
    return FlushFileBuffers(reinterpret_cast<HANDLE>(h)) != 0;
#elif defined(__APPLE__)
    // A plain fsync on macOS does not reach the disk's own cache; F_FULLFSYNC does.
    if (::fcntl(static_cast<int>(h), F_FULLFSYNC) == 0) return true;
    return ::fsync(static_cast<int>(h)) == 0;  // GCOVR_EXCL_LINE: file systems without F_FULLFSYNC
#elif defined(__linux__)
    return ::fdatasync(static_cast<int>(h)) == 0;
#else
    return ::fsync(static_cast<int>(h)) == 0;
#endif
}

// ---------------------------------------------------------------- reading

/// Reads a segment; `keep` false skips storing payloads (for reopening).
WalScan scan(const std::string& path, bool keep) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read log segment " + path);
    WalScan s;
    s.file_size = std::filesystem::file_size(path);
    unsigned char h[wal::kHeaderSize];
    if (s.file_size < wal::kHeaderSize || !in.read(reinterpret_cast<char*>(h), wal::kHeaderSize))  // GCOVR_EXCL_BR_LINE: a read fails only on a disk error
        throw FormatError("log segment header incomplete");
    if (std::memcmp(h, kMagic, 8) != 0) throw FormatError("not a log segment (wrong magic)");
    if (get<std::uint32_t>(h + 36) != persist::crc32(0, h, 36)) throw FormatError("log segment header checksum mismatch");
    if (get<std::uint32_t>(h + 8) != kVersion) throw FormatError("unsupported log segment version");
    s.identity = get<std::uint64_t>(h + 16);
    s.first_lsn = get<std::uint64_t>(h + 24);
    std::uint64_t pos = wal::kHeaderSize, expected = s.first_lsn;
    std::vector<unsigned char> payload;
    // From `from` to the end of the file: only zero bytes?
    auto zeros_to_end = [&](std::uint64_t from) {
        std::ifstream z(path, std::ios::binary);
        z.seekg(static_cast<std::streamoff>(from));
        char buf[4096];
        while (z.read(buf, sizeof buf) || z.gcount() > 0) {
            for (std::streamsize i = 0; i < z.gcount(); ++i)
                if (buf[i] != 0) return false;
        }
        return true;
    };
    while (pos < s.file_size) {
        unsigned char r[wal::kRecordHeaderSize];
        if (s.file_size - pos < wal::kRecordHeaderSize) break;  // incomplete record header: torn tail
        in.seekg(static_cast<std::streamoff>(pos));
        in.read(reinterpret_cast<char*>(r), wal::kRecordHeaderSize);
        const std::uint32_t length = get<std::uint32_t>(r);
        const std::uint64_t end = pos + wal::kRecordHeaderSize + length;
        if (length > wal::kMaxPayload) {  // garbage length: its end is unknown
            if (zeros_to_end(pos)) break;  // GCOVR_EXCL_BR_LINE: zeros make a length of 0, never too large
            throw FormatError("log segment corrupt (invalid record length) at offset " + std::to_string(pos));
        }
        if (end > s.file_size) break;  // extends past the end: torn tail
        payload.resize(length);
        if (length > 0) in.read(reinterpret_cast<char*>(payload.data()), length);
        std::uint32_t crc = persist::crc32(0, r + 8, 9);
        if (length > 0) crc = persist::crc32(crc, payload.data(), length);
        const std::uint64_t lsn = get<std::uint64_t>(r + 8);
        const std::uint8_t type = r[16];
        const bool valid = crc == get<std::uint32_t>(r + 4) && lsn == expected && type != 0;
        if (!valid) {
            if (end == s.file_size || zeros_to_end(pos)) break;  // a torn last record, or zero-filled space
            throw FormatError(std::string("log segment corrupt (") + (crc != get<std::uint32_t>(r + 4) ? "checksum" : "sequence") +
                              ") at offset " + std::to_string(pos));
        }
        if (keep) s.records.push_back({lsn, type, payload});
        else s.records.push_back({lsn, type, {}});
        pos = end;
        ++expected;
    }
    s.valid_end = pos;
    // A tail too short for a record header is torn as well (or zero-filled).
    if (s.valid_end > s.file_size) s.valid_end = s.file_size;  // GCOVR_EXCL_LINE: cannot happen (pos only advances to record ends)
    return s;
}

}  // namespace

WalScan wal::read_segment(const std::string& path) { return scan(path, true); }

// ---------------------------------------------------------------- writing

WalWriter WalWriter::create(const std::string& path, std::uint64_t identity, std::uint64_t first_lsn) {
    WalWriter w;
    w.path_ = path;
    w.handle_ = open_file(path, true);
    unsigned char h[wal::kHeaderSize] = {};
    std::memcpy(h, kMagic, 8);
    put<std::uint32_t>(h + 8, kVersion);
    put<std::uint64_t>(h + 16, identity);
    put<std::uint64_t>(h + 24, first_lsn);
    put<std::uint32_t>(h + 36, persist::crc32(0, h, 36));
    if (write_at(w.handle_, 0, h, sizeof h) != sizeof h) {
        w.close();
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        throw std::runtime_error("cannot write log segment header " + path);
    }
    w.size_ = wal::kHeaderSize;
    w.next_lsn_ = first_lsn;
    return w;
}

WalWriter WalWriter::open(const std::string& path, std::uint64_t identity) {
    const WalScan s = scan(path, false);
    if (s.identity != identity) throw FormatError("log segment belongs to another index");
    WalWriter w;
    w.path_ = path;
    w.handle_ = open_file(path, false);
    if (s.torn_tail() && !truncate_to(w.handle_, s.valid_end)) {
        w.close();
        throw std::runtime_error("cannot cut the torn tail off log segment " + path);
    }
    w.size_ = s.valid_end;
    w.next_lsn_ = s.first_lsn + s.records.size();
    return w;
}

WalWriter::WalWriter(WalWriter&& o) noexcept
    : path_(std::move(o.path_)), handle_(o.handle_), next_lsn_(o.next_lsn_), size_(o.size_.load()), broken_(o.broken_.load()) {
    o.handle_ = -1;
}

WalWriter& WalWriter::operator=(WalWriter&& o) noexcept {
    if (this != &o) {
        close();
        path_ = std::move(o.path_);
        handle_ = o.handle_;
        next_lsn_ = o.next_lsn_;
        size_ = o.size_.load();
        broken_ = o.broken_.load();
        o.handle_ = -1;
    }
    return *this;
}

WalWriter::~WalWriter() { close(); }

void WalWriter::close() noexcept {
    if (handle_ != -1) close_file(handle_);
    handle_ = -1;
}

std::uint64_t WalWriter::append(std::uint8_t type, std::span<const unsigned char> payload) {
    if (broken_) throw std::runtime_error("log segment " + path_ + " is broken: reopen it");
    if (type == 0) throw std::invalid_argument("log record type 0 is reserved");
    if (payload.size() > wal::kMaxPayload) throw std::invalid_argument("log record payload larger than 1 GiB");  // GCOVR_EXCL_BR_LINE: needs a 1 GiB payload
    std::vector<unsigned char> rec(wal::kRecordHeaderSize + payload.size());
    put<std::uint32_t>(rec.data(), static_cast<std::uint32_t>(payload.size()));
    put<std::uint64_t>(rec.data() + 8, next_lsn_);
    rec[16] = type;
    if (!payload.empty()) std::memcpy(rec.data() + wal::kRecordHeaderSize, payload.data(), payload.size());
    put<std::uint32_t>(rec.data() + 4, persist::crc32(0, rec.data() + 8, 9 + payload.size()));
    const std::uint64_t at = size_.load();
    if (write_at(handle_, at, rec.data(), rec.size()) != rec.size()) {
        // Cut back to the last complete record, so that nothing follows garbage.
        if (!truncate_to(handle_, at)) broken_ = true;
        throw std::runtime_error("cannot write to log segment " + path_ + (broken_.load() ? " (and cannot cut it back: broken)" : ""));
    }
    size_ = at + rec.size();
    return next_lsn_++;
}

void WalWriter::sync() {
    if (broken_) throw std::runtime_error("log segment " + path_ + " is broken: reopen it");
    if (!sync_file(handle_)) {
        broken_ = true;  // after a failed sync the operating system may have dropped the data
        throw std::runtime_error("cannot sync log segment " + path_ + " (broken: reopen it)");
    }
}

// ---------------------------------------------------------------- files and directories

void wal::sync_path(const std::string& path) {
    const std::intptr_t h = open_file(path, false);
    const bool ok = sync_file(h);
    close_file(h);
    if (!ok) throw std::runtime_error("cannot sync " + path);
}

void wal::sync_directory(const std::string& dir) {
#ifdef _WIN32
    (void)dir;  // NTFS journals directory entries; there is no directory handle to flush
#else
    const int fd = ::open(dir.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open directory " + dir);
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    if (!ok) throw std::runtime_error("cannot sync directory " + dir);  // GCOVR_EXCL_LINE: a real disk error
#endif
}

FileLock::FileLock(const std::string& path) {
#ifdef _WIN32
    HANDLE h = CreateFileW(std::filesystem::path(path).wstring().c_str(), GENERIC_READ | GENERIC_WRITE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open lock file " + path);
    OVERLAPPED ov{};
    if (!LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &ov)) {
        CloseHandle(h);
        throw std::runtime_error("index directory is in use (locked): " + path);
    }
    handle_ = reinterpret_cast<std::intptr_t>(h);
#else
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) throw std::runtime_error("cannot open lock file " + path);
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        ::close(fd);
        throw std::runtime_error("index directory is in use (locked): " + path);
    }
    handle_ = fd;
#endif
}

FileLock::~FileLock() {
    if (handle_ == -1) return;
#ifdef _WIN32
    OVERLAPPED ov{};
    UnlockFileEx(reinterpret_cast<HANDLE>(handle_), 0, 1, 0, &ov);
#endif
    close_file(handle_);  // closing releases a flock
}

}  // namespace vecdb