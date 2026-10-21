/**
 * @file change_log.h
 * @brief An index's write-ahead log hook, shared by HnswIndex and FlatIndex.
 *
 * Every public change is logged under the index's write lock before it is
 * applied (begin), and a ChangeLog::Scope appends an abort record if the
 * operation then leaves by an exception. If the log cannot be written the
 * change is not made; if even an abort record cannot be written, the log is
 * marked failed and every later change is refused until rotate() (a
 * checkpoint, whose snapshot holds the true state) or a new attach().
 */
#pragma once
#include <cstdint>
#include <exception>
#include <optional>
#include <stdexcept>

#include "wal.h"
#include "wal_ops.h"

namespace vecdb {

class ChangeLog {
public:
    bool attached() const { return log_.has_value(); }
    void attach(WalWriter log) {
        if (log.next_lsn() == 0) throw std::invalid_argument("attach_log: LSN 0 is reserved (means not logged)");
        log_ = std::move(log);
        failed_ = false;
        synced_ = 0;
    }
    std::optional<WalWriter> detach() {
        std::optional<WalWriter> out = std::move(log_);
        log_.reset();
        failed_ = false;
        return out;
    }
    /// A new segment (a checkpoint): the snapshot written with it holds the true
    /// state, so a failed log is healed.
    void rotate(WalWriter log) {
        log_ = std::move(log);
        failed_ = false;
        synced_ = 0;
    }
    /// Durability::Synced: each record is flushed before its change is applied.
    void set_sync_each(bool on) { sync_each_ = on; }
    WalWriter& writer() { return *log_; }
    const WalWriter& writer() const { return *log_; }

    /// Logs an operation (0 if no log is attached). Caller holds the write lock.
    template <class Encode>
    std::uint64_t begin(std::uint8_t type, Encode&& encode) {
        if (!log_) return 0;
        if (failed_) throw std::runtime_error("index changes refused: its write-ahead log failed (checkpoint, or attach a new one)");
        const std::uint64_t lsn = log_->append(type, encode());
        if (sync_each_) log_->sync();
        return lsn;
    }

    /// Makes the log durable if it grew since the last flush (background flusher,
    /// closing). Caller holds the durable directory's checkpoint mutex.
    void flush() noexcept {
        if (!log_ || log_->broken()) return;  // GCOVR_EXCL_BR_LINE: called only for durable indexes, which always have a log
        const std::uint64_t size = log_->size();
        if (size == synced_) return;
        try {
            log_->sync();
            synced_ = size;
        } catch (...) {  // the writer is now broken: changes are refused until a checkpoint
        }
    }

    /// Appends an abort record for `lsn` if the scope is left by an exception.
    class Scope {
    public:
        Scope(ChangeLog& log, std::uint64_t lsn) : log_(log), lsn_(lsn), exceptions_(std::uncaught_exceptions()) {}
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        ~Scope() {
            if (lsn_ != 0 && std::uncaught_exceptions() > exceptions_) log_.abort(lsn_);
        }

    private:
        ChangeLog& log_;
        std::uint64_t lsn_;
        int exceptions_;
    };

private:
    void abort(std::uint64_t lsn) noexcept {
        try {
            (void)log_->append(walop::kAbort, walop::encode_abort(lsn));
            log_->sync();  // aborts are rare: on disk at once, so no operation record outlives its abort
        } catch (...) {
            // The log holds an operation that failed, without its abort record:
            // replaying it would apply a change never made.
            failed_ = true;
        }
    }

    std::optional<WalWriter> log_;
    bool failed_ = false;
    bool sync_each_ = false;
    std::uint64_t synced_ = 0;
};

}  // namespace vecdb
