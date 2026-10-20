/**
 * @file durable.h
 * @brief Durable index directories: snapshots plus a write-ahead log, so that
 *        acknowledged changes survive a crash (durability, phase D3).
 *
 * A directory holds:
 *   DURABLE                  magic, index kind and a random identity (created once)
 *   snapshot-<LSN>.idx       a saved index with every change before LSN
 *   wal-<LSN>.log            a log segment whose first record is LSN
 *   LOCK                     held while the index is open (one writer per directory)
 *
 * open_durable() creates a directory (writing a snapshot of the empty index, so
 * there is always one) or recovers it: the newest snapshot that loads, falling
 * back to an older one, then every logged change since, in order. A torn record
 * at the end of the last segment is discarded (a crash cut it); damage anywhere
 * else is refused (FormatError). checkpoint() starts a new segment, writes a
 * snapshot as of that point, and deletes what the kept snapshots no longer need.
 */
#pragma once
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <thread>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "wal.h"

namespace vecdb {

/// When logged changes reach the disk itself (they reach the operating system at
/// once in every mode, so a process crash loses nothing acknowledged).
enum class Durability {
    Buffered,  ///< flushed at checkpoints and when the index is closed: survives a process crash
    Grouped,   ///< and every group_interval by a background thread: survives power loss but the last interval
    Synced,    ///< flushed before each change is applied: survives power loss entirely (slowest)
};

struct DurableOptions {
    /// Snapshots kept (at least 1); the older ones are fallbacks if the newest
    /// cannot be read. Log segments are kept as long as a kept snapshot needs them.
    std::size_t keep_snapshots = 2;
    Durability durability = Durability::Grouped;
    std::chrono::milliseconds group_interval{10};  ///< Grouped: at least 1 ms
};

/// What open_durable() did.
struct RecoveryStats {
    bool created = false;            ///< a new directory (nothing recovered)
    std::uint64_t snapshot_lsn = 0;  ///< the snapshot recovery started from
    std::size_t fallbacks = 0;       ///< newer snapshots that could not be read
    std::size_t segments = 0;        ///< log segments read
    std::size_t replayed = 0;        ///< log records replayed (aborted ones included)
    std::uint64_t torn_bytes = 0;    ///< bytes of a torn last record discarded
    /// The last logged operation failed on replay: it was in progress when the
    /// process stopped (logged, failed, and stopped before its abort record), so
    /// it was dropped, and a checkpoint taken so that it is never replayed again.
    bool dropped_unfinished = false;
};

namespace detail {
/// A durable index's directory, its identity and lock.
struct DurableState {
    std::string dir;
    std::uint64_t identity = 0;
    FileLock lock;
    DurableOptions options;
    std::mutex checkpoint_mutex;  // one checkpoint (or background flush) at a time
    DurableState(std::string d, std::uint64_t id, FileLock l, DurableOptions o)
        : dir(std::move(d)), identity(id), lock(std::move(l)), options(o) {}
    DurableState(const DurableState&) = delete;
    DurableState& operator=(const DurableState&) = delete;

    /// `flush` makes the log durable (the index's; called under checkpoint_mutex).
    /// Grouped: started on a background thread, every group_interval.
    void start(std::function<void()> f) {
        flush = std::move(f);
        if (options.durability != Durability::Grouped) return;
        flusher = std::thread([this] {
            std::unique_lock<std::mutex> l(m);
            while (!stopping) {
                cv.wait_for(l, options.group_interval, [this] { return stopping; });
                const std::lock_guard<std::mutex> one(checkpoint_mutex);
                flush();
            }
        });
    }
    /// Stops the flusher and flushes one last time (closing the index).
    ~DurableState() {
        {
            const std::lock_guard<std::mutex> l(m);
            stopping = true;
        }
        cv.notify_all();
        if (flusher.joinable()) flusher.join();
        if (flush) {
            const std::lock_guard<std::mutex> one(checkpoint_mutex);
            flush();
        }
    }

private:
    std::function<void()> flush;
    std::thread flusher;
    std::mutex m;
    std::condition_variable cv;
    bool stopping = false;
};
}  // namespace detail

}  // namespace vecdb
