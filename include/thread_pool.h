/**
 * @file thread_pool.h
 * @brief A small, reusable pool of worker threads for batch search.
 */
#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>

namespace vecdb {

/// The thread count a pool actually uses: 0 means one per hardware thread, and
/// requests are capped at max(64, 4 x hardware threads), since more threads
/// than that only add overhead and can exhaust the operating system's limits.
inline std::size_t effective_threads(std::size_t requested) {
    const std::size_t hw = std::max(1u, std::thread::hardware_concurrency());
    if (requested == 0) return hw;
    return std::min(requested, std::max<std::size_t>(64, 4 * hw));
}

/// True on a thread that is running a pool task. parallel_for() called there
/// (for example, by a predicate that runs a batch search) runs inline instead
/// of waiting for pool threads, which could all be busy waiting too.
inline bool& in_pool_task() noexcept {
    thread_local bool flag = false;
    return flag;
}

/**
 * @brief Fixed set of worker threads that run tasks from a shared queue.
 *
 * parallel_for(count, fn) runs fn(0) .. fn(count - 1) spread over the workers
 * and waits for all of them. If any call throws, the remaining indexes are
 * skipped and the first exception is rethrown to the caller once every worker
 * has stopped; the pool stays usable. Several threads may call parallel_for at
 * the same time. Called from inside any pool's task, it runs the calls inline,
 * in order, on the calling thread, so nested batches can never deadlock.
 *
 * The destructor finishes queued work, then joins the workers.
 */
class ThreadPool {
public:
    /// Starts `threads` workers (0 = one per hardware thread; capped, see effective_threads).
    explicit ThreadPool(std::size_t threads = 0) {
        threads = effective_threads(threads);
        workers_.reserve(threads);
        try {
            for (std::size_t i = 0; i < threads; ++i) workers_.emplace_back([this] { work(); });
        } catch (...) {
            shutdown();
            throw;
        }
    }

    ~ThreadPool() { shutdown(); }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    std::size_t size() const { return workers_.size(); }

    /// Runs fn(i) for every i in [0, count) on the pool and waits for all of them.
    void parallel_for(std::size_t count, const std::function<void(std::size_t)>& fn) {
        if (count == 0) return;
        if (in_pool_task()) {  // nested: run inline, so it cannot wait on busy workers
            for (std::size_t i = 0; i < count; ++i) fn(i);
            return;
        }
        struct Shared {
            std::atomic<std::size_t> next{0};
            std::mutex m;
            std::condition_variable done;
            std::size_t running = 0;
            std::exception_ptr error;
        };
        auto shared = std::make_shared<Shared>();
        const std::size_t tasks = std::min(count, workers_.size());
        auto task = [shared, count, &fn] {
            for (std::size_t i; (i = shared->next.fetch_add(1)) < count;) {
                try {
                    fn(i);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(shared->m);
                    if (!shared->error) shared->error = std::current_exception();
                    shared->next = count;  // skip the rest
                }
            }
            std::lock_guard<std::mutex> lock(shared->m);
            if (--shared->running == 0) shared->done.notify_all();
        };
        std::size_t submitted = 0;
        try {
            for (; submitted < tasks; ++submitted) {
                {
                    std::lock_guard<std::mutex> lock(shared->m);
                    ++shared->running;
                }
                try {
                    submit(task);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(shared->m);
                    --shared->running;
                    throw;
                }
            }
        } catch (...) {
            shared->next = count;  // stop submitted tasks early, then wait for them
            std::unique_lock<std::mutex> lock(shared->m);
            shared->done.wait(lock, [&] { return shared->running == 0; });
            throw;
        }
        std::unique_lock<std::mutex> lock(shared->m);
        shared->done.wait(lock, [&] { return shared->running == 0; });
        if (shared->error) std::rethrow_exception(shared->error);
    }

private:
    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(task));
        }
        ready_.notify_one();
    }

    void work() {
        in_pool_task() = true;
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) return;  // stopping and nothing left
                task = std::move(queue_.front());
                queue_.pop();
            }
            task();
        }
    }

    void shutdown() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        ready_.notify_all();
        for (std::thread& t : workers_)
            if (t.joinable()) t.join();
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> queue_;
    std::mutex mutex_;
    std::condition_variable ready_;
    bool stop_ = false;
};

/**
 * @brief Owns one ThreadPool shared by an index's batch searches, created on
 *        first use and recreated only if a different thread count is asked for.
 *        Callers keep the pool alive through the returned shared_ptr.
 */
class SharedPool {
public:
    std::shared_ptr<ThreadPool> get(std::size_t threads) {
        threads = effective_threads(threads);
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pool_ || pool_->size() != threads) pool_ = std::make_shared<ThreadPool>(threads);
        return pool_;
    }

private:
    std::mutex mutex_;
    std::shared_ptr<ThreadPool> pool_;
};


/**
 * @brief The lock inside every index: any number of reads (searches, saves,
 *        lookups) at once, or one change at a time.
 *
 * **Writers are preferred:** while a change waits, new reads wait too, so a
 * steady stream of searches can never starve a change. (std::shared_mutex is
 * not used because with glibc it prefers readers, and a change could then wait
 * forever.)
 *
 * **Nesting:** each thread records which indexes it is inside. Two rules make
 * deadlocks impossible, even with predicates that search other indexes:
 *  - **No change from inside any index operation.** A change requested while
 *    the thread is inside an operation on any index (a predicate that adds, to
 *    its own index or another) throws std::logic_error. Changes nested in reads
 *    are what create lock cycles between threads.
 *  - **Nested reads skip the writer queue.** A read requested from inside an
 *    index operation waits only for a change that is actually running, never
 *    for queued ones. A running change never waits for anything (rule 1), so the
 *    nested read always proceeds. Top-level reads still queue behind waiting
 *    changes, so changes cannot be starved.
 * A read nested in an operation on the same index does not lock again at all.
 * Batch tasks on pool threads inherit their caller's read with Hold.
 */
class IndexLock {
public:
    /// Reads the index: shared with other reads; free if this thread is already inside it.
    class ReadGuard {
    public:
        explicit ReadGuard(const IndexLock& lock) : lock_(lock), nested_(lock.held()) {
            if (nested_) return;
            lock_.lock_shared(inside_any());
            try {
                lock_.push();
            } catch (...) {
                lock_.unlock_shared();
                throw;
            }
        }
        ~ReadGuard() {
            if (nested_) return;
            lock_.pop();
            lock_.unlock_shared();
        }
        ReadGuard(const ReadGuard&) = delete;
        ReadGuard& operator=(const ReadGuard&) = delete;

    private:
        const IndexLock& lock_;
        bool nested_;
    };

    /// Changes the index: exclusive. Throws std::logic_error if this thread is
    /// inside an operation on any index.
    class WriteGuard {
    public:
        WriteGuard(const IndexLock& lock, const char* operation) : lock_(lock) {
            if (inside_any())
                throw std::logic_error(std::string("vecdb: ") + operation +
                                       " called from inside an index operation (for example a filter predicate); "
                                       "change indexes only outside searches");
            lock_.lock();
            try {
                lock_.push();
                // GCOVR_EXCL_START: unreachable, a change starts only when this thread is inside no index (nothing to overflow)
            } catch (...) {
                lock_.unlock();
                throw;
            }
            // GCOVR_EXCL_STOP
        }
        ~WriteGuard() {
            lock_.writes_.fetch_add(1, std::memory_order_release);  // still exclusive: counted before anyone else enters
            lock_.pop();
            lock_.unlock();
        }
        WriteGuard(const WriteGuard&) = delete;
        WriteGuard& operator=(const WriteGuard&) = delete;

    private:
        const IndexLock& lock_;
    };

    /// Marks this thread as inside the index without locking, for work done on
    /// behalf of a caller that already holds a read (batch tasks on pool threads).
    class Hold {
    public:
        explicit Hold(const IndexLock& lock) : lock_(lock) { lock_.push(); }
        ~Hold() { lock_.pop(); }
        Hold(const Hold&) = delete;
        Hold& operator=(const Hold&) = delete;

    private:
        const IndexLock& lock_;
    };

    ReadGuard read() const { return ReadGuard(*this); }
    WriteGuard write(const char* operation) const { return WriteGuard(*this, operation); }

    /// Whether this thread is inside an operation on this index.
    bool held() const {
        const Held& h = held_by_this_thread();
        return std::find(h.items, h.items + h.count, this) != h.items + h.count;
    }

    /// Whether this thread is inside an operation on any index.
    static bool inside_any() { return held_by_this_thread().count > 0; }

    /// Changes completed so far (one per write guard released). Equal before and after a
    /// span means nothing changed in it: work decided under the shared lock is still valid.
    std::uint64_t writes() const { return writes_.load(std::memory_order_acquire); }

    /// Maximum nesting of index operations within one thread (across all indexes).
    static constexpr std::size_t kMaxNesting = 64;

private:
    mutable std::atomic<std::uint64_t> writes_{0};
    struct Held {
        const IndexLock* items[kMaxNesting];
        std::size_t count = 0;
    };
    static Held& held_by_this_thread() {
        thread_local Held h;
        return h;
    }
    void push() const {
        Held& h = held_by_this_thread();
        if (h.count == kMaxNesting) throw std::logic_error("vecdb: index operations nested too deeply");
        h.items[h.count++] = this;
    }
    void pop() const {  // removes the most recent entry for this lock
        Held& h = held_by_this_thread();
        for (std::size_t i = h.count; i-- > 0;)  // GCOVR_EXCL_BR_LINE: always found, every pop follows its push
            if (h.items[i] == this) {
                std::copy(h.items + i + 1, h.items + h.count, h.items + i);
                --h.count;
                return;
            }
    }

    void lock_shared(bool nested) const {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [&] { return !writer_ && (nested || waiting_writers_ == 0); });
        ++readers_;
    }
    void unlock_shared() const {
        std::lock_guard<std::mutex> l(m_);
        if (--readers_ == 0) cv_.notify_all();
    }
    void lock() const {
        std::unique_lock<std::mutex> l(m_);
        ++waiting_writers_;
        cv_.wait(l, [&] { return !writer_ && readers_ == 0; });
        --waiting_writers_;
        writer_ = true;
    }
    void unlock() const {
        std::lock_guard<std::mutex> l(m_);
        writer_ = false;
        cv_.notify_all();
    }

    mutable std::mutex m_;
    mutable std::condition_variable cv_;
    mutable std::size_t readers_ = 0;
    mutable std::size_t waiting_writers_ = 0;
    mutable bool writer_ = false;
};

}  // namespace vecdb