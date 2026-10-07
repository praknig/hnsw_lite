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

/**
 * @brief Fixed set of worker threads that run tasks from a shared queue.
 *
 * parallel_for(count, fn) runs fn(0) .. fn(count - 1) spread over the workers
 * and waits for all of them. If any call throws, the remaining indexes are
 * skipped and the first exception is rethrown to the caller once every worker
 * has stopped; the pool stays usable. Several threads may call parallel_for at
 * the same time. Calling it from inside a task can deadlock and is not allowed.
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

}  // namespace vecdb