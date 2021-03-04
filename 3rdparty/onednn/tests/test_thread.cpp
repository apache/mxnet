/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL

#include <mutex>

#ifdef _WIN32
#include <windows.h>
#else
#include <stdlib.h>
#endif

#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#include "src/common/counting_barrier.hpp"
#include "tests/test_thread.hpp"

#if !defined(DNNL_TEST_THREADPOOL_USE_TBB)
namespace dnnl {
namespace testing {
namespace {
inline int read_num_threads_from_env() {
    const char *env_num_threads = nullptr;
    const char *env_var_name = "OMP_NUM_THREADS";
#ifdef _WIN32
    // This is only required to avoid using _CRT_SECURE_NO_WARNINGS
    const size_t buf_size = 12;
    char buf[buf_size];
    size_t val_size = GetEnvironmentVariable(env_var_name, buf, buf_size);
    if (val_size > 0 && val_size < buf_size) env_num_threads = buf;
#else
    env_num_threads = ::getenv(env_var_name);
#endif

    int num_threads = 0;
    if (env_num_threads) {
        char *endp;
        int nt = strtol(env_num_threads, &endp, 10);
        if (*endp == '\0') num_threads = nt;
    }
    if (num_threads <= 0)
        num_threads = (int)std::thread::hardware_concurrency();
    return num_threads;
}
} // namespace
} // namespace testing
} // namespace dnnl
#endif

#ifdef DNNL_TEST_THREADPOOL_USE_EIGEN

#include <memory>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/ThreadPool"

#if !(EIGEN_WORLD_VERSION == 3 && EIGEN_MAJOR_VERSION == 3)
#error Unsupported Eigen version (need 3.3.x)
#endif

#if EIGEN_MINOR_VERSION >= 90
using EigenThreadPool = Eigen::ThreadPool;
#else
using EigenThreadPool = Eigen::NonBlockingThreadPool;
#endif

namespace dnnl {
namespace testing {

class threadpool : public dnnl::threadpool_interop::threadpool_iface {
private:
    std::unique_ptr<EigenThreadPool> tp_;

public:
    explicit threadpool(int num_threads = 0) {
        if (num_threads <= 0) num_threads = read_num_threads_from_env();
        tp_.reset(new EigenThreadPool(num_threads));
    }
    int get_num_threads() const override { return tp_->NumThreads(); }
    bool get_in_parallel() const override {
        return tp_->CurrentThreadId() != -1;
    }
    uint64_t get_flags() const override { return ASYNCHRONOUS; }
    void parallel_for(int n, const std::function<void(int, int)> &fn) override {
        int nthr = get_num_threads();
        int njobs = std::min(n, nthr);

        for (int i = 0; i < njobs; i++) {
            tp_->Schedule([i, n, njobs, fn]() {
                int start, end;
                impl::balance211(n, njobs, i, start, end);
                for (int j = start; j < end; j++)
                    fn(j, n);
            });
        }
    };
};

} // namespace testing
} // namespace dnnl

#elif DNNL_TEST_THREADPOOL_USE_TBB
#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"

namespace dnnl {
namespace testing {

class threadpool : public dnnl::threadpool_interop::threadpool_iface {
public:
    explicit threadpool(int num_threads = 0) { (void)num_threads; }
    int get_num_threads() const override {
        return tbb::this_task_arena::max_concurrency();
    }
    bool get_in_parallel() const override { return 0; }
    uint64_t get_flags() const override { return 0; }
    void parallel_for(int n, const std::function<void(int, int)> &fn) override {
        tbb::parallel_for(
                0, n, [&](int i) { fn(i, n); }, tbb::static_partitioner());
    }
};

} // namespace testing
} // namespace dnnl

#else

#include <atomic>
#include <thread>
#include <vector>
#include <condition_variable>

namespace dnnl {
namespace testing {

// Naiive synchronous threadpool:
// - Only a single parallel_for is executed at the same time.
// - Recursive parallel_for results in sequential execution.
class threadpool : public dnnl::threadpool_interop::threadpool_iface {
public:
    using task_func = std::function<void(int, int)>;

    explicit threadpool(int num_threads = 0) {
        if (num_threads <= 0) num_threads = read_num_threads_from_env();
        num_threads_ = num_threads;
        master_sense_ = 0;

        for (int i = 0; i < 2; i++) {
            tasks_[i].go_flag.store(0);
            tasks_[i].fn = nullptr;
            tasks_[i].n = 0;
        }

        barrier_init();
        workers_.reset(new std::vector<worker_data>(num_threads_));
        for (int i = 0; i < num_threads_; i++) {
            auto wd = &workers_->at(i);
            wd->thread_id = i;
            wd->tp = this;
            wd->thread.reset(new std::thread(worker_loop, &workers_->at(i)));
        }
        barrier_wait();
    }

    virtual ~threadpool() {
        std::unique_lock<std::mutex> l(master_mutex_);
        barrier_init();
        task_submit(nullptr, 0);
        for (int i = 0; i < num_threads_; i++)
            workers_->at(i).thread->join();
        barrier_wait();
    }

    virtual int get_num_threads() const { return num_threads_; }

    virtual bool get_in_parallel() const { return worker_self() != nullptr; }

    virtual uint64_t get_flags() const { return 0; }

    virtual void parallel_for(int n, const task_func &fn) {
        if (worker_self() != nullptr)
            task_execute(0, 1, &fn, n);
        else {
            std::unique_lock<std::mutex> l(master_mutex_);
            barrier_init();
            task_submit(&fn, n);
            barrier_wait();
        }
    }

private:
    int num_threads_;
    std::mutex master_mutex_;

    struct worker_data {
        int thread_id;
        threadpool *tp;
        std::condition_variable cv;
        std::unique_ptr<std::thread> thread;
    };
    std::unique_ptr<std::vector<worker_data>> workers_;
    static thread_local worker_data *worker_self_;
    worker_data *worker_self() const {
        return worker_self_ != nullptr && worker_self_->tp == this
                ? worker_self_
                : nullptr;
    }

    struct task_data {
        std::atomic<int> go_flag;
        const task_func *fn;
        int n;
    };
    int master_sense_;
    task_data tasks_[2];

    dnnl::impl::counting_barrier_t barrier_;

    void barrier_init() { barrier_.init(num_threads_); }

    void barrier_wait() {
        barrier_.wait();
        tasks_[master_sense_].go_flag.store(0);
        master_sense_ = !master_sense_;
    }

    void barrier_notify(int worker_sense) { barrier_.notify(); }

    void task_submit(const task_func *fn, int n) {
        tasks_[master_sense_].fn = fn;
        tasks_[master_sense_].n = n;
        tasks_[master_sense_].go_flag.store(1);
        for (int i = 0; i < num_threads_; i++) {
            workers_->at(i).cv.notify_one();
        }
    }

    void task_execute(int ithr, int nthr, const task_func *fn, int n) {
        if (fn != nullptr && n > 0) {
            int start, end;
            impl::balance211(n, nthr, ithr, start, end);
            for (int i = start; i < end; i++)
                (*fn)(i, n);
        }
    }

    static void worker_loop(worker_data *wd) {
        worker_self_ = wd;
        int worker_sense = 0;

        wd->tp->barrier_notify(worker_sense);

        bool time_to_exit = false;
        std::mutex m;
        std::unique_lock<std::mutex> l(m);

        do {
            worker_sense = !worker_sense;
            auto *t = &wd->tp->tasks_[worker_sense];
            wd->tp->workers_->at(wd->thread_id).cv.wait(l, [t]() {
                return t->go_flag.load() != 0;
            });
            wd->tp->task_execute(
                    wd->thread_id, wd->tp->num_threads_, t->fn, t->n);
            time_to_exit = t->fn == nullptr;
            wd->tp->barrier_notify(worker_sense);
        } while (!time_to_exit);
    }
};

thread_local threadpool::worker_data *threadpool::worker_self_ = nullptr;

} // namespace testing
} // namespace dnnl
#endif

namespace dnnl {

namespace testing {
// Threadpool singleton
dnnl::threadpool_interop::threadpool_iface *get_threadpool() {
    static dnnl::testing::threadpool tp;
    return &tp;
}

} // namespace testing

// Implement a dummy threadpools_utils protocol here so that it is picked up
// by parallel*() calls from the tests.
namespace impl {
namespace testing_threadpool_utils {
void activate_threadpool(dnnl::threadpool_interop::threadpool_iface *tp) {}
void deactivate_threadpool() {}
dnnl::threadpool_interop::threadpool_iface *get_active_threadpool() {
    return testing::get_threadpool();
}
} // namespace testing_threadpool_utils

} // namespace impl
} // namespace dnnl

#endif
