Using oneDNN with Threadpool-based Threading {#dev_guide_threadpool}
====================================================================

When oneDNN is built with the threadpool CPU runtime (see @ref
dev_guide_build_options), oneDNN requires a user to implement a threadpool
interface to enable the library to perform computations using multiple
threads.

The threadpool interface is defined in `include/dnnl_threadpool_iface.hpp`.
Below is a sample implementation based on Eigen threadpool that is also used
for testing (see `tests/testing_threadpool.hpp`).

~~~cpp
#include "dnnl_threadpool_iface.hpp"

class threadpool : public dnnl::threadpool_interop::threadpool_iface {
private:
    // Change to Eigen::NonBlockingThreadPool if using Eigen <= 3.3.7
    std::unique_ptr<Eigen::ThreadPool> tp_;

public:
    explicit threadpool(int num_threads = 0) {
        if (num_threads <= 0)
            num_threads = (int)std::thread::hardware_concurrency();
        tp_.reset(new Eigen::ThreadPool(num_threads));
    }
    int get_num_threads() override { return tp_->NumThreads(); }
    bool get_in_parallel() override {
        return tp_->CurrentThreadId() != -1;
    }
    uint64_t get_flags() override { return ASYNCHRONOUS; }
    void parallel_for(
            int n, const std::function<void(int, int)> &fn) override {
        for (int i = 0; i < n; i++)
            tp_->Schedule([i, n, fn]() { fn(i, n); });
    }
};
~~~
