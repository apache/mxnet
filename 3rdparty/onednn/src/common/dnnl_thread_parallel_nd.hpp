/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef COMMON_DNNL_THREAD_PARALLEL_ND_HPP
#define COMMON_DNNL_THREAD_PARALLEL_ND_HPP

/* This header must be included by dnnl_thread.hpp only */

/* Functions:
 *  - parallel(nthr, f)                  - executes f in parallel using at
 *                                         most nthr threads. If nthr equals
 *                                         0 dnnl_get_max_threads() threads
 *                                         is used
 *  - for_nd(ithr, nthr, dims..., f)     - multidimensional for loop for
 *                                         already created threads
 *  - for_nd_ext(ithr, nthr, dims..., f) - multidimensional for loop for
 *                                         already created threads that passes
 *                                         ithr and nthr
 *  - parallel_nd(dims..., f)            - creates a parallel section and then
 *                                         calls for_nd
 *  - parallel_nd_ext(dims..., f)        - creates a parallel section and then
 *                                         calls for_nd_ext
 *  - parallel_nd_in_omp(dims..., f)     - queries current nthr and ithr and
 *                                         then calls for_nd (mostly for
 *                                         convenience)
 */

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "counting_barrier.hpp"
#endif

namespace dnnl {
namespace impl {

namespace {
inline int adjust_num_threads(int nthr, size_t work_amount) {
    if (nthr == 0) nthr = dnnl_get_max_threads();
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
    return (work_amount == 1 || omp_in_parallel()) ? 1 : nthr;
#else
    return (int)std::min((size_t)nthr, work_amount);
#endif
}
} // namespace

/* general parallelization */
template <typename F>
void parallel(int nthr, F f) {
    nthr = adjust_num_threads(nthr, SIZE_MAX);
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
    assert(nthr == 1);
    f(0, 1);
#else
    if (nthr == 1) {
        f(0, 1);
        return;
    }
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#pragma omp parallel num_threads(nthr)
    {
        int nthr_ = omp_get_num_threads();
        int ithr_ = omp_get_thread_num();
        assert(nthr_ == nthr);
        f(ithr_, nthr_);
    }
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
    tbb::parallel_for(
            0, nthr, [&](int ithr) { f(ithr, nthr); },
            tbb::static_partitioner());
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    using namespace dnnl::impl::threadpool_utils;
    dnnl::threadpool_interop::threadpool_iface *tp = get_active_threadpool();
    if (!tp || dnnl_in_parallel()) {
        threadpool_utils::deactivate_threadpool();
        for (int ithr = 0; ithr < nthr; ithr++)
            f(ithr, nthr);
        threadpool_utils::activate_threadpool(tp);
    } else {
        bool async = tp->get_flags()
                & dnnl::threadpool_interop::threadpool_iface::ASYNCHRONOUS;
        counting_barrier_t b;
        if (async) b.init(nthr);
        tp->parallel_for(nthr, [tp, &f, &b, async](int ithr, int nthr) {
            bool is_master = threadpool_utils::get_active_threadpool() == tp;
            if (!is_master) threadpool_utils::activate_threadpool(tp);
            f(ithr, nthr);
            if (!is_master) threadpool_utils::deactivate_threadpool();
            if (async) b.notify();
        });
        if (async) b.wait();
    }
#endif
#endif
}

/* for_nd section */

template <typename T0, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, F f) {
    T0 start {0}, end {0};
    balance211(D0, nthr, ithr, start, end);
    for (T0 d0 = start; d0 < end; ++d0)
        f(d0);
}

template <typename T0, typename T1, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, const T1 &D1, F f) {
    const size_t work_amount = (size_t)D0 * D1;
    if (work_amount == 0) return;
    size_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0 {0};
    T1 d1 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1);
        utils::nd_iterator_step(d0, D0, d1, D1);
    }
}

template <typename T0, typename T1, typename T2, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2;
    if (work_amount == 0) return;
    size_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0 {0};
    T1 d1 {0};
    T2 d2 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, const T3 &D3, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3;
    if (work_amount == 0) return;
    size_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0 {0};
    T1 d1 {0};
    T2 d2 {0};
    T3 d3 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
        typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, const T3 &D3, const T4 &D4, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4;
    if (work_amount == 0) return;
    size_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0 {0};
    T1 d1 {0};
    T2 d2 {0};
    T3 d3 {0};
    T4 d4 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3, d4);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
        typename T5, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, const T3 &D3, const T4 &D4, const T5 &D5, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4 * D5;
    if (work_amount == 0) return;
    size_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0 {0};
    T1 d1 {0};
    T2 d2 {0};
    T3 d3 {0};
    T4 d4 {0};
    T5 d5 {0};
    utils::nd_iterator_init(
            start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3, d4, d5);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    }
}

/* for_nd_ext section */

template <typename T0, typename F>
void for_nd_ext(const int ithr, const int nthr, const T0 &D0, F f) {
    T0 start {0}, end {0};
    balance211(D0, nthr, ithr, start, end);
    for (T0 d0 = start; d0 < end; ++d0)
        f(ithr, nthr, d0);
}

template <typename T0, typename T1, typename F>
void for_nd_ext(
        const int ithr, const int nthr, const T0 &D0, const T1 &D1, F f) {
    const size_t work_amount = (size_t)D0 * D1;
    if (work_amount == 0) return;
    size_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0 {0};
    T1 d1 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(ithr, nthr, d0, d1);
        utils::nd_iterator_step(d0, D0, d1, D1);
    }
}

template <typename T0, typename T1, typename T2, typename F>
void for_nd_ext(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2;
    if (work_amount == 0) return;
    size_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0 {0};
    T1 d1 {0};
    T2 d2 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(ithr, nthr, d0, d1, d2);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename F>
void for_nd_ext(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, const T3 &D3, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3;
    if (work_amount == 0) return;
    size_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0 {0};
    T1 d1 {0};
    T2 d2 {0};
    T3 d3 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(ithr, nthr, d0, d1, d2, d3);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
        typename F>
void for_nd_ext(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, const T3 &D3, const T4 &D4, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4;
    if (work_amount == 0) return;
    size_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0 {0};
    T1 d1 {0};
    T2 d2 {0};
    T3 d3 {0};
    T4 d4 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(ithr, nthr, d0, d1, d2, d3, d4);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
        typename T5, typename F>
void for_nd_ext(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, const T3 &D3, const T4 &D4, const T5 &D5, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4 * D5;
    if (work_amount == 0) return;
    size_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0 {0};
    T1 d1 {0};
    T2 d2 {0};
    T3 d3 {0};
    T4 d4 {0};
    T5 d5 {0};
    utils::nd_iterator_init(
            start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(ithr, nthr, d0, d1, d2, d3, d4, d5);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    }
}

/* parallel_nd_ext section */

template <typename T0, typename F>
void parallel_nd_ext(int nthr, const T0 &D0, F f) {
    const size_t work_amount = (size_t)D0;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr,
                [&](int ithr, int nthr) { for_nd_ext(ithr, nthr, D0, f); });
}

template <typename T0, typename T1, typename F>
void parallel_nd_ext(int nthr, const T0 &D0, const T1 &D1, F f) {
    const size_t work_amount = (size_t)D0 * D1;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr,
                [&](int ithr, int nthr) { for_nd_ext(ithr, nthr, D0, D1, f); });
}

template <typename T0, typename T1, typename T2, typename F>
void parallel_nd_ext(int nthr, const T0 &D0, const T1 &D1, const T2 &D2, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd_ext(ithr, nthr, D0, D1, D2, f);
        });
}

template <typename T0, typename T1, typename T2, typename T3, typename F>
void parallel_nd_ext(
        int nthr, const T0 &D0, const T1 &D1, const T2 &D2, const T3 &D3, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd_ext(ithr, nthr, D0, D1, D2, D3, f);
        });
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
        typename F>
void parallel_nd_ext(int nthr, const T0 &D0, const T1 &D1, const T2 &D2,
        const T3 &D3, const T4 &D4, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd_ext(ithr, nthr, D0, D1, D2, D3, D4, f);
        });
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
        typename T5, typename F>
void parallel_nd_ext(int nthr, const T0 &D0, const T1 &D1, const T2 &D2,
        const T3 &D3, const T4 &D4, const T5 &D5, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4 * D5;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd_ext(ithr, nthr, D0, D1, D2, D3, D4, D5, f);
        });
}

/* parallel_nd section */

template <typename T0, typename F>
void parallel_nd(const T0 &D0, F f) {
    const size_t work_amount = (size_t)D0;
    int nthr = adjust_num_threads(dnnl_get_max_threads(), work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) { for_nd(ithr, nthr, D0, f); });
}

template <typename T0, typename T1, typename F>
void parallel_nd(const T0 &D0, const T1 &D1, F f) {
    const size_t work_amount = (size_t)D0 * D1;
    int nthr = adjust_num_threads(dnnl_get_max_threads(), work_amount);
    if (nthr)
        parallel(nthr,
                [&](int ithr, int nthr) { for_nd(ithr, nthr, D0, D1, f); });
}

template <typename T0, typename T1, typename T2, typename F>
void parallel_nd(const T0 &D0, const T1 &D1, const T2 &D2, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2;
    int nthr = adjust_num_threads(dnnl_get_max_threads(), work_amount);
    if (nthr)
        parallel(nthr,
                [&](int ithr, int nthr) { for_nd(ithr, nthr, D0, D1, D2, f); });
}

template <typename T0, typename T1, typename T2, typename T3, typename F>
void parallel_nd(const T0 &D0, const T1 &D1, const T2 &D2, const T3 &D3, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3;
    int nthr = adjust_num_threads(dnnl_get_max_threads(), work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd(ithr, nthr, D0, D1, D2, D3, f);
        });
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
        typename F>
void parallel_nd(const T0 &D0, const T1 &D1, const T2 &D2, const T3 &D3,
        const T4 &D4, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4;
    int nthr = adjust_num_threads(dnnl_get_max_threads(), work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd(ithr, nthr, D0, D1, D2, D3, D4, f);
        });
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
        typename T5, typename F>
void parallel_nd(const T0 &D0, const T1 &D1, const T2 &D2, const T3 &D3,
        const T4 &D4, const T5 &D5, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4 * D5;
    int nthr = adjust_num_threads(dnnl_get_max_threads(), work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd(ithr, nthr, D0, D1, D2, D3, D4, D5, f);
        });
}

/* parallel_nd_in_omp section */

template <typename... Args>
void parallel_nd_in_omp(Args &&... args) {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
    for_nd(0, 1, utils::forward<Args>(args)...);
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
    for_nd(omp_get_thread_num(), omp_get_num_threads(),
            utils::forward<Args>(args)...);
#elif (DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB \
        || DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL)
    assert(!"parallel_nd_in_omp() is not supported by this DNNL_CPU_RUNTIME");
#endif
}

} // namespace impl
} // namespace dnnl

#endif
