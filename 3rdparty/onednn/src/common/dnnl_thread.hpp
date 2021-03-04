/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#ifndef COMMON_DNNL_THREAD_HPP
#define COMMON_DNNL_THREAD_HPP

#include <algorithm>

#include "utils.hpp"
#include "z_magic.hpp"

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
#define DNNL_THR_SYNC 1
inline int dnnl_get_max_threads() {
    return 1;
}
inline int dnnl_in_parallel() {
    return 0;
}
inline void dnnl_thr_barrier() {}

#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#include "omp.h"
#define DNNL_THR_SYNC 1
inline int dnnl_get_max_threads() {
    return omp_get_max_threads();
}
inline int dnnl_in_parallel() {
    return omp_in_parallel();
}
inline void dnnl_thr_barrier() {
#pragma omp barrier
}

#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"
#define DNNL_THR_SYNC 0
inline int dnnl_get_max_threads() {
    return tbb::this_task_arena::max_concurrency();
}
inline int dnnl_in_parallel() {
    return 0;
}
inline void dnnl_thr_barrier() {
    assert(!"no barrier in TBB");
}

#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include <thread>
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#define DNNL_THR_SYNC 0

namespace dnnl {
namespace impl {
namespace threadpool_utils {

// Each thread maintains a thread-local pointer to a threadpool which is
// 'active' for the current thread. If this pointer is a nullptr, all the work
// is executed sequentially.

// Sets `tp` to be the active threadpool for the calling thread. This will
// make all calls to `get_active_threadpool()` to return `tp` thus enabling
// `parallel()` and `parallel_nd()` to submit work to `tp`.
void activate_threadpool(dnnl::threadpool_interop::threadpool_iface *tp);

// Resets the active threadpool for the calling thread to nullptr. After this
// call `parallel()` and `parallel_nd()` would execute work sequentially.
void deactivate_threadpool();

// Returns the active threadpool for the calling thread.
dnnl::threadpool_interop::threadpool_iface *get_active_threadpool();

} // namespace threadpool_utils
} // namespace impl
} // namespace dnnl

inline int dnnl_get_max_threads() {
    using namespace dnnl::impl::threadpool_utils;
    dnnl::threadpool_interop::threadpool_iface *tp = get_active_threadpool();
    // This is the maximum number of threads oneDNN would use
    int def_max_threads = std::thread::hardware_concurrency();
    assert(def_max_threads > 0);
    // Use the default value if the threadpool-provided is outside the range
    // [1, def_max_threads]
    return tp ? std::min(std::max(1, tp->get_num_threads()), def_max_threads)
              : def_max_threads;
}
inline int dnnl_in_parallel() {
    using namespace dnnl::impl::threadpool_utils;
    dnnl::threadpool_interop::threadpool_iface *tp = get_active_threadpool();
    return tp ? tp->get_in_parallel() : 0;
}
inline void dnnl_thr_barrier() {
    assert(!"no barrier with THREADPOOL");
}
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#define PRAGMA_OMP(...) PRAGMA_MACRO(CHAIN2(omp, __VA_ARGS__))
#define OMP_GET_THREAD_NUM() omp_get_thread_num()
#define OMP_GET_NUM_THREADS() omp_get_num_threads()
#else
#define PRAGMA_OMP(...)
#define OMP_GET_THREAD_NUM() 0
#define OMP_GET_NUM_THREADS() 1
#endif

// MSVC still supports omp 2.0 only
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#define collapse(x)
#define PRAGMA_OMP_SIMD(...)
#else
#define PRAGMA_OMP_SIMD(...) PRAGMA_MACRO(CHAIN2(omp, simd __VA_ARGS__))
#endif // defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)

// process simdlen; it is supported for Clang >= 3.9; ICC >= 17.0; GCC >= 6.1
// No support on Windows.
#if (defined(__clang_major__) \
        && (__clang_major__ < 3 \
                || (__clang_major__ == 3 && __clang_minor__ < 9))) \
        || (defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1700) \
        || (!defined(__INTEL_COMPILER) && !defined(__clang__) \
                && (defined(_MSC_VER) || __GNUC__ < 6 \
                        || (__GNUC__ == 6 && __GNUC_MINOR__ < 1)))
#define simdlen(x)
#endif // long simdlen if

namespace dnnl {
namespace impl {

inline bool dnnl_thr_syncable() {
    return DNNL_THR_SYNC == 1;
}

template <typename T, typename U>
inline void balance211(T n, U team, U tid, T &n_start, T &n_end) {
    T n_min = 1;
    T &n_my = n_end;
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } else if (n_min == 1) {
        // team = T1 + T2
        // n = T1*n1 + T2*n2  (n1 - n2 = 1)
        T n1 = utils::div_up(n, (T)team);
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_my = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

template <typename T, typename U>
void balance2D(U nthr, U ithr, T ny, T &ny_start, T &ny_end, T nx, T &nx_start,
        T &nx_end, T nx_divider) {
    const T grp_count = nstl::min(nx_divider, static_cast<T>(nthr));
    const int grp_size_big = nthr / static_cast<int>(grp_count) + 1;
    const int grp_size_small = nthr / static_cast<int>(grp_count);
    const int n_grp_big = nthr % static_cast<int>(grp_count);
    const int threads_in_big_groups = n_grp_big * grp_size_big;

    const int ithr_bound_distance = ithr - threads_in_big_groups;
    T grp, grp_ithr, grp_nthr;
    if (ithr_bound_distance < 0) { // ithr in first groups
        grp = ithr / grp_size_big;
        grp_ithr = ithr % grp_size_big;
        grp_nthr = grp_size_big;
    } else { // ithr in last groups
        grp = n_grp_big + ithr_bound_distance / grp_size_small;
        grp_ithr = ithr_bound_distance % grp_size_small;
        grp_nthr = grp_size_small;
    }

    balance211(nx, grp_count, grp, nx_start, nx_end);
    balance211(ny, grp_nthr, grp_ithr, ny_start, ny_end);
}

} // namespace impl
} // namespace dnnl

#include "dnnl_thread_parallel_nd.hpp"

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
