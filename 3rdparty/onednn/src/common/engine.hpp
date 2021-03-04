/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef COMMON_ENGINE_HPP
#define COMMON_ENGINE_HPP

#include "oneapi/dnnl/dnnl.h"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#endif

#include "c_types_map.hpp"
#include "memory.hpp"
#include "memory_storage.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

/** \brief An abstraction of an execution unit with shared resources
 *
 * Responsibilities:
 *   - Provide engine specific memory allocation
 *   - Provide engine specific primitive_desc_t creators
 */
struct dnnl_engine : public dnnl::impl::c_compatible {
    dnnl_engine(dnnl::impl::engine_kind_t kind,
            dnnl::impl::runtime_kind_t runtime_kind)
        : kind_(kind), runtime_kind_(runtime_kind) {}
    virtual ~dnnl_engine() = default;

    /** get kind of the current engine */
    dnnl::impl::engine_kind_t kind() const { return kind_; }

    /** get the runtime kind of the current engine */
    dnnl::impl::runtime_kind_t runtime_kind() const { return runtime_kind_; }

    virtual dnnl::impl::device_id_t device_id() const = 0;

    /** create memory storage */
    virtual dnnl::impl::status_t create_memory_storage(
            dnnl::impl::memory_storage_t **storage, unsigned flags, size_t size,
            void *handle)
            = 0;
    dnnl::impl::status_t create_memory_storage(
            dnnl::impl::memory_storage_t **storage, size_t size) {
        return create_memory_storage(
                storage, dnnl::impl::memory_flags_t::alloc, size, nullptr);
    }

    /** create stream */
    virtual dnnl::impl::status_t create_stream(
            dnnl::impl::stream_t **stream, unsigned flags)
            = 0;

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    virtual dnnl::impl::status_t create_stream(dnnl::impl::stream_t **stream,
            dnnl::threadpool_interop::threadpool_iface *threadpool) {
        return dnnl::impl::status::invalid_arguments;
    }
#endif

    virtual dnnl::impl::status_t get_service_stream(
            dnnl::impl::stream_t *&stream) {
        stream = nullptr;
        return dnnl::impl::status::success;
    }
    /** implementation section (typedefs) */

    // TODO: remove engine?
    typedef dnnl::impl::status_t (*reorder_primitive_desc_create_f)(
            dnnl::impl::reorder_pd_t **, dnnl::impl::engine_t *engine,
            const dnnl::impl::primitive_attr_t *attr,
            dnnl::impl::engine_t *src_engine,
            const dnnl::impl::memory_desc_t *src_md,
            dnnl::impl::engine_t *dst_engine,
            const dnnl::impl::memory_desc_t *dst_md);

    typedef dnnl::impl::status_t (*concat_primitive_desc_create_f)(
            dnnl::impl::concat_pd_t **, dnnl::impl::engine_t *engine,
            const dnnl::impl::primitive_attr_t *attr,
            const dnnl::impl::memory_desc_t *dst_md, int n, int concat_dim,
            const dnnl::impl::memory_desc_t *src_mds);

    typedef dnnl::impl::status_t (*sum_primitive_desc_create_f)(
            dnnl::impl::sum_pd_t **, dnnl::impl::engine_t *engine,
            const dnnl::impl::primitive_attr_t *attr,
            const dnnl::impl::memory_desc_t *dst_md, int n, const float *scales,
            const dnnl::impl::memory_desc_t *src_mds);

    typedef dnnl::impl::status_t (*primitive_desc_create_f)(
            dnnl::impl::primitive_desc_t **, const dnnl::impl::op_desc_t *,
            const dnnl::impl::primitive_attr_t *attr, dnnl::impl::engine_t *,
            const dnnl::impl::primitive_desc_t *);

    /* implementation section */

    /** return the list of reorder implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const reorder_primitive_desc_create_f *
    get_reorder_implementation_list(const dnnl::impl::memory_desc_t *src_md,
            const dnnl::impl::memory_desc_t *dst_md) const = 0;

    /** return the list of concat implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const concat_primitive_desc_create_f *
    get_concat_implementation_list() const = 0;

    /** return the list of sum implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const sum_primitive_desc_create_f *
    get_sum_implementation_list() const = 0;

    /** return the list of implementations for a given descriptor.
     * engine guarantees to return a NULL-terminated list */
    virtual const primitive_desc_create_f *get_implementation_list(
            const dnnl::impl::op_desc_t *desc) const = 0;

protected:
    dnnl::impl::engine_kind_t kind_;
    dnnl::impl::runtime_kind_t runtime_kind_;
};

namespace dnnl {
namespace impl {

inline runtime_kind_t get_default_runtime(engine_kind_t kind) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (kind == engine_kind::gpu) return runtime_kind::ocl;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    if (kind == engine_kind::gpu) return runtime_kind::sycl;
#endif
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SEQ
    return runtime_kind::seq;
#elif DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
    return runtime_kind::omp;
#elif DNNL_CPU_RUNTIME == DNNL_RUNTIME_TBB
    return runtime_kind::tbb;
#elif DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    return runtime_kind::threadpool;
#elif DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    return runtime_kind::sycl;
#else
    return runtime_kind::none;
#endif
}

inline runtime_kind_t get_cpu_native_runtime() {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
    return runtime_kind::seq;
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
    return runtime_kind::omp;
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
    return runtime_kind::tbb;
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    return runtime_kind::threadpool;
#else
    return runtime_kind::none;
#endif
}

inline bool is_native_runtime(runtime_kind_t kind) {
    return utils::one_of(kind, runtime_kind::seq, runtime_kind::omp,
            runtime_kind::tbb, runtime_kind::threadpool);
}

struct engine_factory_t : public c_compatible {
    virtual size_t count() const = 0;
    virtual status_t engine_create(engine_t **engine, size_t index) const = 0;
    virtual ~engine_factory_t() = default;
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
