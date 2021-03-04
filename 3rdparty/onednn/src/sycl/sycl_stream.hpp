/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef SYCL_STREAM_HPP
#define SYCL_STREAM_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "sycl/sycl_gpu_engine.hpp"
#include "sycl/sycl_memory_storage.hpp"
#include "sycl/sycl_stream_cpu_thunk.hpp"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "sycl/sycl_stream_submit_cpu_primitive.hpp"
#endif

#include <algorithm>
#include <map>
#include <memory>
#include <utility>
#include <CL/cl.h>
#include <CL/sycl.hpp>

namespace dnnl {
namespace impl {
namespace sycl {

struct sycl_stream_t : public gpu::compute::compute_stream_t {
    static status_t create_stream(
            stream_t **stream, engine_t *engine, unsigned flags) {
        std::unique_ptr<sycl_stream_t> sycl_stream(
                new sycl_stream_t(engine, flags));
        if (!sycl_stream) return status::out_of_memory;

        status_t status = sycl_stream->init();
        if (status != status::success) return status;
        *stream = sycl_stream.release();
        return status::success;
    }

    static status_t create_stream(
            stream_t **stream, engine_t *engine, cl::sycl::queue &queue) {
        unsigned flags;
        status_t status = sycl_stream_t::init_flags(&flags, queue);
        if (status != status::success) return status;

        std::unique_ptr<sycl_stream_t> sycl_stream(
                new sycl_stream_t(engine, flags, queue));

        status = sycl_stream->init();
        if (status != status::success) return status;

        *stream = sycl_stream.release();
        return status::success;
    }

    virtual status_t wait() override {
        queue_->wait_and_throw();
        return status::success;
    }

    cl::sycl::queue &queue() { return *queue_; }

    virtual status_t enqueue_primitive(const primitive_iface_t *prim_iface,
            exec_ctx_t &exec_ctx) override {
        auto execute_func = [&]() {
            status_t status = status::success;
            if (engine()->kind() == engine_kind::cpu) {

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
                auto event = queue_->submit([&](cl::sycl::handler &cgh) {
                    register_deps(cgh);
                    submit_cpu_primitive(this, prim_iface, exec_ctx, cgh);
                });
                set_deps({event});
#else
                assert(!"not expected");
                return status::runtime_error;
#endif
            } else if (engine()->kind() == engine_kind::gpu) {
                status = prim_iface->execute(exec_ctx);
            } else {
                assert(!"not expected");
            }
            return status;
        };
        status_t status = execute_func();
#ifndef DNNL_SYCL_DPCPP
        // Emulate in-order behavior
        if (flags() & stream_flags::in_order) wait();
#endif
        return status;
    }

    virtual status_t copy(const memory_storage_t &src,
            const memory_storage_t &dst, size_t size) override {
        if (size == 0) return status::success;
        // TODO: add src and dst sizes check

        auto *sycl_src
                = utils::downcast<const sycl_memory_storage_base_t *>(&src);
        auto *sycl_dst
                = utils::downcast<const sycl_memory_storage_base_t *>(&dst);
        bool usm_src = sycl_src->memory_kind() == memory_kind::usm;
        bool usm_dst = sycl_dst->memory_kind() == memory_kind::usm;
        cl::sycl::event e;

#ifdef DNNL_SYCL_DPCPP
        if (usm_src && usm_dst) {
            auto *usm_src
                    = utils::downcast<const sycl_usm_memory_storage_t *>(&src);
            auto *usm_dst
                    = utils::downcast<const sycl_usm_memory_storage_t *>(&dst);
            e = queue_->submit([&](cl::sycl::handler &cgh) {
                register_deps(cgh);
                cgh.memcpy(usm_dst->usm_ptr(), usm_src->usm_ptr(), size);
            });
        } else if (usm_src && !usm_dst) {
            auto *usm_src
                    = utils::downcast<const sycl_usm_memory_storage_t *>(&src);
            auto *buffer_dst
                    = utils::downcast<const sycl_buffer_memory_storage_t *>(
                            &dst);
            auto &b_dst = buffer_dst->buffer();
            e = queue_->submit([&](cl::sycl::handler &cgh) {
                register_deps(cgh);
                auto acc_dst
                        = b_dst.get_access<cl::sycl::access::mode::write>(cgh);
                cgh.copy(usm_src->usm_ptr(), acc_dst);
            });
        } else if (!usm_src && usm_dst) {
            auto *buffer_src
                    = utils::downcast<const sycl_buffer_memory_storage_t *>(
                            &src);
            auto &b_src = buffer_src->buffer();
            auto *usm_dst
                    = utils::downcast<const sycl_usm_memory_storage_t *>(&dst);
            e = queue_->submit([&](cl::sycl::handler &cgh) {
                register_deps(cgh);
                auto acc_src
                        = b_src.get_access<cl::sycl::access::mode::read>(cgh);
                cgh.copy(acc_src, usm_dst->usm_ptr());
            });
        } else // if (!usm_src && !usm_dst)
#endif
        {
            assert(!usm_src && !usm_dst && "USM is not supported yet");
            auto *buffer_src
                    = utils::downcast<const sycl_buffer_memory_storage_t *>(
                            &src);
            auto *buffer_dst
                    = utils::downcast<const sycl_buffer_memory_storage_t *>(
                            &dst);
            auto &b_src = buffer_src->buffer();
            auto &b_dst = buffer_dst->buffer();
            e = queue_->submit([&](cl::sycl::handler &cgh) {
                auto acc_src
                        = b_src.get_access<cl::sycl::access::mode::read>(cgh);
                auto acc_dst
                        = b_dst.get_access<cl::sycl::access::mode::write>(cgh);
                register_deps(cgh);
                cgh.copy(acc_src, acc_dst);
            });
        }
        set_deps({e});

        return status::success;
    }

    virtual status_t fill(const memory_storage_t &dst, uint8_t pattern,
            size_t size) override {
        auto *sycl_dst
                = utils::downcast<const sycl_memory_storage_base_t *>(&dst);
        bool usm = sycl_dst->memory_kind() == memory_kind::usm;

        cl::sycl::event out_event;
        std::vector<cl::sycl::event> in_deps = get_deps();

#ifdef DNNL_SYCL_DPCPP
        if (usm) {
            auto *usm_dst
                    = utils::downcast<const sycl_usm_memory_storage_t *>(&dst);
            auto dst_ptr = static_cast<uint8_t *>(usm_dst->usm_ptr());
            // Note: we cannot use queue_.fill since it cannot handle
            // events as input
            out_event = queue_->submit([&](cl::sycl::handler &cgh) {
                register_deps(cgh, in_deps);
                cgh.memset(dst_ptr, pattern, size);
            });
        } else
#endif
        {
            auto *buffer_dst
                    = utils::downcast<const sycl_buffer_memory_storage_t *>(
                            &dst);
            out_event = queue_->submit([&](cl::sycl::handler &cgh) {
                // need a u8 accessor to get the proper range
                cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                        cl::sycl::access::target::global_buffer>
                        acc_dst(buffer_dst->buffer(), cgh,
                                cl::sycl::range<1>(size), cl::sycl::id<1>(0));
                register_deps(cgh, in_deps);
                cgh.fill(acc_dst, pattern);
            });
        }
        set_deps({out_event});
        return status::success;
    }

#ifdef DNNL_SYCL_DPCPP
    const std::vector<cl::sycl::event> &get_deps() const { return deps_; }
    void set_deps(const std::vector<cl::sycl::event> &deps) { deps_ = deps; }
    void add_dep(const cl::sycl::event &dep) { deps_.push_back(dep); }
    cl::sycl::event get_output_event() const {
        // Fast path: if only one event, return it.
        if (deps_.size() == 1) return deps_[0];

        // Otherwise, we run a trivial kernel to gather all deps. The
        // dummy task is needed to not get an error related to empty
        // kernel.
        auto e = queue_->submit([&](cl::sycl::handler &cgh) {
            register_deps(cgh);
            cgh.single_task<class dnnl_dummy_kernel>([]() {});
        });
        return e;
    }
    void register_deps(cl::sycl::handler &cgh,
            const std::vector<cl::sycl::event> &event_list) const {
        cgh.depends_on(event_list);
    }
    void register_deps(cl::sycl::handler &cgh) const {
        register_deps(cgh, get_deps());
    }
#else
    // Here, we use only buffers, so dependency tracking is handled
    // via buffers and events tracking is useless. However, if we
    // really wanted to, we could gather events by creating a dummy
    // sycl buffer from dummy cl_mem and deps_ events, enqueue a dummy
    // kernel, and return the output event
    const std::vector<cl::sycl::event> &get_deps() const {
        static std::vector<cl::sycl::event> empty_list = {};
        return empty_list;
    }
    void set_deps(const std::vector<cl::sycl::event> &) {}
    void add_dep(const cl::sycl::event &) {}
    cl::sycl::event get_output_event() const { return cl::sycl::event(); }
    void register_deps(cl::sycl::handler &cgh,
            const std::vector<cl::sycl::event> &event_list) const {}
    void register_deps(cl::sycl::handler &cgh) const {}
#endif

private:
    sycl_stream_t(engine_t *engine, unsigned flags)
        : gpu::compute::compute_stream_t(engine, flags) {}
    sycl_stream_t(engine_t *engine, unsigned flags, cl::sycl::queue &queue)
        : gpu::compute::compute_stream_t(engine, flags)
        , queue_(new cl::sycl::queue(queue)) {}

    status_t init();

    static status_t init_flags(unsigned *flags, cl::sycl::queue &queue) {
        // SYCL queue is always out-of-order
        *flags = stream_flags::out_of_order;
        return status::success;
    }

private:
    std::unique_ptr<cl::sycl::queue> queue_;
    // XXX: This is a temporary solution, ideally events should be a part of
    // execution context.
    std::vector<cl::sycl::event> deps_;
}; // namespace sycl

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
