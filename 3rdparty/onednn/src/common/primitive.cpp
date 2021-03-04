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

#include <assert.h>

#include "c_types_map.hpp"
#include "engine.hpp"
#include "primitive.hpp"
#include "primitive_desc.hpp"
#include "primitive_exec_types.hpp"
#include "reorder_pd.hpp"
#include "scratchpad_debug.hpp"
#include "stream.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;
using namespace dnnl::impl::primitive_kind;

namespace {
// XXX: this is a huge hammer. This disables all and any msan checks on
// primitives outputs.
//
// A proper approach would be an implementation-specific unpoisoning.
void unpoison_outputs(const exec_args_t &args) {
    for (const auto &arg : args) {
        if (arg.second.is_const) continue;
        auto *mem = arg.second.mem;
        void *p;
        mem->get_data_handle(&p);
        size_t s = memory_desc_wrapper(*mem->md()).size();
        msan_unpoison(p, s);
    }
}
} // namespace

namespace dnnl {
namespace impl {

nested_scratchpad_t::nested_scratchpad_t(const exec_ctx_t &master_ctx, int key,
        const std::shared_ptr<primitive_t> &nested_p) {
    auto scratchpad = master_ctx.get_scratchpad_grantor();
    scratchpad_mem_storage_ = scratchpad.get_memory_storage(key);
    grantor_ = utils::make_unique<memory_tracking::grantor_t>(
            nested_p->pd()->scratchpad_registry().grantor(
                    scratchpad_mem_storage_.get(), master_ctx));
#ifdef DNNL_ENABLE_MEM_DEBUG
    if (scratchpad_debug::is_protect_scratchpad()) {
        scratchpad_debug::protect_scratchpad_buffer(
                grantor_->get_base_storage(), grantor_->get_registry());
    }
#endif
}

#ifdef DNNL_ENABLE_MEM_DEBUG
nested_scratchpad_t::~nested_scratchpad_t() {
    if (scratchpad_debug::is_protect_scratchpad()) {
        scratchpad_debug::unprotect_scratchpad_buffer(
                grantor_->get_base_storage(), grantor_->get_registry());
    }
}
#else
nested_scratchpad_t::~nested_scratchpad_t() = default;
#endif

} // namespace impl
} // namespace dnnl

// API
status_t dnnl_primitive_desc_destroy(
        primitive_desc_iface_t *primitive_desc_iface) {
    delete primitive_desc_iface;
    return success;
}

status_t dnnl_primitive_create(primitive_iface_t **primitive_iface,
        const primitive_desc_iface_t *primitive_desc_iface) {
    if (utils::any_null(primitive_iface, primitive_desc_iface))
        return invalid_arguments;
    return primitive_desc_iface->create_primitive_iface(primitive_iface);
}

namespace dnnl {
namespace impl {
status_t primitive_execute(
        const primitive_iface_t *primitive_iface, exec_ctx_t &ctx) {
    auto stream = ctx.stream();
    status_t status = success;

    stream->before_exec_hook();

    if (get_verbose()) {
        double ms = get_msec();
        status = stream->enqueue_primitive(primitive_iface, ctx);
        stream->wait();
        ms = get_msec() - ms;
        printf("dnnl_verbose,exec,%s,%g\n", primitive_iface->pd()->info(), ms);
        fflush(stdout);
    } else {
        status = stream->enqueue_primitive(primitive_iface, ctx);
    }

    stream->after_exec_hook();

    if (msan_enabled) unpoison_outputs(ctx.args());

    return status;
}

} // namespace impl
} // namespace dnnl

status_t dnnl_primitive_execute(const primitive_iface_t *primitive_iface,
        stream_t *stream, int nargs, const dnnl_exec_arg_t *c_args) {
    bool ok = true && !utils::any_null(primitive_iface, stream)
            && primitive_iface->engine() == stream->engine()
            && IMPLICATION(nargs > 0, c_args != nullptr);
    if (!ok) return invalid_arguments;

    exec_args_t args;
    status_t status = cvt_primitive_args(
            primitive_iface->pd()->impl().get(), nargs, c_args, args);
    if (status != status::success) return status;

    exec_ctx_t ctx(stream, std::move(args));
    status = dnnl::impl::primitive_execute(primitive_iface, ctx);

    return status;
}

status_t dnnl_primitive_get_primitive_desc(
        const primitive_iface_t *primitive_iface,
        const primitive_desc_iface_t **primitive_desc_iface) {
    if (utils::any_null(primitive_iface, primitive_desc_iface))
        return invalid_arguments;
    return safe_ptr_assign(*primitive_desc_iface, primitive_iface->pd());
}

status_t dnnl_primitive_destroy(primitive_iface_t *primitive_iface) {
    if (primitive_iface != nullptr) primitive_iface->release();
    return success;
}

// primitive_iface_t implementation
dnnl_primitive::dnnl_primitive(
        const std::shared_ptr<primitive_t> &primitive, engine_t *engine)
    : counter_(1)
    , primitive_(primitive)
    , pd_(utils::make_unique<primitive_desc_iface_t>(
              primitive_->pd(), engine)) {}

// reorder specialization
dnnl_primitive::dnnl_primitive(const std::shared_ptr<primitive_t> &primitive,
        engine_t *engine, engine_t *src_engine, engine_t *dst_engine)
    : counter_(1)
    , primitive_(primitive)
    , pd_(utils::make_unique<reorder_primitive_desc_iface_t>(
              primitive_->pd(), engine, src_engine, dst_engine)) {}

dnnl_primitive::~dnnl_primitive() {
    if (scratchpad_debug::is_protect_scratchpad() && scratchpad_ != nullptr
            && scratchpad_->get_memory_storage() != nullptr) {
        const memory_tracking::registry_t &registry
                = primitive_->pd()->scratchpad_registry();
        scratchpad_debug::unprotect_scratchpad_buffer(
                scratchpad_->get_memory_storage(), registry);
    }
}

status_t dnnl_primitive::init() {
    const size_t scratchpad_size
            = primitive_->pd()->scratchpad_size(scratchpad_mode::library);

    if (scratchpad_size) {
        const memory_tracking::registry_t &registry
                = primitive_->pd()->scratchpad_registry();
        bool use_global_scratchpad = scratchpad_debug::is_protect_scratchpad()
                ? false
                : primitive_->use_global_scratchpad();
        auto *scratchpad_ptr = create_scratchpad(
                pd_->engine(), scratchpad_size, use_global_scratchpad);
        if (scratchpad_ptr == nullptr) return out_of_memory;
        if (scratchpad_ptr->get_memory_storage() == nullptr) {
            delete scratchpad_ptr;
            return out_of_memory;
        }

        if (scratchpad_debug::is_protect_scratchpad()) {
            scratchpad_debug::protect_scratchpad_buffer(
                    scratchpad_ptr->get_memory_storage(), registry);
        }
        scratchpad_.reset(scratchpad_ptr);
        if (scratchpad_ptr->size() < scratchpad_size) return out_of_memory;
    }
    return primitive_->create_resource(pd()->engine(), resource_mapper_);
}

engine_t *dnnl_primitive::engine() const {
    return pd_->engine();
}

const primitive_desc_iface_t *dnnl_primitive::pd() const {
    return pd_.get();
}

status_t dnnl_primitive::execute(exec_ctx_t &ctx) const {
    const memory_storage_t *mem_storage = nullptr;
    if (primitive_->pd()->attr()->scratchpad_mode_ == scratchpad_mode::user) {
        memory_t *scratchpad_memory = ctx.output(DNNL_ARG_SCRATCHPAD);
        mem_storage = scratchpad_memory ? scratchpad_memory->memory_storage()
                                        : nullptr;
    } else if (scratchpad_) {
        mem_storage = scratchpad_->get_memory_storage();
    }

    auto scratchpad_grantor
            = primitive_->pd()->scratchpad_registry().grantor(mem_storage, ctx);
    ctx.set_scratchpad_grantor(&scratchpad_grantor);
    ctx.set_resource_mapper(&resource_mapper_);

    auto status = primitive_->execute(ctx);
    ctx.set_scratchpad_grantor(nullptr);
    return status;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
