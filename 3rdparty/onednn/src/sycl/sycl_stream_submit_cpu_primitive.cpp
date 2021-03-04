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

#include "oneapi/dnnl/dnnl.hpp"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL

#include "sycl/sycl_stream_submit_cpu_primitive.hpp"

#include "common/dnnl_traits.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_c_types_map.hpp"
#include "sycl/sycl_memory_storage.hpp"

#include <assert.h>
#include <exception>
#include <tuple>
#include <vector>
#include <CL/sycl.hpp>

// A global scope tag type to use for enqueueing a single task
template <typename... types>
class dnnl_submit_primitive_tag_t;

namespace dnnl {
namespace impl {
namespace sycl {

namespace {

template <size_t N>
void init_thunk_params(thunk_params_t *p) {
    p->size = N;
}

template <size_t N, typename accessor_t, typename... accessor_types>
void init_thunk_params(
        thunk_params_t *p, accessor_t acc, accessor_types... accessors) {
    p->native_pointers[N - sizeof...(accessor_types) - 1]
            = reinterpret_cast<uintptr_t>(&acc[0]);
    init_thunk_params<N>(p, accessors...);
}

template <typename... accessor_types>
using make_kernel_tag
        = dnnl_submit_primitive_tag_t<typename accessor_types::value_type...>;

template <typename... param_types>
status_t submit_cpu_primitive_with_params_impl(submit_ctx_t *submit_ctx,
        cl::sycl::handler &cgh, param_types... params) {
    // Trick the compiler by capturing scalar values in the kernel
    // instead of pointers what is not allowed.
    uintptr_t submit_ctx_ptr = reinterpret_cast<uintptr_t>(submit_ctx);
    using tag_type = make_kernel_tag<param_types...>;

    host_task<tag_type>(cgh, [=]() {
        thunk_params_t thunk_params;
        thunk_params.submit_ctx_ptr = submit_ctx_ptr;

        constexpr size_t nparams = sizeof...(param_types);

        // Extract pointers from params
        init_thunk_params<nparams>(&thunk_params, params...);

        // Call C-linkage thunk which executes CPU primitive natively
        dnnl_impl_sycl_cpu_thunk(&thunk_params);
    });
    return status::success;
}

template <typename params_tuple_t, size_t... Is>
void submit_cpu_primitive_with_params_impl(submit_ctx_t *submit_ctx,
        cl::sycl::handler &cgh, params_tuple_t &params_tp,
        nstl::index_sequence<Is...>) {
    submit_cpu_primitive_with_params_impl(submit_ctx, cgh,
            std::get<Is>(params_tp)
                    .template get_access<cl::sycl::access::mode::read_write>(
                            cgh)...);
}

template <typename... storage_types>
void fast_dispatch_by_size(submit_ctx_t *submit_ctx, cl::sycl::handler &cgh,
        const storage_types *... storages) {
    constexpr size_t nparams = sizeof...(storage_types);

    auto params_tp = std::make_tuple(
            utils::downcast<const sycl_buffer_memory_storage_t *>(storages)
                    ->buffer()...);
    submit_cpu_primitive_with_params_impl(
            submit_ctx, cgh, params_tp, nstl::make_index_sequence<nparams> {});
}

} // namespace

// CPU primitive submission is implemented this way:
// 1. Obtain all accessible SYCL memory storages from iterating
//    over the execution context.
// 2. Use variadic templates to pass SYCL accessors for these
//    storages to the SYCL kernel inside single_task().
// 3. Stream, primitive and execution context pointers are
//    passed to the kernel via the submit context structure.
// 4. Pass a submit context via uintptr_t to work around
//    SYCL kernel restrictions. The context structure is
//    unpacked and deallocated on kernel side.
// 5. The SYCL kernel "registers" mapping
//    memory storage -> raw pointer via execution context.
// 6. Call the thunk function that executes the primitve
//    natively.
void submit_cpu_primitive(stream_t *stream, const primitive_iface_t *prim_iface,
        const exec_ctx_t &exec_ctx, cl::sycl::handler &cgh) {
    const_cast<primitive_iface_t *>(prim_iface)->retain();

    std::vector<const memory_storage_t *> sycl_mem_storages;
    for (auto &a : exec_ctx.args()) {
        if (a.second.mem->engine()->runtime_kind() == runtime_kind::sycl) {
            auto *mem_storage = a.second.mem->memory_storage();
            if (!mem_storage->is_null()) {
#ifdef DNNL_SYCL_DPCPP
                // Skip USM memory storages as they do not require special
                // handling and can be accessed directly
                auto mem_api_kind
                        = utils::downcast<const sycl_memory_storage_base_t *>(
                                mem_storage)
                                  ->memory_kind();
                if (mem_api_kind == memory_kind::usm) continue;
#endif
                sycl_mem_storages.push_back(mem_storage);
            }
        }
    }

    // Keep unique only
    std::sort(sycl_mem_storages.begin(), sycl_mem_storages.end());
    auto last = std::unique(sycl_mem_storages.begin(), sycl_mem_storages.end());
    sycl_mem_storages.erase(last, sycl_mem_storages.end());

    auto *submit_ctx = new submit_ctx_t(exec_ctx);
    submit_ctx->stream = stream;
    submit_ctx->prim_iface = prim_iface;
    submit_ctx->sycl_mem_storages = sycl_mem_storages;

    switch (sycl_mem_storages.size()) {
        case 0: fast_dispatch_by_size(submit_ctx, cgh); break;
        case 1:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0]);
            break;
        case 2:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1]);
            break;
        case 3:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2]);
            break;
        case 4:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3]);
            break;
        case 5:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4]);
            break;
        case 6:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5]);
            break;
        case 7:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6]);
            break;
        case 8:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6],
                    sycl_mem_storages[7]);
            break;
        case 9:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6],
                    sycl_mem_storages[7], sycl_mem_storages[8]);
            break;
        case 10:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6],
                    sycl_mem_storages[7], sycl_mem_storages[8],
                    sycl_mem_storages[9]);
            break;
        case 11:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6],
                    sycl_mem_storages[7], sycl_mem_storages[8],
                    sycl_mem_storages[9], sycl_mem_storages[10]);
            break;
        case 12:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6],
                    sycl_mem_storages[7], sycl_mem_storages[8],
                    sycl_mem_storages[9], sycl_mem_storages[10],
                    sycl_mem_storages[11]);
            break;
        case 13:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6],
                    sycl_mem_storages[7], sycl_mem_storages[8],
                    sycl_mem_storages[9], sycl_mem_storages[10],
                    sycl_mem_storages[11], sycl_mem_storages[12]);
            break;
        case 14:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6],
                    sycl_mem_storages[7], sycl_mem_storages[8],
                    sycl_mem_storages[9], sycl_mem_storages[10],
                    sycl_mem_storages[11], sycl_mem_storages[12],
                    sycl_mem_storages[13]);
            break;
        case 15:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6],
                    sycl_mem_storages[7], sycl_mem_storages[8],
                    sycl_mem_storages[9], sycl_mem_storages[10],
                    sycl_mem_storages[11], sycl_mem_storages[12],
                    sycl_mem_storages[13], sycl_mem_storages[14]);
            break;
        case 16:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6],
                    sycl_mem_storages[7], sycl_mem_storages[8],
                    sycl_mem_storages[9], sycl_mem_storages[10],
                    sycl_mem_storages[11], sycl_mem_storages[12],
                    sycl_mem_storages[13], sycl_mem_storages[14],
                    sycl_mem_storages[15]);
            break;
        case 17:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6],
                    sycl_mem_storages[7], sycl_mem_storages[8],
                    sycl_mem_storages[9], sycl_mem_storages[10],
                    sycl_mem_storages[11], sycl_mem_storages[12],
                    sycl_mem_storages[13], sycl_mem_storages[14],
                    sycl_mem_storages[15], sycl_mem_storages[16]);
            break;
        case 18:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6],
                    sycl_mem_storages[7], sycl_mem_storages[8],
                    sycl_mem_storages[9], sycl_mem_storages[10],
                    sycl_mem_storages[11], sycl_mem_storages[12],
                    sycl_mem_storages[13], sycl_mem_storages[14],
                    sycl_mem_storages[15], sycl_mem_storages[16],
                    sycl_mem_storages[17]);
            break;
        case 19:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6],
                    sycl_mem_storages[7], sycl_mem_storages[8],
                    sycl_mem_storages[9], sycl_mem_storages[10],
                    sycl_mem_storages[11], sycl_mem_storages[12],
                    sycl_mem_storages[13], sycl_mem_storages[14],
                    sycl_mem_storages[15], sycl_mem_storages[16],
                    sycl_mem_storages[17], sycl_mem_storages[18]);
            break;
        case 20:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6],
                    sycl_mem_storages[7], sycl_mem_storages[8],
                    sycl_mem_storages[9], sycl_mem_storages[10],
                    sycl_mem_storages[11], sycl_mem_storages[12],
                    sycl_mem_storages[13], sycl_mem_storages[14],
                    sycl_mem_storages[15], sycl_mem_storages[16],
                    sycl_mem_storages[17], sycl_mem_storages[18],
                    sycl_mem_storages[19]);
            break;
        case 21:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6],
                    sycl_mem_storages[7], sycl_mem_storages[8],
                    sycl_mem_storages[9], sycl_mem_storages[10],
                    sycl_mem_storages[11], sycl_mem_storages[12],
                    sycl_mem_storages[13], sycl_mem_storages[14],
                    sycl_mem_storages[15], sycl_mem_storages[16],
                    sycl_mem_storages[17], sycl_mem_storages[18],
                    sycl_mem_storages[19], sycl_mem_storages[20]);
            break;
        case 22:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6],
                    sycl_mem_storages[7], sycl_mem_storages[8],
                    sycl_mem_storages[9], sycl_mem_storages[10],
                    sycl_mem_storages[11], sycl_mem_storages[12],
                    sycl_mem_storages[13], sycl_mem_storages[14],
                    sycl_mem_storages[15], sycl_mem_storages[16],
                    sycl_mem_storages[17], sycl_mem_storages[18],
                    sycl_mem_storages[19], sycl_mem_storages[20],
                    sycl_mem_storages[21]);
            break;
        case 23:
            fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                    sycl_mem_storages[1], sycl_mem_storages[2],
                    sycl_mem_storages[3], sycl_mem_storages[4],
                    sycl_mem_storages[5], sycl_mem_storages[6],
                    sycl_mem_storages[7], sycl_mem_storages[8],
                    sycl_mem_storages[9], sycl_mem_storages[10],
                    sycl_mem_storages[11], sycl_mem_storages[12],
                    sycl_mem_storages[13], sycl_mem_storages[14],
                    sycl_mem_storages[15], sycl_mem_storages[16],
                    sycl_mem_storages[17], sycl_mem_storages[18],
                    sycl_mem_storages[19], sycl_mem_storages[20],
                    sycl_mem_storages[21], sycl_mem_storages[22]);
            break;
        default:
            delete submit_ctx;
            assert(!"Please add another case");
            throw std::runtime_error("Internal error");
    }
}

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
