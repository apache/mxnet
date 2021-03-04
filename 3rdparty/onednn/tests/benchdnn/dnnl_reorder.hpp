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

#ifndef DNNL_REORDER_HPP
#define DNNL_REORDER_HPP

#include <memory>

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "dnn_types.hpp"

int execute_reorder(const dnn_mem_t &src, dnn_mem_t &dst,
        const_dnnl_primitive_attr_t attr) {
    std::shared_ptr<const dnn_mem_t> r_src(&src, [](const dnn_mem_t *) {});
    std::shared_ptr<dnn_mem_t> r_dst(&dst, [](dnn_mem_t *) {});

    dnnl_primitive_desc_t r_pd = nullptr;
    dnnl_primitive_t r {};

    // Optimization to reduce testing time for GPU.
    //
    // For CPU <-> GPU reorders, the library creates GPU-side kernels.
    // Benchdnn heavily relies on reorders and this greatly increases execution
    // time because of big overhead on building OpenCL kernels.
    //
    // First, try to create CPU reorder for the requested GPU reorder. If
    // succeeded, then create CPU memory object wrapping mapped pointers of
    // source and destination and execute CPU reorder. If CPU reorder can't be
    // create, then just execute a regular GPU reorder.
    //
    // This optimization is skipped when testing reorder, sum and concat
    // primitives because they are used specifically to test GPU reorders.
#if (DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL) \
        || (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL)
    std::string driver = std::string(driver_name);
    bool is_reorder_related_driver = (driver == std::string("reorder")
            || driver == std::string("sum") || driver == std::string("concat"));
    const auto &cpu_engine = get_cpu_engine();
    if (!is_reorder_related_driver
            && (src.engine_kind() == dnnl_gpu
                    || dst.engine_kind() == dnnl_gpu)) {

        dnnl_status_t status = dnnl_reorder_primitive_desc_create(
                &r_pd, &src.md_, cpu_engine, &dst.md_, cpu_engine, attr);
        if (status == dnnl_success) {
            // Create CPU memory objects wrapping mapped pointers of source and
            // destination
            r_src.reset(new dnn_mem_t(dnn_mem_t::create_from_host_ptr(
                    src.md_, cpu_engine, (void *)src)));
            r_dst.reset(new dnn_mem_t(dnn_mem_t::create_from_host_ptr(
                    dst.md_, cpu_engine, (void *)dst)));
        }
    }
#endif

    if (!r_pd) {
        DNN_SAFE(dnnl_reorder_primitive_desc_create(&r_pd, &src.md_,
                         src.engine(), &dst.md_, dst.engine(), attr),
                CRIT);
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                r_pd, dnnl_query_exec_arg_md, index);
    };
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);
    dnn_mem_t scratchpad(scratchpad_md, src.engine());

    DNN_SAFE(dnnl_primitive_create(&r, r_pd), CRIT);
    dnnl_status_t pd_destroy_status = dnnl_primitive_desc_destroy(r_pd);
    if (pd_destroy_status != dnnl_success) {
        dnnl_primitive_destroy(r);
        DNN_SAFE(pd_destroy_status, CRIT);
    }

    args_t args;
    args.set(DNNL_ARG_FROM, *r_src);
    args.set(DNNL_ARG_TO, *r_dst);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad);

    SAFE(execute_and_wait(r, args), CRIT);
    DNN_SAFE(dnnl_primitive_destroy(r), CRIT);

    return OK;
}

#endif
