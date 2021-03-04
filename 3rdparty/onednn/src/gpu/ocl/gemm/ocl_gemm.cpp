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

#include <memory>
#include <CL/cl.h>

#include "oneapi/dnnl/dnnl.h"

#include "common/dnnl_traits.hpp"
#include "common/gemm_utils.hpp"
#include "common/memory_storage.hpp"
#include "common/nstl.hpp"
#include "common/primitive_desc.hpp"
#include "gpu/ocl/gemm/gen9_gemm.hpp"
#include "gpu/ocl/gemm/gen9_gemm_x8x8s32.hpp"
#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"

using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::gpu::ocl;

namespace {

dnnl_status_t gemm_generic(cl_command_queue queue, const char *transa,
        const char *transb, dim_t m, dim_t n, dim_t k, cl_float alpha, cl_mem a,
        dim_t offset_a, dim_t lda, cl_mem b, dim_t offset_b, dim_t ldb,
        cl_float beta, cl_mem c, dim_t offset_c, dim_t ldc, data_type_t a_type,
        data_type_t b_type = data_type::undef,
        data_type_t c_type = data_type::undef,
        data_type_t acc_type = data_type::undef) {

    if (b_type == data_type::undef) b_type = a_type;
    if (c_type == data_type::undef) c_type = a_type;
    if (acc_type == data_type::undef) acc_type = c_type;

    status_t status;

    // Check inputs
    status = check_gemm_input(
            *transa, *transb, m, n, k, lda, ldb, ldc, alpha, beta);
    if (status != dnnl_success) return status;

    // Create engine
    cl_context ocl_ctx;
    OCL_CHECK(clGetCommandQueueInfo(
            queue, CL_QUEUE_CONTEXT, sizeof(ocl_ctx), &ocl_ctx, nullptr));

    cl_device_id ocl_dev;
    OCL_CHECK(clGetCommandQueueInfo(
            queue, CL_QUEUE_DEVICE, sizeof(ocl_dev), &ocl_dev, nullptr));

    std::unique_ptr<ocl_gpu_engine_t> engine;
    engine_t *engine_ptr;

    status = ocl_engine_factory_t(engine_kind::gpu)
                     .engine_create(&engine_ptr, ocl_dev, ocl_ctx);
    if (status != status::success) return status;
    engine.reset(utils::downcast<ocl_gpu_engine_t *>(engine_ptr));

    // Create stream
    std::unique_ptr<stream_t> s;
    stream_t *s_ptr;
    status = engine->create_stream(&s_ptr, queue);
    if (status != status::success) return status;
    s.reset(s_ptr);

    // Create primitive descriptor
    auto op_desc = gemm_desc_t();
    op_desc.primitive_kind = dnnl_gemm;
    op_desc.transa = (*transa == 'n' || *transa == 'N') ? transpose::notrans
                                                        : transpose::trans;
    op_desc.transb = (*transb == 'n' || *transb == 'N') ? transpose::notrans
                                                        : transpose::trans;
    op_desc.batch = 1;
    op_desc.m = m;
    op_desc.n = n;
    op_desc.k = k;
    op_desc.stride_a = lda;
    op_desc.stride_b = ldb;
    op_desc.stride_c = ldc;
    op_desc.lda = lda;
    op_desc.ldb = ldb;
    op_desc.ldc = ldc;
    op_desc.a_type = a_type;
    op_desc.b_type = b_type;
    op_desc.c_type = c_type;
    op_desc.acc_type = acc_type;

    dnnl_memory_desc_t a_desc, b_desc, c_desc;

    status = create_gemm_memory_desc(&a_desc, &op_desc, 0, a_type);
    assert(status == status::success);
    status = create_gemm_memory_desc(&b_desc, &op_desc, 1, b_type);
    assert(status == status::success);
    status = create_gemm_memory_desc(&c_desc, &op_desc, 2, c_type);
    assert(status == status::success);

    std::unique_ptr<primitive_desc_iface_t> pd;
    primitive_attr_t attr;
    if (alpha != 1.0f) attr.output_scales_.set(alpha);
    if (beta != 0.0f) attr.post_ops_.append_sum(beta);

    primitive_desc_iface_t *pd_ptr;
    status = dnnl_primitive_desc_create(
            &pd_ptr, &op_desc, &attr, engine.get(), nullptr);
    if (status != status::success) return status;
    pd.reset(pd_ptr);

    // Create memory objects
    std::unique_ptr<memory_t> a_mem(new memory_t(
            engine.get(), &a_desc, memory_flags_t::use_runtime_ptr, a));
    std::unique_ptr<memory_t> b_mem(new memory_t(
            engine.get(), &b_desc, memory_flags_t::use_runtime_ptr, b));
    std::unique_ptr<memory_t> c_mem(new memory_t(
            engine.get(), &c_desc, memory_flags_t::use_runtime_ptr, c));
    if (!a_mem || !b_mem || !c_mem) return status::out_of_memory;

    a_mem->memory_storage()->set_offset(
            offset_a * types::data_type_size(a_type));
    b_mem->memory_storage()->set_offset(
            offset_b * types::data_type_size(b_type));
    c_mem->memory_storage()->set_offset(
            offset_c * types::data_type_size(c_type));

    // Create primitive
    primitive_iface_t *gemm_prim;
    status = pd->create_primitive_iface(&gemm_prim);
    if (status != status::success) return status;

    exec_args_t args = {
            {DNNL_ARG_SRC, {a_mem.get(), true}},
            {DNNL_ARG_WEIGHTS, {b_mem.get(), true}},
            {DNNL_ARG_DST, {c_mem.get(), false}},
    };

    exec_ctx_t ctx(s.get(), std::move(args));
    status = gemm_prim->execute(ctx);
    gemm_prim->release();
    if (status != status::success) return status;

    return s->wait();
}

dnnl_status_t gemm_x8x8s32(cl_command_queue queue, const char *transa,
        const char *transb, const char *offsetc, dim_t m, dim_t n, dim_t k,
        cl_float alpha, cl_mem a, dim_t offset_a, dim_t lda, dim_t ao, cl_mem b,
        dim_t offset_b, dim_t ldb, dim_t bo, cl_float beta, cl_mem c,
        dim_t offset_c, dim_t ldc, cl_mem co, dim_t offset_co,
        data_type_t a_type, data_type_t b_type, data_type_t c_type) {

    status_t status;

    // Check inputs
    status = check_gemm_x8x8s32_input(
            *offsetc, *transa, *transb, m, n, k, lda, ldb, ldc, alpha, beta);
    if (status != dnnl_success) return status;

    // Create engine
    cl_context ocl_ctx;
    OCL_CHECK(clGetCommandQueueInfo(
            queue, CL_QUEUE_CONTEXT, sizeof(ocl_ctx), &ocl_ctx, nullptr));

    cl_device_id ocl_dev;
    OCL_CHECK(clGetCommandQueueInfo(
            queue, CL_QUEUE_DEVICE, sizeof(ocl_dev), &ocl_dev, nullptr));

    std::unique_ptr<ocl_gpu_engine_t> engine;
    engine_t *engine_ptr;

    status = ocl_engine_factory_t(engine_kind::gpu)
                     .engine_create(&engine_ptr, ocl_dev, ocl_ctx);
    if (status != status::success) return status;
    engine.reset(utils::downcast<ocl_gpu_engine_t *>(engine_ptr));

    // Create stream
    std::unique_ptr<stream_t> s;
    stream_t *s_ptr;
    status = engine->create_stream(&s_ptr, queue);
    if (status != status::success) return status;
    s.reset(s_ptr);

    // Create operation descriptor
    auto op_desc = gemm_desc_t();
    op_desc.primitive_kind = dnnl_gemm;
    op_desc.transa = (*transa == 'n' || *transa == 'N') ? transpose::notrans
                                                        : transpose::trans;
    op_desc.transb = (*transb == 'n' || *transb == 'N') ? transpose::notrans
                                                        : transpose::trans;
    op_desc.batch = 1;
    op_desc.m = m;
    op_desc.n = n;
    op_desc.k = k;
    op_desc.lda = lda;
    op_desc.ldb = ldb;
    op_desc.ldc = ldc;
    op_desc.stride_a = lda;
    op_desc.stride_b = ldb;
    op_desc.stride_c = ldc;
    op_desc.a_type = a_type;
    op_desc.b_type = b_type;
    op_desc.c_type = c_type;
    op_desc.acc_type = c_type;

    dnnl_memory_desc_t a_desc, b_desc, c_desc, co_desc;

    status = create_gemm_memory_desc(&a_desc, &op_desc, 0, a_type);
    if (status != status::success) return status;
    status = create_gemm_memory_desc(&b_desc, &op_desc, 1, b_type);
    if (status != status::success) return status;
    status = create_gemm_memory_desc(&c_desc, &op_desc, 2, c_type);
    if (status != status::success) return status;
    status = create_gemm_memory_desc(&co_desc, &op_desc, 2, c_type);
    if (status != status::success) return status;

    // Create primitive descriptor
    std::unique_ptr<primitive_desc_iface_t> pd;
    primitive_attr_t attr;

    auto &zp = attr.zero_points_;
    switch (*offsetc) {
        case 'f':
        case 'F': status = zp.set(DNNL_ARG_DST, DNNL_RUNTIME_S32_VAL); break;
        case 'c':
        case 'C':
            status = zp.set(DNNL_ARG_DST, 1, 1 << 1, &DNNL_RUNTIME_S32_VAL);
            break;
        case 'r':
        case 'R':
            status = zp.set(DNNL_ARG_DST, 1, 1 << 0, &DNNL_RUNTIME_S32_VAL);
            break;
        default: status = status::invalid_arguments;
    }
    if (status != status::success) return status;

    if (ao != 0) {
        status = zp.set(DNNL_ARG_SRC, ao);
        if (status != status::success) return status;
    }
    if (bo != 0) {
        zp.set(DNNL_ARG_WEIGHTS, bo);
        if (status != status::success) return status;
    }

    if (alpha != 1.0f) attr.output_scales_.set(alpha);
    if (beta != 0.0f) attr.post_ops_.append_sum(beta);

    primitive_desc_iface_t *pd_ptr;
    status = dnnl_primitive_desc_create(
            &pd_ptr, &op_desc, &attr, engine.get(), nullptr);
    if (status != status::success) return status;
    pd.reset(pd_ptr);

    // Create memory objects
    std::unique_ptr<memory_t> a_mem(new memory_t(
            engine.get(), &a_desc, memory_flags_t::use_runtime_ptr, a));
    std::unique_ptr<memory_t> b_mem(new memory_t(
            engine.get(), &b_desc, memory_flags_t::use_runtime_ptr, b));
    std::unique_ptr<memory_t> c_mem(new memory_t(
            engine.get(), &c_desc, memory_flags_t::use_runtime_ptr, c));
    std::unique_ptr<memory_t> co_mem(new memory_t(
            engine.get(), &co_desc, memory_flags_t::use_runtime_ptr, co));

    if (!a_mem || !b_mem || !c_mem || !co_mem) return status::out_of_memory;

    a_mem->memory_storage()->set_offset(
            offset_a * types::data_type_size(a_type));
    b_mem->memory_storage()->set_offset(
            offset_b * types::data_type_size(b_type));
    c_mem->memory_storage()->set_offset(
            offset_c * types::data_type_size(c_type));
    co_mem->memory_storage()->set_offset(
            offset_co * types::data_type_size(c_type));

    // Create primitive
    primitive_iface_t *gemm_prim;
    status = pd->create_primitive_iface(&gemm_prim);
    if (status != status::success) return status;

    exec_args_t args = {
            {DNNL_ARG_SRC, {a_mem.get(), true}},
            {DNNL_ARG_WEIGHTS, {b_mem.get(), true}},
            {DNNL_ARG_DST | DNNL_ARG_ATTR_ZERO_POINTS, {co_mem.get(), true}},
            {DNNL_ARG_DST, {c_mem.get(), false}},
    };

    exec_ctx_t ctx(s.get(), std::move(args));
    status = gemm_prim->execute(ctx);
    gemm_prim->release();
    if (status != status::success) return status;

    return s->wait();
}

const char *c2f_offsetC(const char *offC) {
    if (offC) {
        if (offC[0] == 'R' || offC[0] == 'r') return "C";
        if (offC[0] == 'C' || offC[0] == 'c') return "R";
    }
    return offC;
}

} // namespace

extern "C" {
dnnl_status_t DNNL_API dnnl_ocl_sgemm(cl_command_queue queue, char transa,
        char transb, dim_t m, dim_t n, dim_t k, cl_float alpha, cl_mem a,
        dim_t offset_a, dim_t lda, cl_mem b, dim_t offset_b, dim_t ldb,
        cl_float beta, cl_mem c, dim_t offset_c, dim_t ldc) {
    return gemm_generic(queue, &transb, &transa, n, m, k, alpha, b, offset_b,
            ldb, a, offset_a, lda, beta, c, offset_c, ldc, data_type::f32);
}

dnnl_status_t DNNL_API dnnl_ocl_hgemm(cl_command_queue queue, char transa,
        char transb, dim_t m, dim_t n, dim_t k, cl_float alpha, cl_mem a,
        dim_t offset_a, dim_t lda, cl_mem b, dim_t offset_b, dim_t ldb,
        cl_float beta, cl_mem c, dim_t offset_c, dim_t ldc) {
    return gemm_generic(queue, &transb, &transa, n, m, k, alpha, b, offset_b,
            ldb, a, offset_a, lda, beta, c, offset_c, ldc, data_type::f16);
}

dnnl_status_t DNNL_API dnnl_ocl_gemm_f16f16f32(cl_command_queue queue,
        char transa, char transb, dim_t m, dim_t n, dim_t k, cl_float alpha,
        cl_mem a, dim_t offset_a, dim_t lda, cl_mem b, dim_t offset_b,
        dim_t ldb, cl_float beta, cl_mem c, dim_t offset_c, dim_t ldc) {
    return gemm_generic(queue, &transb, &transa, n, m, k, alpha, b, offset_b,
            ldb, a, offset_a, lda, beta, c, offset_c, ldc, data_type::f16,
            data_type::f16, data_type::f32);
}

dnnl_status_t DNNL_API dnnl_ocl_gemm_bf16bf16f32(cl_command_queue queue,
        char transa, char transb, dim_t m, dim_t n, dim_t k, cl_float alpha,
        cl_mem a, dim_t offset_a, dim_t lda, cl_mem b, dim_t offset_b,
        dim_t ldb, cl_float beta, cl_mem c, dim_t offset_c, dim_t ldc) {
    return gemm_generic(queue, &transb, &transa, n, m, k, alpha, b, offset_b,
            ldb, a, offset_a, lda, beta, c, offset_c, ldc, data_type::bf16,
            data_type::bf16, data_type::f32);
}

dnnl_status_t DNNL_API dnnl_ocl_gemm_bf16bf16bf16(cl_command_queue queue,
        char transa, char transb, dim_t m, dim_t n, dim_t k, cl_float alpha,
        cl_mem a, dim_t offset_a, dim_t lda, cl_mem b, dim_t offset_b,
        dim_t ldb, cl_float beta, cl_mem c, dim_t offset_c, dim_t ldc) {
    return gemm_generic(queue, &transb, &transa, n, m, k, alpha, b, offset_b,
            ldb, a, offset_a, lda, beta, c, offset_c, ldc, data_type::bf16,
            data_type::bf16, data_type::bf16, data_type::f32);
}

dnnl_status_t DNNL_API dnnl_ocl_gemm_s8s8s32(cl_command_queue queue,
        char transa, char transb, char offsetc, dim_t m, dim_t n, dim_t k,
        cl_float alpha, cl_mem a, dim_t offset_a, dim_t lda, int8_t ao,
        cl_mem b, dim_t offset_b, dim_t ldb, int8_t bo, cl_float beta, cl_mem c,
        dim_t offset_c, dim_t ldc, cl_mem co, dim_t offset_co) {

    return gemm_x8x8s32(queue, &transb, &transa, c2f_offsetC(&offsetc), n, m, k,
            alpha, b, offset_b, ldb, bo, a, offset_a, lda, ao, beta, c,
            offset_c, ldc, co, offset_co, data_type::s8, data_type::s8,
            data_type::s32);
}

dnnl_status_t DNNL_API dnnl_ocl_gemm_u8s8s32(cl_command_queue queue,
        char transa, char transb, char offsetc, dim_t m, dim_t n, dim_t k,
        cl_float alpha, cl_mem a, dim_t offset_a, dim_t lda, uint8_t ao,
        cl_mem b, dim_t offset_b, dim_t ldb, int8_t bo, cl_float beta, cl_mem c,
        dim_t offset_c, dim_t ldc, cl_mem co, dim_t offset_co) {

    return gemm_x8x8s32(queue, &transb, &transa, c2f_offsetC(&offsetc), n, m, k,
            alpha, b, offset_b, ldb, bo, a, offset_a, lda, ao, beta, c,
            offset_c, ldc, co, offset_co, data_type::s8, data_type::u8,
            data_type::s32);
}

dnnl_status_t DNNL_API dnnl_ocl_gemm_s8u8s32(cl_command_queue queue,
        char transa, char transb, char offsetc, dim_t m, dim_t n, dim_t k,
        cl_float alpha, cl_mem a, dim_t offset_a, dim_t lda, int8_t ao,
        cl_mem b, dim_t offset_b, dim_t ldb, uint8_t bo, cl_float beta,
        cl_mem c, dim_t offset_c, dim_t ldc, cl_mem co, dim_t offset_co) {

    return gemm_x8x8s32(queue, &transb, &transa, c2f_offsetC(&offsetc), n, m, k,
            alpha, b, offset_b, ldb, bo, a, offset_a, lda, ao, beta, c,
            offset_c, ldc, co, offset_co, data_type::u8, data_type::s8,
            data_type::s32);
}

dnnl_status_t DNNL_API dnnl_ocl_gemm_u8u8s32(cl_command_queue queue,
        char transa, char transb, char offsetc, dim_t m, dim_t n, dim_t k,
        cl_float alpha, cl_mem a, dim_t offset_a, dim_t lda, uint8_t ao,
        cl_mem b, dim_t offset_b, dim_t ldb, uint8_t bo, cl_float beta,
        cl_mem c, dim_t offset_c, dim_t ldc, cl_mem co, dim_t offset_co) {

    return gemm_x8x8s32(queue, &transb, &transa, c2f_offsetC(&offsetc), n, m, k,
            alpha, b, offset_b, ldb, bo, a, offset_a, lda, ao, beta, c,
            offset_c, ldc, co, offset_co, data_type::u8, data_type::u8,
            data_type::s32);
}
}
