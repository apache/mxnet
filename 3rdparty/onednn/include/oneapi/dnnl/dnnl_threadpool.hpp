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

#ifndef ONEAPI_DNNL_DNNL_THREADPOOL_HPP
#define ONEAPI_DNNL_DNNL_THREADPOOL_HPP

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_threadpool.h"

#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"

/// @addtogroup dnnl_api
/// @{

namespace dnnl {

/// @addtogroup dnnl_api_interop
/// @{

/// @addtogroup dnnl_api_threadpool_interop Threadpool interoperability API
/// API extensions to interact with the underlying Threadpool run-time.
/// @{

/// Threadpool interoperability namespace
namespace threadpool_interop {

/// Constructs an execution stream for the specified engine and threadpool.
///
/// @sa @ref dev_guide_threadpool
///
/// @param aengine Engine to create the stream on.
/// @param threadpool Pointer to an instance of a C++ class that implements
///     dnnl::threapdool_iface interface.
/// @returns An execution stream.
inline dnnl::stream make_stream(
        const dnnl::engine &aengine, threadpool_iface *threadpool) {
    dnnl_stream_t c_stream;
    dnnl::error::wrap_c_api(dnnl_threadpool_interop_stream_create(
                                    &c_stream, aengine.get(), threadpool),
            "could not create stream");
    return dnnl::stream(c_stream);
}

/// Returns the pointer to a threadpool that is used by an execution stream.
///
/// @sa @ref dev_guide_threadpool
///
/// @param astream An execution stream.
/// @returns Output pointer to an instance of a C++ class that implements
///     dnnl::threapdool_iface interface or NULL if the stream was created
///     without threadpool.
inline threadpool_iface *get_threadpool(const dnnl::stream &astream) {
    void *tp;
    dnnl::error::wrap_c_api(
            dnnl_threadpool_interop_stream_get_threadpool(astream.get(), &tp),
            "could not get stream threadpool");
    return static_cast<threadpool_iface *>(tp);
}

/// @copydoc dnnl_sgemm_tp()
inline status sgemm(char transa, char transb, dnnl_dim_t M, dnnl_dim_t N,
        dnnl_dim_t K, float alpha, const float *A, dnnl_dim_t lda,
        const float *B, dnnl_dim_t ldb, float beta, float *C, dnnl_dim_t ldc,
        threadpool_iface *tp) {
    return static_cast<status>(dnnl_threadpool_interop_sgemm(
            transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, tp));
}
/// @copydoc dnnl_gemm_u8s8s32_tp()
inline status gemm_u8s8s32(char transa, char transb, char offsetc, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, float alpha, const uint8_t *A,
        dnnl_dim_t lda, uint8_t ao, const int8_t *B, dnnl_dim_t ldb, int8_t bo,
        float beta, int32_t *C, dnnl_dim_t ldc, const int32_t *co,
        threadpool_iface *tp) {
    return static_cast<status>(
            dnnl_threadpool_interop_gemm_u8s8s32(transa, transb, offsetc, M, N,
                    K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co, tp));
}

/// @copydoc dnnl_gemm_s8s8s32_tp()
inline status gemm_s8s8s32(char transa, char transb, char offsetc, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, float alpha, const int8_t *A,
        dnnl_dim_t lda, int8_t ao, const int8_t *B, dnnl_dim_t ldb, int8_t bo,
        float beta, int32_t *C, dnnl_dim_t ldc, const int32_t *co,
        threadpool_iface *tp) {
    return static_cast<status>(
            dnnl_threadpool_interop_gemm_s8s8s32(transa, transb, offsetc, M, N,
                    K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co, tp));
}

} // namespace threadpool_interop

/// @} dnnl_api_threadpool_interop

/// @} dnnl_api_interop

} // namespace dnnl

/// @} dnnl_api

#endif
