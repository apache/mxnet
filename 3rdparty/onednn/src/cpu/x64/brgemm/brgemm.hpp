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

#ifndef CPU_X64_BRGEMM_BRGEMM_HPP
#define CPU_X64_BRGEMM_BRGEMM_HPP

#include "cpu/x64/brgemm/brgemm_amx.hpp"
#include "cpu/x64/brgemm/brgemm_types.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
/// Initializes a BRGEMM descriptor
///
/// @param brg Output BRGEMM descriptor
/// @param type Type of batch
/// @param dt_a Data type of A matrix, can be
///     AVX512: f32, u8(row-major layout), s8(column-major layout), bf16
///     AMX: u8, s8, bf16
/// @param dt_b Data type of B matrix
///     AVX512: f32, s8(row-major layout), u8(column-major layout), bf16
///     AMX: u8, s8, bf16
/// @note
///     Data type of matrix C depends on data types of matrices A and B
///     If A and B have integer u8/s8 data type, C has int32 data type
///     If A and B have bfloat16 or f32 data type, C has f32 data type
/// @param transA Specifies the form of A used in the matrix multiplication
///        'false' - A is not transposed, 'true' - A is transposed
/// @param transB Specifies the form of B used in the matrix multiplication
///        'false' - B is not transposed, 'true' - B is transposed
/// @param layout Specifies whether two-dimensional array storage is row-major
///        (brgemm_row_major) or column-major (brgemm_col_major).
/// @param alpha Specifies the scalar alpha
/// @param beta Specifies the scalar beta
/// @param LDA Specifies the leading dimension of matrix A.
///        LDA must be at least max(1, K)
/// @param LDB Specifies the leading dimension of matrix B.
///        LDB must be at least max(1, N)
/// @param LDC Specifies the leading dimension of matrix C.
///       LDC must be at least max(1, N)
/// @param M Specifies the number of rows of the matrix A and of the matrix C.
/// @param N Specifies the number of columns of the matrix B and
///        the number of columns of the matrix C
/// @param K Specifies the number of columns of the matrix A and
///        the number of rows of the matrces B
/// @param strides Strides between the matrices in the batch. Can be nullptr.
///
status_t brgemm_desc_init(brgemm_t *brg, brgemm_batch_kind_t type,
        impl::data_type_t dt_a, impl::data_type_t dt_b, bool transA,
        bool transB, brgemm_layout_t layout, float alpha, float beta, dim_t LDA,
        dim_t LDB, dim_t LDC, dim_t M, dim_t N, dim_t K,
        const brgemm_strides_t *strides = nullptr);

/// Adds post-operations to BRGEMM descriptor
///
/// @param brg Output BRGEMM descriptor
/// @param attr Primitive attributes (can be NULL). Specifies element-wise
///     operations
/// @param dt_d Specifies the data type of D matrix
///     Can be u8, s8, s32, bf16 or fp32
/// @param LDD Specifies the leading dimension of matrix D
///        LDD must be at least max(1, N)
/// @param dt_bias Specifies the data type Bias
///     Can be u8, s8, s32, bf16 or fp32
///
status_t brgemm_desc_add_postops(brgemm_t *brg, const primitive_attr_t *attr,
        impl::data_type_t dt_d, int LDD,
        impl::data_type_t dt_bias = impl::data_type::undef);

/// Generates a BRGEMM kernel based on descriptor
///
/// @param brg_kernel Output BRGEMM kernel
/// @param brg BRGEMM descritpor
///
status_t brgemm_kernel_create(
        brgemm_kernel_t **brg_kernel, const brgemm_t &brg);

/// Destroys a BRGEMM kernel
///
/// @param brg_kernel BRGEMM kernel
///
void brgemm_kernel_destroy(brgemm_kernel_t *brg_kernel);

/// Execute BRGEMM kernel (brgemm_addr version)
///
/// @note
///     Only BRGEMM kernel will be execute even if post-ops are added to BRGEMM
///     descriptor
/// @param brg_kernel BRGEMM kernel
/// @param bs Specifies the size of batch
/// @param addr_A Array of addresses of matrices A
/// @param addr_B Array of addresses of matrices B
/// @param ptr_C Pointer to destination matrix C
/// @param scratch Scratchpad needed for AMX version, can be nullptr for
///     avx512 version
///
void brgemm_kernel_execute(const brgemm_kernel_t *brg_kernel, int bs,
        const void **addr_A, const void **addr_B, void *ptr_C,
        void *scratch = nullptr);

/// Execute BRGEMM kernel (brgemm_offs version)
///
/// @note
///     Only BRGEMM kernel will be execute even if post-ops are added to BRGEMM
///     descriptor
/// @param brg_kernel BRGEMM kernel
/// @param bs Specifies the size of batch
/// @param addr_A Pointer to first matrix A in the batch
/// @param addr_B Pointer to first matrix B in the batch
/// @param offs_A Array of offsets to matrices A in the batch
/// @param offs_B Array of offsets to matrices B in the batch
/// @param ptr_C Pointer to destination matrix C
/// @param scratch Scratchpad needed for AMX version, can be nullptr for
///     avx512 version
///
void brgemm_kernel_execute(const brgemm_kernel_t *brg_kernel, int bs,
        const void *addr_A, const dim_t *offs_A, const void *addr_B,
        const dim_t *offs_B, void *ptr_C, void *scratch = nullptr);

/// Execute BRGEMM kernel (brgemm_strd version)
///
/// @note
///     Only BRGEMM kernel will be execute even if post-ops are added to BRGEMM
///     descriptor
/// @param brg_kernel BRGEMM kernel
/// @param bs Specifies the size of batch
/// @param addr_A Pointer to first matrix A in the batch
/// @param addr_B Pointer to first matrix B in the batch
/// @param ptr_C Pointer to destination matrix C
/// @param scratch Scratchpad needed for AMX version, can be nullptr for
///     avx512 version
///
void brgemm_kernel_execute(const brgemm_kernel_t *brg_kernel, int bs,
        const void *addr_A, const void *addr_B, void *ptr_C,
        void *scratch = nullptr);

/// Execute BRGEMM kernel (brgemm_addr version)
///
/// @note
///     BRGEMM kernel and post-operations will be executed
/// @param brg_kernel BRGEMM kernel
/// @param bs Specifies the size of batch
/// @param addr_A Array of addresses of matrices A
/// @param addr_B Array of addresses of matrices B
/// @param ptr_C Pointer to matrix C
/// @param ptr_D Pointer to destination matrix D
/// @param bias Vector of bias (vector length is N)
/// @param scales Vector of scales (vector length is N)
/// @param scratch Scratchpad needed for AMX version, can be nullptr for
///     avx512 version
///
void brgemm_kernel_execute_postops(const brgemm_kernel_t *brg_kernel, int bs,
        const void **addr_A, const void **addr_B, void *ptr_C, void *ptr_D,
        const void *bias, const float *scales, void *scratch = nullptr);

/// Execute BRGEMM kernel (brgemm_offs version)
///
/// @note
///     BRGEMM kernel and post-operations will be executed
/// @param brg_kernel BRGEMM kernel
/// @param bs Specifies the size of batch
/// @param addr_A Pointer to first matrix A in the batch
/// @param addr_B Pointer to first matrix B in the batch
/// @param offs_A Array of offsets to matrices A in the batch
/// @param offs_B Array of offsets to matrices B in the batch
/// @param ptr_C Pointer to destination matrix C
/// @param ptr_D Pointer to destination matrix D
/// @param bias Vector of bias (vector length is N)
/// @param scales Vector of scales (vector length is N)
/// @param scratch Scratchpad needed for AMX version, can be nullptr for
///     avx512 version
///
void brgemm_kernel_execute_postops(const brgemm_kernel_t *brg_kernel, int bs,
        const void *addr_A, const dim_t *offs_A, const void *addr_B,
        const dim_t *offs_B, void *ptr_C, void *ptr_D, const void *bias,
        const float *scales, void *scratch = nullptr);

/// Execute BRGEMM kernel (brgemm_strd version)
///
/// @note
///     BRGEMM kernel and post-operations will be executed
/// @param brg_kernel BRGEMM kernel
/// @param bs Specifies the size of batch
/// @param addr_A Pointer to first matrix A in the batch
/// @param addr_B Pointer to first matrix B in the batch
/// @param ptr_C Pointer to destination matrix C
/// @param ptr_D Pointer to destination matrix D
/// @param bias Vector of bias (vector length is N)
/// @param scales Vector of scales (vector length is N)
/// @param scratch Scratchpad needed for AMX version, can be nullptr for
///     avx512 version
///
void brgemm_kernel_execute_postops(const brgemm_kernel_t *brg_kernel, int bs,
        const void *addr_A, const void *addr_B, void *ptr_C, void *ptr_D,
        const void *bias, const float *scales, void *scratch = nullptr);

/// AMX utilities: Creates a palette based on BRGEMM descriptor
///
/// @note
///     Caller is expected to subsequently configure AMX tiles by calling
///     amx_tile_configure(palette).
/// @param brg BRGEMM descritpor
/// @param palette 64 bytes array contains tiles configuration
///
status_t brgemm_init_tiles(const brgemm_t &brg, char palette[64]);

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
