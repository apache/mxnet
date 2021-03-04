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

#include <cassert>

#include "dnnl_thread.hpp"
#include "dnnl_traits.hpp"
#include "stream.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "memory.hpp"
#include "primitive_exec_types.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::status;

enum blk_kind_t { a, b, c, ab, ba, bc, cb };

template <data_type_t dt, blk_kind_t blk_kind, int blksize>
void typed_zero_pad_blk(const memory_desc_wrapper &m_d, void *data_handle) {
    /* Note: for bf16 memory,
     * use uint16_t for initialization of padding to zero,
     * in order to avoid using assign operators defined in bfloat16_t.
     * This allows user will be to create bf16 memory
     * on non-avx512_core machines. */
    using data_t = typename utils::conditional<dt == bf16, uint16_t,
            typename prec_traits<dt>::type>::type;
    auto data = reinterpret_cast<data_t *>(data_handle);
    const auto &dims = m_d.dims();
    const auto &pdims = m_d.padded_dims();
    const auto &blk = m_d.blocking_desc();
    auto dim_is_blocked = [&](int dim) {
        for (int i = 0; i < blk.inner_nblks; i++)
            if (blk.inner_idxs[i] == dim) return true;
        return false;
    };
    bool A_blocked = dim_is_blocked(0), B_blocked = dim_is_blocked(1),
         C_blocked = dim_is_blocked(2);

    assert(blk.inner_nblks < 4);
    assert((A_blocked || B_blocked || C_blocked) || (A_blocked && B_blocked)
            || (C_blocked && B_blocked));

    const int a_tail_s = A_blocked ? dims[0] % blksize : 0;
    const int b_tail_s = B_blocked ? dims[1] % blksize : 0;
    const int c_tail_s = C_blocked ? dims[2] % blksize : 0;
    assert(a_tail_s || b_tail_s || c_tail_s);

    const int ndims = m_d.ndims();
    assert(1 <= ndims && ndims <= 6);
    const int A = A_blocked ? pdims[0] / blksize : dims[0];
    const int B = ndims <= 1 ? 1 : B_blocked ? pdims[1] / blksize : dims[1];
    const int C = ndims <= 2 ? 1 : C_blocked ? pdims[2] / blksize : dims[2];
    const int D = ndims <= 3 ? 1 : dims[3];
    const int E = ndims <= 4 ? 1 : dims[4];
    const int F = ndims <= 5 ? 1 : dims[5];
    const int inner_blk = blk.inner_nblks == 3 ? blk.inner_blks[2] : 1;

    auto zeroize_tail = [&](data_t *d, const int tail_s) {
        for (int b = tail_s; b < blksize; ++b)
            d[b] = 0;
    };
    auto zeroize_tail_inner = [&](data_t *d, const int tail_s) {
        for (int b1 = 0; b1 < blksize; ++b1)
            for (int b2 = tail_s; b2 < blksize; ++b2)
                d[(b1 / inner_blk) * blksize * inner_blk + inner_blk * b2
                        + b1 % inner_blk]
                        = 0;
    };
    auto zeroize_tail_outer = [&](data_t *d, const int tail_s) {
        for (int b1 = tail_s; b1 < blksize; ++b1)
            for (int b2 = 0; b2 < blksize; ++b2)
                d[(b1 / inner_blk) * blksize * inner_blk + inner_blk * b2
                        + b1 % inner_blk]
                        = 0;
    };

    if (c_tail_s) {
        parallel_nd(A, B, D, E, F, [&](int a, int b, int d, int e, int f) {
            auto x = &data[m_d.blk_off(a, b, C - 1, d, e, f)];
            if (blk_kind == c)
                zeroize_tail(x, c_tail_s);
            else if (blk_kind == bc)
                zeroize_tail_inner(x, c_tail_s);
            else if (blk_kind == cb)
                zeroize_tail_outer(x, c_tail_s);
        });
    }

    if (b_tail_s) {
        parallel_nd(A, C, D, E, F, [&](int a, int c, int d, int e, int f) {
            auto x = &data[m_d.blk_off(a, B - 1, c, d, e, f)];
            if (blk_kind == b)
                zeroize_tail(x, b_tail_s);
            else if (blk_kind == ab || blk_kind == cb)
                zeroize_tail_inner(x, b_tail_s);
            else if (blk_kind == ba || blk_kind == bc)
                zeroize_tail_outer(x, b_tail_s);
        });
    }

    if (a_tail_s) {
        parallel_nd(B, C, D, E, F, [&](int b, int c, int d, int e, int f) {
            auto x = &data[m_d.blk_off(A - 1, b, c, d, e, f)];
            if (blk_kind == a)
                zeroize_tail(x, a_tail_s);
            else if (blk_kind == ba)
                zeroize_tail_inner(x, a_tail_s);
            else if (blk_kind == ab)
                zeroize_tail_outer(x, a_tail_s);
        });
    }
}

/*
 * all
 */
template <data_type_t dt>
void typed_zero_pad_generic_blocked(
        const memory_desc_wrapper &m_d, void *data_handle) {
    /* Note: for bf16 memory,
     * use uint16_t for initialization of padding to zero,
     * in order to avoid using assign operators defined in bfloat16_t.
     * This allows user will be to create bf16 memory
     * on non-avx512_core machines. */
    using data_t = typename utils::conditional<dt == bf16, uint16_t,
            typename prec_traits<dt>::type>::type;
    auto data = reinterpret_cast<data_t *>(data_handle);
    const int ndims = m_d.ndims();
    const auto &dims = m_d.dims();
    const auto &pdims = m_d.padded_dims();

    const ptrdiff_t nelems = (ptrdiff_t)m_d.nelems(true);

    /* [D_0] .. [D_k][D_k+1] .. [D_ndim - 1]
     *            |  \                     /
     *            |   ---------------------
     *           has        contiguous
     *         padding
     *
     * step     <-- D_k+1 * ... * D_ndims-1
     * step_dim <-- k
     */

    ptrdiff_t step = 1;
    int step_dim = ndims - 1;
    for (; step_dim >= 0; --step_dim) {
        if (dims[step_dim] != pdims[step_dim]) break;
        step *= dims[step_dim];
    }

    assert(step_dim >= 0 && "no zero padding is required");
    if (step_dim < 0) return;

    parallel_nd(nelems / step, [&](ptrdiff_t e1) {
        bool need_zero = false;

        ptrdiff_t idx = e1;
        for (int d = step_dim; d >= 0; --d) {
            if (idx % pdims[d] >= dims[d]) {
                need_zero = true;
                break;
            }
            idx /= pdims[d];
        }

        if (need_zero) {
            for (ptrdiff_t e0 = 0; e0 < step; ++e0)
                data[m_d.off_l(e1 * step + e0, true)] = 0;
        }
    });
}

template <data_type_t dt>
status_t typed_zero_pad(const memory_t *memory, const exec_ctx_t &ctx) {
    const memory_desc_wrapper mdw(memory->md());
    memory_storage_t *memory_storage = memory->memory_storage();

    if (mdw.format_kind() != format_kind::blocked) return unimplemented;

    if (mdw.nelems(false) == mdw.nelems(true)) return success;

    const size_t map_size = mdw.size();
    assert(map_size != DNNL_RUNTIME_SIZE_VAL);

    void *mapped_ptr
            = ctx.map_memory_storage(memory_storage, ctx.stream(), map_size);

    auto *data = static_cast<typename prec_traits<dt>::type *>(mapped_ptr);
    auto blk = mdw.blocking_desc();

    auto get_blksize = [&](int ind) {
        int blksize = 1;
        for (int i = 0; i < blk.inner_nblks; i++) {
            if (blk.inner_idxs[i] == ind) blksize *= blk.inner_blks[i];
        }
        return blksize;
    };
    const int blksize = get_blksize(blk.inner_idxs[0]);

#define CASE(blksize_, blk_kind) \
    do { \
        if (blksize == (blksize_)) { \
            typed_zero_pad_blk<dt, blk_kind, blksize_>(mdw, data); \
            ctx.unmap_memory_storage( \
                    memory_storage, mapped_ptr, ctx.stream()); \
            return success; \
        } \
    } while (0)

    switch (blk.inner_nblks) {
        case 1:
            if (blk.inner_idxs[0] == 0) {
                CASE(4, a);
                CASE(8, a);
                CASE(16, a);
            } else if (blk.inner_idxs[0] == 1) {
                CASE(4, b);
                CASE(8, b);
                CASE(16, b);
            }
            break;
        case 2:
        case 3:
            if (blk.inner_nblks == 3 && blk.inner_idxs[0] != blk.inner_idxs[2])
                break;
            if (blksize != get_blksize(blk.inner_idxs[1])) break;

            if (blk.inner_idxs[0] == 0 && blk.inner_idxs[1] == 1) {
                CASE(4, ab);
                CASE(8, ab);
                CASE(16, ab);
            } else if (blk.inner_idxs[0] == 1 && blk.inner_idxs[1] == 0) {
                CASE(4, ba);
                CASE(8, ba);
                CASE(16, ba);
            }
            if (blk.inner_idxs[0] == 1 && blk.inner_idxs[1] == 2) {
                CASE(4, bc);
                CASE(8, bc);
                CASE(16, bc);
            } else if (blk.inner_idxs[0] == 2 && blk.inner_idxs[1] == 1) {
                CASE(4, cb);
                CASE(8, cb);
                CASE(16, cb);
            }
            break;
        default: break;
    }

#undef CASE

    // the last line of defence
    typed_zero_pad_generic_blocked<dt>(mdw, data);

    ctx.unmap_memory_storage(memory_storage, mapped_ptr, ctx.stream());
    return success;
}

static status_t zero_pad(const memory_t *memory, const exec_ctx_t &ctx) {
    memory_desc_wrapper mdw(memory->md());
    switch (mdw.data_type()) {
        case f16: return typed_zero_pad<f16>(memory, ctx);
        case bf16: return typed_zero_pad<bf16>(memory, ctx);
        case f32: return typed_zero_pad<f32>(memory, ctx);
        case s32: return typed_zero_pad<s32>(memory, ctx);
        case s8: return typed_zero_pad<s8>(memory, ctx);
        case u8: return typed_zero_pad<u8>(memory, ctx);
        default: assert(!"memory is undefined"); return unimplemented;
    }
    return unimplemented;
}

status_t stream_t::zero_pad(const memory_t *memory, const exec_ctx_t &ctx) {
    return ::zero_pad(memory, ctx);
}

status_t memory_t::zero_pad(stream_t *stream) const {
    if (stream == nullptr) {
        engine_t *engine;
        engine = memory_storage()->engine();
        CHECK(engine->get_service_stream(stream));
    }
    return zero_pad(exec_ctx_t(stream));
}

status_t memory_t::zero_pad(const exec_ctx_t &ctx) const {
    memory_desc_wrapper mdw(md());
    const bool skip_zeroing = false || memory_storage()->is_null()
            || mdw.is_zero() || !mdw.is_blocking_desc();
    if (skip_zeroing) return success;

    stream_t *stream = ctx.stream();
    status_t status;
    if (stream == nullptr) {
        engine_t *engine;
        engine = memory_storage()->engine();
        CHECK(engine->get_service_stream(stream));
    }

    if (stream != nullptr)
        status = stream->zero_pad(this, ctx);
    else
        status = ::zero_pad(this, ctx);

    return status;
}
