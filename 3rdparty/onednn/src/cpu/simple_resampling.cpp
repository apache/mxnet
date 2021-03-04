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

#include <assert.h>
#include <float.h>
#include <math.h>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/simple_resampling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace format_tag;
using namespace resampling_utils;

template <impl::data_type_t data_type>
status_t simple_resampling_fwd_t<data_type>::init(engine_t *engine) {
    if (pd()->desc()->alg_kind == alg_kind::resampling_nearest)
        interpolate = &simple_resampling_fwd_t::nearest;
    else {
        if (pd()->ndims() == 5)
            interpolate = &simple_resampling_fwd_t::trilinear;
        else if (pd()->ndims() == 4)
            interpolate = &simple_resampling_fwd_t::bilinear;
        else
            interpolate = &simple_resampling_fwd_t::linear;

        fill_coeffs();
    }
    const memory_desc_wrapper src_d(pd()->src_md());
    inner_stride_ = src_d.blocking_desc().strides[pd()->ndims() - 1];
    nsp_outer_ = src_d.nelems(true)
            / (pd()->ID() * pd()->IH() * pd()->IW() * inner_stride_);
    stride_d_ = pd()->IH() * pd()->IW() * inner_stride_;
    stride_h_ = pd()->IW() * inner_stride_;
    stride_w_ = inner_stride_;
    return status::success;
}

template <impl::data_type_t data_type>
simple_resampling_fwd_t<data_type>::~simple_resampling_fwd_t() = default;

template <impl::data_type_t data_type>
void simple_resampling_fwd_t<data_type>::fill_coeffs() {
    using namespace resampling_utils;
    linear_coeffs_.reserve(pd()->OD() + pd()->OH() + pd()->OW());
    for (dim_t od = 0; od < pd()->OD(); od++)
        linear_coeffs_.push_back(linear_coeffs_t(od, pd()->OD(), pd()->ID()));
    for (dim_t oh = 0; oh < pd()->OH(); oh++)
        linear_coeffs_.push_back(linear_coeffs_t(oh, pd()->OH(), pd()->IH()));
    for (dim_t ow = 0; ow < pd()->OW(); ow++)
        linear_coeffs_.push_back(linear_coeffs_t(ow, pd()->OW(), pd()->IW()));
}

template <impl::data_type_t data_type>
void simple_resampling_fwd_t<data_type>::nearest(
        const data_t *src, data_t *dst, dim_t od, dim_t oh, dim_t ow) const {
    dim_t id = nearest_idx(od, pd()->OD(), pd()->ID());
    dim_t ih = nearest_idx(oh, pd()->OH(), pd()->IH());
    dim_t iw = nearest_idx(ow, pd()->OW(), pd()->IW());

    PRAGMA_OMP_SIMD()
    for (dim_t innermost_el = 0; innermost_el < inner_stride_; innermost_el++)
        dst[innermost_el] = src[id * stride_d_ + ih * stride_h_ + iw * stride_w_
                + innermost_el];
}

template <impl::data_type_t data_type>
void simple_resampling_fwd_t<data_type>::linear(
        const data_t *src, data_t *dst, dim_t od, dim_t oh, dim_t ow) const {
    linear_coeffs_t iw = linear_coeffs_[pd()->OD() + pd()->OH() + ow];

    PRAGMA_OMP_SIMD()
    for (dim_t innermost_el = 0; innermost_el < inner_stride_; innermost_el++) {
        float d = 0;
        for (int k = 0; k < 2; k++)
            d += (float)src[iw.idx[k] * stride_w_ + innermost_el] * iw.wei[k];
        dst[innermost_el] = d;
    }
}

template <impl::data_type_t data_type>
void simple_resampling_fwd_t<data_type>::bilinear(
        const data_t *src, data_t *dst, dim_t od, dim_t oh, dim_t ow) const {
    linear_coeffs_t ih = linear_coeffs_[pd()->OD() + oh];
    linear_coeffs_t iw = linear_coeffs_[pd()->OD() + pd()->OH() + ow];

    PRAGMA_OMP_SIMD()
    for (dim_t innermost_el = 0; innermost_el < inner_stride_; innermost_el++) {
        float d = 0;
        for_(int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
            d += (float)src[ih.idx[j] * stride_h_ + iw.idx[k] * stride_w_
                         + innermost_el]
                    * ih.wei[j] * iw.wei[k];
        dst[innermost_el] = d;
    }
}

template <impl::data_type_t data_type>
void simple_resampling_fwd_t<data_type>::trilinear(
        const data_t *src, data_t *dst, dim_t od, dim_t oh, dim_t ow) const {
    linear_coeffs_t id = linear_coeffs_[od];
    linear_coeffs_t ih = linear_coeffs_[pd()->OD() + oh];
    linear_coeffs_t iw = linear_coeffs_[pd()->OD() + pd()->OH() + ow];

    PRAGMA_OMP_SIMD()
    for (dim_t innermost_el = 0; innermost_el < inner_stride_; innermost_el++) {
        float d = 0;
        for_(int i = 0; i < 2; i++)
        for_(int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
            d += (float)src[id.idx[i] * stride_d_ + ih.idx[j] * stride_h_
                         + iw.idx[k] * stride_w_ + innermost_el]
                    * id.wei[i] * ih.wei[j] * iw.wei[k];
        dst[innermost_el] = d;
    }
}

template <impl::data_type_t data_type>
void simple_resampling_fwd_t<data_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();

    parallel_nd(nsp_outer_, OD, OH, OW,
            [&](dim_t nsp0, dim_t od, dim_t oh, dim_t ow) {
                dim_t src_off = nsp0 * ID * IH * IW * inner_stride_;
                dim_t dst_off
                        = (nsp0 * OD * OH * OW + od * OH * OW + oh * OW + ow)
                        * inner_stride_;
                (this->*(interpolate))(
                        src + src_off, dst + dst_off, od, oh, ow);
            });
}

template struct simple_resampling_fwd_t<data_type::f32>;
template struct simple_resampling_fwd_t<data_type::bf16>;

template <impl::data_type_t data_type>
status_t simple_resampling_bwd_t<data_type>::init(engine_t *engine) {
    if (pd()->desc()->alg_kind == alg_kind::resampling_nearest)
        interpolate = &simple_resampling_bwd_t::nearest;
    else {
        if (pd()->ndims() == 5)
            interpolate = &simple_resampling_bwd_t::trilinear;
        else if (pd()->ndims() == 4)
            interpolate = &simple_resampling_bwd_t::bilinear;
        else
            interpolate = &simple_resampling_bwd_t::linear;

        fill_coeffs();
        fill_weights();
    }
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    inner_stride_ = diff_src_d.blocking_desc().strides[pd()->ndims() - 1];
    nsp_outer_ = diff_src_d.nelems(true)
            / (pd()->ID() * pd()->IH() * pd()->IW() * inner_stride_);
    stride_d_ = pd()->OH() * pd()->OW() * inner_stride_;
    stride_h_ = pd()->OW() * inner_stride_;
    stride_w_ = inner_stride_;
    return status::success;
}

template <impl::data_type_t data_type>
simple_resampling_bwd_t<data_type>::~simple_resampling_bwd_t() = default;

template <impl::data_type_t data_type>
void simple_resampling_bwd_t<data_type>::fill_coeffs() {
    using namespace resampling_utils;
    bwd_linear_coeffs_.reserve(pd()->ID() + pd()->IH() + pd()->IW());
    for (dim_t id = 0; id < pd()->ID(); id++)
        bwd_linear_coeffs_.push_back(
                bwd_linear_coeffs_t(id, pd()->OD(), pd()->ID()));
    for (dim_t ih = 0; ih < pd()->IH(); ih++)
        bwd_linear_coeffs_.push_back(
                bwd_linear_coeffs_t(ih, pd()->OH(), pd()->IH()));
    for (dim_t iw = 0; iw < pd()->IW(); iw++)
        bwd_linear_coeffs_.push_back(
                bwd_linear_coeffs_t(iw, pd()->OW(), pd()->IW()));
}

template <impl::data_type_t data_type>
void simple_resampling_bwd_t<data_type>::fill_weights() {
    using namespace resampling_utils;
    bwd_linear_weights_.reserve(2 * (pd()->OD() + pd()->OH() + pd()->OW()));
    for (dim_t od = 0; od < pd()->OD(); od++) {
        bwd_linear_weights_.push_back(
                linear_weight(0, od, pd()->OD(), pd()->ID()));
        bwd_linear_weights_.push_back(
                linear_weight(1, od, pd()->OD(), pd()->ID()));
    }
    for (dim_t oh = 0; oh < pd()->OH(); oh++) {
        bwd_linear_weights_.push_back(
                linear_weight(0, oh, pd()->OH(), pd()->IH()));
        bwd_linear_weights_.push_back(
                linear_weight(1, oh, pd()->OH(), pd()->IH()));
    }
    for (dim_t ow = 0; ow < pd()->OW(); ow++) {
        bwd_linear_weights_.push_back(
                linear_weight(0, ow, pd()->OW(), pd()->IW()));
        bwd_linear_weights_.push_back(
                linear_weight(1, ow, pd()->OW(), pd()->IW()));
    }
}

template <impl::data_type_t data_type>
void simple_resampling_bwd_t<data_type>::nearest(data_t *diff_src,
        const data_t *diff_dst, dim_t id, dim_t ih, dim_t iw) const {
    dim_t ow_start = ceil_idx(((float)iw * pd()->OW() / pd()->IW()) - 0.5f);
    dim_t oh_start = ceil_idx(((float)ih * pd()->OH() / pd()->IH()) - 0.5f);
    dim_t od_start = ceil_idx(((float)id * pd()->OD() / pd()->ID()) - 0.5f);
    dim_t ow_end = ceil_idx(((iw + 1.f) * pd()->OW() / pd()->IW()) - 0.5f);
    dim_t oh_end = ceil_idx(((ih + 1.f) * pd()->OH() / pd()->IH()) - 0.5f);
    dim_t od_end = ceil_idx(((id + 1.f) * pd()->OD() / pd()->ID()) - 0.5f);

    PRAGMA_OMP_SIMD()
    for (dim_t innermost_el = 0; innermost_el < inner_stride_; innermost_el++) {
        float sum = 0;
        for_(dim_t od = od_start; od < od_end; od++)
        for_(dim_t oh = oh_start; oh < oh_end; oh++)
        for (dim_t ow = ow_start; ow < ow_end; ow++) {
            sum += (float)diff_dst[od * stride_d_ + oh * stride_h_
                    + ow * stride_w_ + innermost_el];
        }
        diff_src[innermost_el] = sum;
    }
}

template <impl::data_type_t data_type>
void simple_resampling_bwd_t<data_type>::linear(data_t *diff_src,
        const data_t *diff_dst, dim_t id, dim_t ih, dim_t iw) const {
    bwd_linear_coeffs_t w = bwd_linear_coeffs_[pd()->ID() + pd()->IH() + iw];

    PRAGMA_OMP_SIMD()
    for (dim_t innermost_el = 0; innermost_el < inner_stride_; innermost_el++) {
        float sum = 0;
        for_(int k = 0; k < 2; k++)
        for (dim_t ow = w.start[k]; ow < w.end[k]; ow++) {
            sum += (float)diff_dst[ow * stride_w_ + innermost_el]
                    * bwd_linear_weights_[2 * (pd()->OD() + pd()->OH() + ow)
                            + k];
        }
        diff_src[innermost_el] = sum;
    }
}

template <impl::data_type_t data_type>
void simple_resampling_bwd_t<data_type>::bilinear(data_t *diff_src,
        const data_t *diff_dst, dim_t id, dim_t ih, dim_t iw) const {
    bwd_linear_coeffs_t h = bwd_linear_coeffs_[pd()->ID() + ih],
                        w = bwd_linear_coeffs_[pd()->ID() + pd()->IH() + iw];

    PRAGMA_OMP_SIMD()
    for (dim_t innermost_el = 0; innermost_el < inner_stride_; innermost_el++) {
        float sum = 0;
        for_(int j = 0; j < 2; j++)
        for_(int k = 0; k < 2; k++)
        for_(dim_t oh = h.start[j]; oh < h.end[j]; oh++)
        for (dim_t ow = w.start[k]; ow < w.end[k]; ow++) {
            sum += (float)diff_dst[oh * stride_h_ + ow * stride_w_
                           + innermost_el]
                    * bwd_linear_weights_[2 * (pd()->OD() + oh) + j]
                    * bwd_linear_weights_[2 * (pd()->OD() + pd()->OH() + ow)
                            + k];
        }
        diff_src[innermost_el] = sum;
    }
}

template <impl::data_type_t data_type>
void simple_resampling_bwd_t<data_type>::trilinear(data_t *diff_src,
        const data_t *diff_dst, dim_t id, dim_t ih, dim_t iw) const {
    bwd_linear_coeffs_t d = bwd_linear_coeffs_[id];
    bwd_linear_coeffs_t h = bwd_linear_coeffs_[pd()->ID() + ih];
    bwd_linear_coeffs_t w = bwd_linear_coeffs_[pd()->ID() + pd()->IH() + iw];

    PRAGMA_OMP_SIMD()
    for (dim_t innermost_el = 0; innermost_el < inner_stride_; innermost_el++) {
        float sum = 0;
        for_(int i = 0; i < 2; i++)
        for_(int j = 0; j < 2; j++)
        for_(int k = 0; k < 2; k++)
        for_(dim_t od = d.start[i]; od < d.end[i]; od++)
        for_(dim_t oh = h.start[j]; oh < h.end[j]; oh++)
        for (dim_t ow = w.start[k]; ow < w.end[k]; ow++) {
            sum += (float)diff_dst[od * stride_d_ + oh * stride_h_
                           + ow * stride_w_ + innermost_el]
                    * bwd_linear_weights_[2 * od + i]
                    * bwd_linear_weights_[2 * (pd()->OD() + oh) + j]
                    * bwd_linear_weights_[2 * (pd()->OD() + pd()->OH() + ow)
                            + k];
        }
        diff_src[innermost_el] = sum;
    }
}

template <impl::data_type_t data_type>
void simple_resampling_bwd_t<data_type>::execute_backward(
        const exec_ctx_t &ctx) const {
    const auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();

    parallel_nd(nsp_outer_, ID, IH, IW,
            [&](dim_t nsp, dim_t id, dim_t ih, dim_t iw) {
                dim_t diff_dst_off = nsp * OD * OH * OW * inner_stride_;
                dim_t diff_src_off
                        = (nsp * ID * IH * IW + id * IH * IW + ih * IW + iw)
                        * inner_stride_;
                (this->*(interpolate))(diff_src + diff_src_off,
                        diff_dst + diff_dst_off, id, ih, iw);
            });
}

template struct simple_resampling_bwd_t<data_type::f32>;
template struct simple_resampling_bwd_t<data_type::bf16>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
