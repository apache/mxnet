/*******************************************************************************
* Copyright 2016 Intel Corporation
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
*
* \file mkl_concat-inl.h
* \brief
* \author lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKL_CONCAT_INL_H_
#define MXNET_OPERATOR_MKL_MKL_CONCAT_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../operator_common.h"
#include "../channel_op_common.h"
#include "./mkl_util-inl.h"
namespace mxnet {
namespace op {


template<typename xpu, typename DType>
class MKLConcatOp : public Operator {
 public:
  static std::string getName() {
    return "MKLConcatOp";
  }
  explicit MKLConcatOp(ConcatParam param)
    : size_(param.num_args), dimension_(param.dim), init_mkldnn_(false) {
    concatFwd_ = static_cast<dnnPrimitive_t>(NULL);
    concatBwd_ = static_cast<dnnPrimitive_t>(NULL);
    fwd_top_data_ = MKLData<DType>::create();
    bwd_top_diff_ = MKLData<DType>::create();

    num_concats_ = param.num_args;
  }
  virtual ~MKLConcatOp() {
    dnnDelete<DType>(concatFwd_);
    dnnDelete<DType>(concatBwd_);
  }

 private:
  void LayerSetUp(const std::vector<mshadow::Tensor<xpu, 4, DType> > &data,
                  const mshadow::Tensor<xpu, 4, DType> &out,
                  size_t data_shape_size, size_t *split_channels_) {
    size_t dim_src = data_shape_size;
    size_t dim_dst = dim_src;
    num_concats_ = size_;
    channels_ = 0;

    for (size_t i = 1; i < num_concats_; ++i) {
      for (size_t j = 1; j < data_shape_size; ++j) {
        if (j == dimension_) continue;
        CHECK_EQ(data[0].shape_[j], data[i].shape_[j]);
      }
    }

    for (size_t i = 0; i < num_concats_; ++i) {
      CHECK_EQ((int)dim_src, data[i].shape_.kDimension);

      fwd_bottom_data_.push_back(MKLData<DType>::create());
      bwd_bottom_diff_.push_back(MKLData<DType>::create());
      fwd_bottom_data_[i]->name = "fwd_bottom_data_[i]";
      bwd_bottom_diff_[i]->name = "bwd_bottom_data[i]";

      size_t *sizes_src = new size_t[dim_src];
      size_t *strides_src = new size_t[dim_src];
      for (size_t d = 0; d < dim_src; ++d) {
        sizes_src[d] = data[i].shape_[dim_src - d - 1];
        strides_src[d] = (d == 0) ? 1 : strides_src[d - 1] * sizes_src[d - 1];
      }

      split_channels_[i] = data[i].shape_[1];
      channels_ += split_channels_[i];
      fwd_bottom_data_[i]->create_user_layout(dim_src, sizes_src, strides_src);
      bwd_bottom_diff_[i]->create_user_layout(dim_src, sizes_src, strides_src);
      delete[] sizes_src;
      delete[] strides_src;
    }
    size_t *sizes_dst = new size_t[dim_dst];
    size_t *strides_dst = new size_t[dim_dst];
    for (size_t d = 0; d < dim_dst; ++d) {
      if (d == 2)
        sizes_dst[d] = channels_;
      else
        sizes_dst[d] = data[0].shape_[dim_dst - 1 - d];
      strides_dst[d] = (d == 0) ? 1 : strides_dst[d - 1] * sizes_dst[d - 1];
    }
    bwd_top_diff_->create_user_layout(dim_dst, sizes_dst, strides_dst);
    fwd_top_data_->create_user_layout(dim_dst, sizes_dst, strides_dst);
    delete[] sizes_dst;
    delete[] strides_dst;
    concatFwd_ = NULL;
    concatBwd_ = NULL;
  }

 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(static_cast<int>(in_data.size()), size_);
    CHECK_EQ(out_data.size(), 1);
    CHECK_LT(dimension_, (size_t)in_data[concat_enum::kData0].ndim());
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 4, DType> > data(size_);
    Tensor<xpu, 4, DType> out;
    if (in_data[0].ndim() == 2) {
      for (int i = 0; i < size_; ++i) {
        Shape<4> dshape = Shape4(in_data[i].shape_[0],
                                 in_data[i].shape_[1], 1, 1);
        data[i] = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
          in_data[i], dshape, s);
      }
      Shape<4> dshape = Shape4(out_data[concat_enum::kOut].shape_[0],
                               out_data[concat_enum::kOut].shape_[1], 1, 1);
      out = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_data[concat_enum::kOut], dshape, s);
    } else if (in_data[0].ndim() == 3) {
      for (int i = 0; i < size_; ++i) {
        Shape<4> dshape = Shape4(in_data[i].shape_[0],
          in_data[i].shape_[1], in_data[i].shape_[2], 1);
        data[i] = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
          in_data[i], dshape, s);
      }
      Shape<4> dshape = Shape4(out_data[concat_enum::kOut].shape_[0],
        out_data[concat_enum::kOut].shape_[1],
        out_data[concat_enum::kOut].shape_[2], 1);
      out = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_data[concat_enum::kOut], dshape, s);
    } else {
      for (int i = 0; i < size_; ++i) {
        data[i] = mkl_experimental_direct_get<xpu, 4, DType>(in_data[i], s);
      }
      out = mkl_experimental_direct_get<xpu, 4, DType>(out_data[concat_enum::kOut], s);
    }
    size_t *split_channels_ = new size_t[num_concats_];
    if (!init_mkldnn_) {
      init_mkldnn_ = true;
      LayerSetUp(data, out, 4, split_channels_);
    }

    dnnError_t e;
    std::vector<void*> bottom_data;
    bool isFirstPass = (concatFwd_ == NULL);
    dnnLayout_t *layouts = NULL;
    if (isFirstPass) {
      layouts = new dnnLayout_t[num_concats_];
    }

    for (size_t i = 0; i < num_concats_; i++) {
      void * bottom_i = NULL;
#if MKL_EXPERIMENTAL == 1
      bottom_i = mkl_prv_data<DType>(in_data[i]);
      if (bottom_i != NULL) {
        if (isFirstPass) {
          std::shared_ptr<MKLData<DType> > mem_descr =
            mkl_get_mem_desc<DType>(in_data[i].Mkl_mem_);
          fwd_bottom_data_[i] = mem_descr;
          layouts[i] = mem_descr->layout_int;
        }
      }
#endif
      if (bottom_i == NULL) {
        bottom_i = data[i].dptr_;
        if (isFirstPass) {
          layouts[i] = fwd_bottom_data_[i]->layout_usr;
        }
      }

      bottom_data.push_back(reinterpret_cast<void *>(bottom_i));
    }

    if (isFirstPass) {
      e = dnnConcatCreate<DType>(&concatFwd_, NULL, num_concats_, layouts);
      CHECK_EQ(e, E_SUCCESS);

      fwd_top_data_->create_internal_layout(concatFwd_, dnnResourceDst);
      bwd_top_diff_->create_internal_layout(concatFwd_, dnnResourceDst);

      e = dnnSplitCreate<DType>(&concatBwd_, NULL, num_concats_,
            bwd_top_diff_->layout_int, split_channels_);
      CHECK_EQ(e, E_SUCCESS);

      for (size_t n = 0; n < num_concats_; ++n) {
        fwd_bottom_data_[n]->create_internal_layout(concatFwd_,
          (dnnResourceType_t)(dnnResourceMultipleSrc + n));
        bwd_bottom_diff_[n]->create_internal_layout(concatBwd_,
          (dnnResourceType_t)(dnnResourceMultipleDst + n));
      }
    }
    delete[] layouts;

    void *concat_res[dnnResourceNumber];
    for (size_t i = 0; i < num_concats_; ++i) {
      concat_res[dnnResourceMultipleSrc + i]
        = reinterpret_cast<void*>(bottom_data[i]);
    }

    concat_res[dnnResourceDst] = fwd_top_data_->get_output_ptr(out.dptr_,
      fwd_top_data_, out_data[concat_enum::kOut]);
    e = dnnExecute<DType>(concatFwd_, concat_res);
    CHECK_EQ(e, E_SUCCESS);
    delete[] split_channels_;
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_grad.size(), static_cast<size_t>(size_));
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 4, DType> > grad_in(size_);
    Tensor<xpu, 4, DType> grad;
    if (in_grad[0].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[concat_enum::kOut].shape_[0],
        out_grad[concat_enum::kOut].shape_[1], 1, 1);
      grad = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_grad[concat_enum::kOut], dshape, s);
      for (int i = 0; i < size_; ++i) {
        dshape = Shape4(in_grad[i].shape_[0],
          in_grad[i].shape_[1], 1, 1);
        grad_in[i] = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
          in_grad[i], dshape, s);
      }
    } else if (in_grad[0].ndim() == 3) {
      Shape<4> dshape = Shape4(out_grad[concat_enum::kOut].shape_[0],
        out_grad[concat_enum::kOut].shape_[1],
        out_grad[concat_enum::kOut].shape_[2], 1);
      grad = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_grad[concat_enum::kOut], dshape, s);
      for (int i = 0; i < size_; ++i) {
        dshape = Shape4(in_grad[i].shape_[0],
          in_grad[i].shape_[1], in_grad[i].shape_[2], 1);
        grad_in[i] = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
          in_grad[i], dshape, s);
      }
    } else {
      grad = mkl_experimental_direct_get<xpu, 4, DType>(out_grad[concat_enum::kOut], s);
      for (int i = 0; i < size_; ++i) {
        grad_in[i] = mkl_experimental_direct_get<xpu, 4, DType>(in_grad[i], s);
      }
    }

    int need_bwd = 0;
    for (size_t n = 0; n < num_concats_; n++) {
      need_bwd += req[n];
    }
    if (!need_bwd) {
      return;
    }

    dnnError_t e;
    void *concat_res[dnnResourceNumber];
    concat_res[dnnResourceSrc] = bwd_top_diff_->get_converted_prv(grad.dptr_, true,
      out_grad[concat_enum::kOut]);
    for (size_t i = 0; i < num_concats_; ++i) {
      concat_res[dnnResourceMultipleDst + i] = bwd_bottom_diff_[i]->get_output_ptr(
        grad_in[i].dptr_, bwd_bottom_diff_[i], in_grad[i]);
    }
    e = dnnExecute<DType>(concatBwd_, concat_res);
    CHECK_EQ(e, E_SUCCESS);
  }

 private:
  int size_;
  size_t dimension_;

  bool init_mkldnn_;

  dnnPrimitive_t concatFwd_;
  dnnPrimitive_t concatBwd_;
  std::shared_ptr<MKLData<DType> > fwd_top_data_;
  std::vector< std::shared_ptr<MKLData<DType> > > fwd_bottom_data_;
  std::shared_ptr<MKLData<DType> > bwd_top_diff_;
  std::vector< std::shared_ptr<MKLData<DType> > > bwd_bottom_diff_;


  size_t width_;
  size_t height_;
  size_t channels_;
  size_t num_;
  size_t num_concats_;
};  // class MKLConcatOp
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MKL_MKL_CONCAT_INL_H_
