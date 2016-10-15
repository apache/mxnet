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
* \file mkl_elementwise-inl.h
* \brief
* \author lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKL_ELEMENTWISE_INL_H_
#define MXNET_OPERATOR_MKL_MKL_ELEMENTWISE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../operator_common.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {
enum EltwiseParameter_EltwiseOp {
  EltwiseParameter_EltwiseOp_PROD = 0,
  EltwiseParameter_EltwiseOp_SUM = 1,
  EltwiseParameter_EltwiseOp_MAX = 2
};

template<typename xpu, typename DType>
class MKLElementWiseOp : public Operator {
 public:
  std::string getName() {
    return "MKLElementWiseOp";
  }
  explicit MKLElementWiseOp(ElementWiseSumParam param,
                            EltwiseParameter_EltwiseOp op)
    : size_(param.num_args), op_(op) {
    fwd_top_data = MKLData<DType>::create();
  }
  virtual ~MKLElementWiseOp() {
    dnnDelete<DType>(sumPrimitive);
  }

 private:
  void LayerSetUp(const std::vector<mshadow::Tensor<xpu, 4, DType> > &data,
                  const mshadow::Tensor<xpu, 4, DType> &out,
                  int data_shape_size) {
    coeffs_ = std::vector<DType>(data.size(), 1);
    // Whether to use an asymptotically slower (for >2 inputs) but stabler method
    // of computing the gradient for the PROD operation. (No effect for SUM op.)
    stable_prod_grad_ = 1;

    num_bottoms = size_;
    size_t dim_src = data_shape_size;
    size_t *sizes_src = new size_t[dim_src];
    size_t *strides_src = new size_t[dim_src];
    for (size_t d = 0; d < dim_src; ++d) {
      sizes_src[d] = data[0].shape_[dim_src - d - 1];
      strides_src[d] = (d == 0) ? 1 : strides_src[d - 1] * sizes_src[d - 1];
    }

    for (size_t i = 0; i < num_bottoms; ++i) {
      fwd_bottom_data_.push_back(MKLData<DType>::create());
      bwd_bottom_diff.push_back(MKLData<DType>::create());
      CHECK_EQ(dim_src, data_shape_size);
      fwd_bottom_data_[i]->create_user_layout(dim_src, sizes_src, strides_src);
      bwd_bottom_diff[i]->create_user_layout(dim_src, sizes_src, strides_src);
    }

    fwd_top_data->create_user_layout(dim_src, sizes_src, strides_src);
    free(sizes_src);
    free(strides_src);
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
    if (req[elemsum::kOut] == kNullOp) return;

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> out;
    std::vector<Tensor<xpu, 4, DType> > data(size_);
    if (in_data[0].ndim() == 1) {
      for (int i = 0; i < size_; ++i) {
        Shape<4> dshape = Shape4(in_data[i].shape_[0], 1, 1, 1);
          data[i] = in_data[i].get_with_shape<xpu, 4, DType>(dshape, s);
      }
      Shape<4> dshape = Shape4(out_data[elemsum::kOut].shape_[0], 1, 1, 1);
      out = out_data[elemsum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
    } else if (in_data[0].ndim() == 2) {
      for (int i = 0; i < size_; ++i) {
        Shape<4> dshape = Shape4(in_data[i].shape_[0],
                                 in_data[i].shape_[1], 1, 1);
        data[i] = in_data[i].get_with_shape<xpu, 4, DType>(dshape, s);
      }
      Shape<4> dshape = Shape4(out_data[elemsum::kOut].shape_[0],
                               out_data[elemsum::kOut].shape_[1], 1, 1);
      out = out_data[elemsum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
    } else if (in_data[0].ndim() == 3) {
      for (int i = 0; i < size_; ++i) {
        Shape<4> dshape = Shape4(in_data[i].shape_[0],
                                 in_data[i].shape_[1], in_data[i].shape_[2], 1);
        data[i] = in_data[i].get_with_shape<xpu, 4, DType>(dshape, s);
      }
      Shape<4> dshape = Shape4(out_data[elemsum::kOut].shape_[0],
                               out_data[elemsum::kOut].shape_[1],
                               out_data[elemsum::kOut].shape_[2], 1);
      out = out_data[elemsum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
      } else {
        out = out_data[elemsum::kOut].get<xpu, 4, DType>(s);
        for (int i = 0; i < size_; ++i) {
          data[i] = in_data[i].get<xpu, 4, DType>(s);
        }
      }
    if (!init_mkldnn_) {
      init_mkldnn_ = true;
      LayerSetUp(data, out, 4);  // in_data[concat_enum::kData0].ndim()=>always 4
    }

    dnnError_t e;
    std::vector<void*> bottom_data;
    for (size_t i = 0; i < num_bottoms; i++) {
      bottom_data.push_back(reinterpret_cast<void *>(in_data[i].dptr_));
    }

    if (sumPrimitive == NULL) {
      e = dnnSumCreate<DType>(&sumPrimitive, NULL, num_bottoms,
                              fwd_top_data->layout_usr, &coeffs_[0]);
      CHECK_EQ(e, E_SUCCESS);
    }


    switch (op_) {
    case EltwiseParameter_EltwiseOp_SUM:
      void *eltwise_res[dnnResourceNumber];
      for (int i = 0; i < num_bottoms; ++i) {
        if (fwd_bottom_data_[i]->conversion_needed()) {
          eltwise_res[dnnResourceMultipleSrc + i] =
          fwd_bottom_data_[i]->get_converted_prv(data[i].dptr_, false);
        } else {
          eltwise_res[dnnResourceMultipleSrc + i] =
            reinterpret_cast<void *>(bottom_data[i]);
        }
      }

      if (fwd_top_data->conversion_needed()) {
        eltwise_res[dnnResourceDst] =
          reinterpret_cast<void*>(fwd_top_data->prv_ptr());
      } else {
        eltwise_res[dnnResourceDst] =
          reinterpret_cast<void*>(const_cast<DType*>(out.dptr_));
      }

      e = dnnExecute<DType>(sumPrimitive, eltwise_res);
      CHECK_EQ(e, E_SUCCESS);

      if (fwd_top_data->conversion_needed()) {
        fwd_top_data->convert_from_prv(out.dptr_);
      }
      break;

    case EltwiseParameter_EltwiseOp_PROD:
    case EltwiseParameter_EltwiseOp_MAX:
      LOG(FATAL) << "Unsupported elementwise operation.";
    default:
      LOG(FATAL) << "Unknown elementwise operation.";
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), static_cast<size_t>(size_));
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> ograd;
    std::vector<Tensor<xpu, 4, DType> > igrad(size_);
    if (in_grad[0].ndim() == 1) {
      Shape<4> dshape = Shape4(out_grad[elemsum::kOut].shape_[0], 1, 1, 1);
      ograd = out_grad[elemsum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);

      for (int i = 0; i < size_; ++i) {
        dshape = Shape4(in_grad[i].shape_[0], 1, 1, 1);
        igrad[i] = in_grad[i].get_with_shape<xpu, 4, DType>(dshape, s);
      }
    } else if (in_grad[0].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[elemsum::kOut].shape_[0],
                               out_grad[elemsum::kOut].shape_[1], 1, 1);
      ograd = out_grad[elemsum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);

      for (int i = 0; i < size_; ++i) {
        dshape = Shape4(in_grad[i].shape_[0], in_grad[i].shape_[1], 1, 1);
                        igrad[i] = in_grad[i].get_with_shape<xpu, 4, DType>(dshape, s);
      }
    } else if (in_grad[0].ndim() == 3) {
      Shape<4> dshape = Shape4(out_grad[elemsum::kOut].shape_[0],
                               out_grad[elemsum::kOut].shape_[1],
                               out_grad[elemsum::kOut].shape_[2], 1);
      ograd = out_grad[elemsum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
      for (int i = 0; i < size_; ++i) {
        dshape = Shape4(in_grad[i].shape_[0], in_grad[i].shape_[1], in_grad[i].shape_[2], 1);
        igrad[i] = in_grad[i].get_with_shape<xpu, 4, DType>(dshape, s);
      }
    } else {
      ograd = out_grad[elemsum::kOut].get<xpu, 4, DType>(s);
      for (int i = 0; i < size_; ++i) {
        if (req[i] == kNullOp || req[i] == kWriteInplace) continue;
        igrad[i] = in_grad[i].get<xpu, 4, DType>(s);
      }
    }
    const DType* top_diff = NULL;
    int count = 0;
    // If there is no diff in prv layout
    // then we are given cpu layout
    // and we will produce bottom at cpu layout as well
    if (top_diff == NULL) {
      top_diff = ograd.dptr_;
      count = ograd.MSize();
    }
    DType* bottom_diff = NULL;
    for (int i = 0; i < size_; ++i) {
      if (req[i] == kNullOp || req[i] == kWriteInplace) continue;
      switch (op_) {
        case EltwiseParameter_EltwiseOp_SUM:
          CHECK_EQ(coeffs_[i], DType(1)) << "Not supported yet";
          bottom_diff = igrad[i].dptr_;
          memcpy(reinterpret_cast<void*>(bottom_diff),
            reinterpret_cast<const void*>(top_diff), sizeof(DType) * count);
          break;
        case EltwiseParameter_EltwiseOp_MAX:
        case EltwiseParameter_EltwiseOp_PROD:
          LOG(FATAL) << "Unsupported elementwise operation.";
        default:
          LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
  inline void Save(dmlc::JSONWriter *writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("size_", size_);
    writer->EndObject();
  }
  inline void Load(dmlc::JSONReader *reader) {
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("size_", &size_);
    helper.ReadAllFields(reader);
  }

 private:
  int size_;
  EltwiseParameter_EltwiseOp op_;
  std::shared_ptr<MKLData<DType> > fwd_top_data;

  std::vector< std::shared_ptr<MKLData<DType> > > fwd_bottom_data_;

  std::vector< std::shared_ptr<MKLData<DType> > > bwd_bottom_diff;
  dnnPrimitive_t sumPrimitive = NULL;
  dnnPrimitive_t convertPrimitive = NULL;


  std::vector<DType> coeffs_;
  DType *max_idx_data;
  size_t num_bottoms;

  bool stable_prod_grad_;
  bool init_mkldnn_ = false;
};  // class ElementWiseSumOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKL_ELEMENTWISE_INL_H_
