/*!
 * Copyright (c) 2016 by Contributors
 * \file nnpack_pooling-inl.h
 * \brief
 * \author Wei Wu
*/
#ifndef MXNET_OPERATOR_NNPACK_NNPACK_POOLING_INL_H_
#define MXNET_OPERATOR_NNPACK_NNPACK_POOLING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../pooling-inl.h"
#include "nnpack.h"
#include "nnpack_util.h"

namespace mxnet {
namespace op {

template <typename xpu, typename DType>
class NNPACKPoolingOp : public PoolingOp<xpu, DType> {
 private:
  PoolingParam param_;

 public:
  explicit NNPACKPoolingOp(PoolingParam p)
      : PoolingOp<xpu, DType>(p) {
    this->param_ = p;
  }

 public:
  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data = in_data[pool_enum::kData].get<xpu, 4, DType>(s);
    const size_t batch_size = data.shape_[0];
    const size_t input_c = data.shape_[1];
    const size_t input_h = data.shape_[2];
    const size_t input_w = data.shape_[3];
    Tensor<xpu, 4, DType> out = out_data[pool_enum::kOut].get<xpu, 4, DType>(s);
    nnp_size input_size = {input_w, input_h};
    nnp_padding input_padding = {param_.pad[0], param_.pad[1], param_.pad[0],
                                 param_.pad[1]};
    nnp_size kernel_size = {param_.kernel[1], param_.kernel[0]};
    nnp_size output_subsampling = {param_.stride[1], param_.stride[0]};
    nnp_status status = nnp_max_pooling_output(
      batch_size,                    // size_t batch size of input tensor
      input_c,                       // size_t input_channels,
      input_size,                    // struct nnp_size input_size,
      input_padding,                 // struct nnp_padding input_padding,
      kernel_size,                   // struct nnp_size kernel_size,
      output_subsampling,            // struct nnp_size output_subsampling,
      data.dptr_,                    // const float input[],
      out.dptr_,                     // float output[],
      nnpackinitialize.threadpool);  // pthreadpool_t threadpool,
    if (nnp_status_success != status) {
      LOG(FATAL) << "nnpack max pooling feedforward failed status=" << status;
    }
  }
};  // class NNPACKPoolingOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NNPACK_NNPACK_POOLING_INL_H_
