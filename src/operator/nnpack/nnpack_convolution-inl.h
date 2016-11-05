/*!
 * Copyright (c) 2016 by Contributors
 * \file nnpack_convolution-inl.h
 * \brief
 * \author Carwin
*/
#ifndef MXNET_OPERATOR_NNPACK_NNPACK_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_NNPACK_NNPACK_CONVOLUTION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../convolution-inl.h"
#include "nnpack.h"

namespace mxnet {
namespace op {

class NNPACKInitialize {
 public:
  pthreadpool_t threadpool;

 public:
  NNPACKInitialize() {
    nnp_status status = nnp_initialize();
    if (nnp_status_success != status) {
      LOG(FATAL) << "nnp_initialize failed status=" << status;
    }
    int num_threads = MXNET_USE_NNPACK_NUM_THREADS;
    this->threadpool = pthreadpool_create(num_threads);
  }
  virtual ~NNPACKInitialize() {
    nnp_status status = nnp_deinitialize();
    if (nnp_status_success != status) {
      LOG(FATAL) << "nnp_deinitialize failed status=" << status;
    }
    pthreadpool_destroy(threadpool);
  }
};

static NNPACKInitialize nnpackinitialize;

template <typename xpu, typename DType>
class NNPACKConvolutionOp : public ConvolutionOp<xpu, DType> {
 private:
  ConvolutionParam param_;

 public:
  explicit NNPACKConvolutionOp(ConvolutionParam p)
      : ConvolutionOp<xpu, DType>(p) {
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
    Tensor<xpu, 4, DType> data = in_data[conv::kData].get<xpu, 4, DType>(s);
    Shape<3> wmat_shape =
        Shape3(param_.num_group, param_.num_filter / param_.num_group,
               data.shape_[1] / param_.num_group * param_.kernel[0] *
                   param_.kernel[1]);
    Tensor<xpu, 3, DType> wmat =
        in_data[conv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
    Tensor<xpu, 4, DType> out = out_data[conv::kOut].get<xpu, 4, DType>(s);

    // nnp_convolution_inference optimize for batch_size==1
    // when W or H less than 16, ConvolutionOp fast than nnpack's convolution
    if ((data.shape_[0] != 1) || (data.shape_[2] < 16) ||
        (data.shape_[3] < 16)) {
      ConvolutionOp<xpu, DType>::Forward(ctx, in_data, req, out_data, aux_args);
    } else {
      nnp_size input_size = {data.shape_[3], data.shape_[2]};
      nnp_padding input_padding = {param_.pad[0], param_.pad[1], param_.pad[0],
                                   param_.pad[1]};
      nnp_size kernel_size = {param_.kernel[1], param_.kernel[0]};
      nnp_size output_subsampling = {param_.stride[1], param_.stride[0]};
      Tensor<xpu, 1, DType> bias = in_data[conv::kBias].get<xpu, 1, DType>(s);

      nnp_convolution_algorithm algorithm = nnp_convolution_algorithm_auto;
      if ((data.shape_[2] < 32) || (data.shape_[3] < 32)) {
        algorithm = nnp_convolution_algorithm_implicit_gemm;
      }

      nnp_status status = nnp_convolution_inference(
          algorithm,           // enum nnp_convolution_algorithm algorithm,
          nnp_convolution_transform_strategy_tuple_based,
          data.shape_[1],               // size_t input_channels,
          param_.num_filter,            // size_t output_channels,
          input_size,                   // struct nnp_size input_size,
          input_padding,                // struct nnp_padding input_padding,
          kernel_size,                  // struct nnp_size kernel_size,
          output_subsampling,           // struct nnp_size output_subsampling,
          data.dptr_,                   // const float input[],
          wmat.dptr_,                   // const float kernel[],
          bias.dptr_,                   // const float bias[],
          out.dptr_,                    // float output[],
          nnpackinitialize.threadpool,  // pthreadpool_t threadpool,
          nullptr);
      if (nnp_status_success != status) {
        LOG(FATAL) << "nnp_convolution_inference failed status=" << status;
      }
    }
  }
};  // class NNPACKConvolutionOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NNPACK_NNPACK_CONVOLUTION_INL_H_
