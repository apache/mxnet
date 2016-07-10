/*!
 * Copyright (c) 2015 by Contributors
 * \file rnn-inl.h
 * \brief
 * \author Sebastian Bodenstein
*/
#ifndef MXNET_OPERATOR_RNN_INL_H_
#define MXNET_OPERATOR_RNN_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace rnn_enum {
  enum RNNOpInputs {kData, kWeight, kStateIn, kCellStateIn};
  enum RNNOpOutputs {kOut, kStateOut, kCellStateOut};
  enum RNNModeType {kRnnRelu, kRnnTanh, kLstm, kGru};  
  enum RNNDirectionType {kUnidirectional, kBidirectional};
  enum RNNOpResource {kTempSpace};
}

// A utility function to calculate input size

inline int rnn_single_param_size(int inputSize,
                                int hiddenSize, 
                                int mode){
  int size = hiddenSize * (hiddenSize + inputSize + 2);
  // Different RNN's have different num weights
  switch(mode)
  {
    case rnn_enum::kRnnRelu:
      size *= 1 ;
      break;
    case rnn_enum::kRnnTanh:
      size *= 1;
      break;
    case rnn_enum::kLstm:
      size *= 4;
      break;
    case rnn_enum::kGru:
      size *= 3;
      break;
  }
  return size;
}

inline int rnn_param_size(int layerNum, 
                          int inputSize,
                          int hiddenSize, 
                          int direction, 
                          int mode){
  // get size of first layer
  int size = rnn_single_param_size(inputSize, hiddenSize, mode);
  // get size of remaining layers
  if(direction == rnn_enum::kUnidirectional)
    size += (layerNum - 1) * rnn_single_param_size(hiddenSize, hiddenSize, mode);
  else // bidirectional case: input size increases by 2
    size += (layerNum - 1) * rnn_single_param_size(2 * hiddenSize, hiddenSize, mode);
  return size;
}

struct RNNParam : public dmlc::Parameter<RNNParam> {
  uint32_t state_size;
  uint32_t num_layers;
  uint64_t workspace;
  bool batch_first;
  int direction;
  int mode;

  DMLC_DECLARE_PARAMETER(RNNParam) {
    DMLC_DECLARE_FIELD(state_size)
    .describe("size of the state for each layer");

    DMLC_DECLARE_FIELD(num_layers)
    .describe("number of stacked layers");

    DMLC_DECLARE_FIELD(workspace).set_default(512).set_range(0, 8192)
    .describe("Tmp workspace for RNN (MB)");

    DMLC_DECLARE_FIELD(direction)
    .add_enum("unidirectional", rnn_enum::kUnidirectional)
    .add_enum("bidirectional", rnn_enum::kBidirectional)
    .describe("specifies the recurrence pattern");

    DMLC_DECLARE_FIELD(mode)
    .add_enum("rnn_relu", rnn_enum::kRnnRelu)
    .add_enum("rnn_tanh", rnn_enum::kRnnTanh)
    .add_enum("lstm", rnn_enum::kLstm)
    .add_enum("gru", rnn_enum::kGru)
    .describe("the type of RNN to compute");
  }
};

template<typename xpu, typename DType>
class RNNOp : public Operator {
 public:
  explicit RNNOp(RNNParam p) {
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(real_t);
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
//     CHECK_EQ(req[rnn_enum::kOut], kWriteTo);
  
//     CHECK_EQ(in_data.size(), expected);
//     CHECK_EQ(out_data.size(), 1);
//     Stream<xpu> *s = ctx.get_stream<xpu>();
//     Tensor<xpu, 4, DType> data = in_data[rnn_enum::kData].get<xpu, 4, DType>(s);
//     Tensor<xpu, 4, DType> out = out_data[rnn_enum::kOut].get<xpu, 4, DType>(s);
//     Shape<3> wmat_shape =
//         Shape3(param_.num_group,
//                data.shape_[1] / param_.num_group,
//                param_.num_filter / param_.num_group * param_.kernel[0] * param_.kernel[1]);
//     Tensor<xpu, 3, DType> wmat =
//         in_data[rnn_enum::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
// #if defined(__CUDACC__)
//     CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
//         << "Must init CuBLAS handle in stream";
// #endif
//     const index_t nbatch = data.size(0);
//     Tensor<xpu, 1, DType> workspace =
//         ctx.requested[rnn_enum::kTempSpace].get_space_typed<xpu, 1, DType>(
//             Shape1(this->InitTemp(out.shape_, data.shape_)), s);
//     for (index_t i = 0; i < nbatch; i += nstep_) {
//       const index_t step = std::min(nstep_, nbatch - i);
//       Tensor<xpu, 2, DType> temp_col = Tensor<xpu, 2, DType>(
//                                             workspace.dptr_,
//                                             Shape2(shape_colunit_[0],
//                                             shape_colunit_[1] * step), s);
//       Tensor<xpu, 3, DType> temp_dst = Tensor<xpu, 3, DType>(
//                                            workspace.dptr_ + temp_col.shape_.Size(),
//                                            Shape3(shape_dstunit_[0],
//                                            shape_dstunit_[1],
//                                            shape_dstunit_[2] * step), s);
//       temp_dst = reshape(swapaxis<1, 0>(data.Slice(i, i + step)), temp_dst.shape_);
//       if (param_.pad[0] == 0 && param_.pad[1] == 0) {
//         temp_col = unpack_patch2col(out.Slice(i, i + step),
//                                     param_.kernel[0],
//                                     param_.kernel[1],
//                                     param_.stride[0],
//                                     param_.stride[1],
//                                     1, 1);  // RNN only support dilate equals 1
//       } else {
//         temp_col = unpack_patch2col(pad(out.Slice(i, i + step),
//                                         param_.pad[0], param_.pad[1]),
//                                     param_.kernel[0],
//                                     param_.kernel[1],
//                                     param_.stride[0],
//                                     param_.stride[1],
//                                     1, 1);  // RNN only support dilate equals 1
//       }
//       const index_t gstride = temp_col.size(0) / param_.num_group;
//       for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
//         mshadow::Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid,
//                                               gstride * (gid + 1));
//         tmpc = dot(wmat[gid].T(), temp_dst[gid]);
//       }
//       if (param_.pad[0] == 0 && param_.pad[1] == 0) {
//         out.Slice(i, i + step) = pack_col2patch(temp_col,
//                                    out.Slice(i, i + step).shape_,
//                                    param_.kernel[0],
//                                    param_.kernel[1],
//                                    param_.stride[0],
//                                    1);  // RNN only support dilate equals 1
//       } else {
//         Shape<4> pshape = out.Slice(i, i + step).shape_;
//         pshape[2] += 2 * param_.pad[0];
//         pshape[3] += 2 * param_.pad[1];
//         out.Slice(i, i + step) = crop(pack_col2patch(temp_col,
//                                         pshape,
//                                         param_.kernel[0],
//                                         param_.kernel[1],
//                                         param_.stride[0],
//                                         1),  // RNN only support dilate equals 1
//                                         out[i][0].shape_);
//       }
//     }
//     if (!param_.no_bias) {
//       // add bias, broadcast bias to dim 1: channel
//       Tensor<xpu, 1, DType> bias = in_data[rnn_enum::kBias].get<xpu, 1, DType>(s);
//       out += broadcast<1>(bias, out.shape_);
//     }
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
    // TODO(bing): check the BLAS Handle, be careful
//     CHECK_EQ(out_grad.size(), 1);
//     size_t expected = param_.no_bias == 0 ? 3 : 2;
//     CHECK(in_data.size() == expected && in_grad.size() == expected);
//     CHECK_EQ(req.size(), expected);
//     CHECK_EQ(in_data[rnn_enum::kWeight].CheckContiguous(), true);
//     // get data
//     Stream<xpu> *s = ctx.get_stream<xpu>();
//     Tensor<xpu, 4, DType> data = in_data[rnn_enum::kData].get<xpu, 4, DType>(s);
//     Tensor<xpu, 4, DType> grad = out_grad[rnn_enum::kOut].get<xpu, 4, DType>(s);
//     Tensor<xpu, 4, DType> gdata = in_grad[rnn_enum::kData].get<xpu, 4, DType>(s);
//     Shape<3> wmat_shape =
//         Shape3(param_.num_group,
//                data.shape_[1] / param_.num_group,
//                param_.num_filter / param_.num_group * param_.kernel[0] * param_.kernel[1]);
//     Tensor<xpu, 3, DType> wmat =
//         in_data[rnn_enum::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
//     Tensor<xpu, 3, DType> gwmat =
//         in_grad[rnn_enum::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
// #if defined(__CUDACC__)
//     CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
//         << "Must init CuBLAS handle in stream";
// #endif
//     const index_t nbatch = data.size(0);
//     Tensor<xpu, 1, DType> workspace =
//         ctx.requested[rnn_enum::kTempSpace].get_space_typed<xpu, 1, DType>(
//             Shape1(this->InitTemp(grad.shape_, data.shape_)), s);
//     for (index_t i = 0; i < nbatch; i += nstep_) {
//       const index_t step = std::min(nstep_, nbatch - i);
//       Tensor<xpu, 2, DType> temp_col = Tensor<xpu, 2, DType>(
//                                            workspace.dptr_,
//                                            Shape2(shape_colunit_[0],
//                                            shape_colunit_[1] * step), s);
//       Tensor<xpu, 3, DType> temp_dst = Tensor<xpu, 3, DType>(
//                                            workspace.dptr_ + temp_col.shape_.Size(),
//                                            Shape3(shape_dstunit_[0],
//                                            shape_dstunit_[1],
//                                            shape_dstunit_[2] * step), s);
//       temp_dst = reshape(swapaxis<1, 0>(data.Slice(i, i + step)), temp_dst.shape_);
//       if (param_.pad[0] == 0 && param_.pad[1] == 0) {
//         temp_col = unpack_patch2col(grad.Slice(i, i + step),
//                                      param_.kernel[0],
//                                      param_.kernel[1],
//                                      param_.stride[0],
//                                      param_.stride[1],
//                                      1, 1);  // RNN only support dilate equals 1
//       } else {
//         temp_col = unpack_patch2col(pad(grad.Slice(i, i + step), param_.pad[0], param_.pad[1]),
//                                      param_.kernel[0],
//                                      param_.kernel[1],
//                                      param_.stride[0],
//                                      param_.stride[1],
//                                      1, 1);  // RNN only support dilate equals 1
//       }
//       const index_t gstride = temp_col.size(0) / param_.num_group;
//       for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
//         Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
//         if (i == 0) {
//           Tensor<xpu, 2, DType> tmp_gwmat = gwmat[gid];
//           Assign(tmp_gwmat, req[rnn_enum::kWeight], dot(temp_dst[gid], tmpc.T()));
//         } else {
//           gwmat[gid] += dot(temp_dst[gid], tmpc.T());
//         }
//       }
//       if (req[rnn_enum::kData] == kWriteTo || req[rnn_enum::kData] == kWriteInplace) {
//         for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
//           Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
//           temp_dst[gid] = dot(wmat[gid], tmpc);
//         }
//         gdata.Slice(i, i + step) = swapaxis<1, 0>(reshape(temp_dst,
//                                                     mshadow::Shape4(gdata.shape_[1],
//                                                     step,
//                                                     gdata.size(2),
//                                                     gdata.size(3))));
//       }
//     }
//     if (!param_.no_bias) {
//       Tensor<xpu, 1, DType> gbias = in_grad[rnn_enum::kBias].get<xpu, 1, DType>(s);
//       Assign(gbias, req[rnn_enum::kBias], sumall_except_dim<1>(grad));
//     }
  }

 private:
//   inline index_t InitTemp(const mshadow::Shape<4> &ishape,
//                           const mshadow::Shape<4> &oshape) {
//     const int ksize_y = param_.kernel[0];
//     const int ksize_x = param_.kernel[1];
//     shape_colunit_ = mshadow::Shape2(ishape[1] * ksize_y * ksize_x,
//                                      oshape[2] * oshape[3]);
//     shape_dstunit_ = mshadow::Shape3(param_.num_group,
//                                      oshape[1] / param_.num_group,
//                                      oshape[2] * oshape[3]);
//     // See convolution for workspace calculations
//     nstep_ = std::max(
//         std::min(
//             static_cast<index_t>(
//                 param_.workspace / (shape_colunit_.Size() + shape_dstunit_.Size())),
//             ishape[0]),
//         1U);

//     mshadow::Shape<2> scol = mshadow::Shape2(shape_colunit_[0],
//                                              shape_colunit_[1] * nstep_);
//     mshadow::Shape<3> sdst = mshadow::Shape3(shape_dstunit_[0],
//                                              shape_dstunit_[1],
//                                              shape_dstunit_[2] * nstep_);
//     index_t required_size = scol.Size() + sdst.Size();
//     CHECK_GE(param_.workspace, required_size)
//       << "\nMinimum workspace size: " << required_size * sizeof(DType) << " Bytes\n"
//       << "Given: " << param_.workspace * sizeof(DType);
//     return required_size;
//   }

 private:
  RNNParam param_;
};  // class RNNOp




template<typename xpu>
Operator* CreateOp(RNNParam param, int dtype);

#if DMLC_USE_CXX11
class RNNProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (param_.mode == rnn_enum::kLstm) {
      return {"data", "weight", "state", "cell_state"};
    } else {
      return {"data", "weight", "state"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (param_.mode == rnn_enum::kLstm) {
      CHECK_EQ(in_shape->size(), 4) << "Input:[data, weight, state, cell_state]";
    } else {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, state]";
    }
    const TShape &dshape = (*in_shape)[rnn_enum::kData];
    if (dshape.ndim() ==  0) return false;
    CHECK_EQ(dshape.ndim(), 3) \
        << "Input data should be rank-3 tensor of dim (seqLength, batch, inputDim).";
    // Infer hidden state + cell state
    int batchSize = dshape[0];
    int inputSize = dshape[2];
    int numDirections = 1;
    if(param_.direction == rnn_enum::kBidirectional){
      numDirections = 2;
    }
    int total_layers = numDirections * param_.num_layers; // double for bidirectional
    SHAPE_ASSIGN_CHECK(*in_shape,
                       rnn_enum::kStateIn,
                       Shape3(total_layers, batchSize, param_.state_size));
    if (param_.mode == rnn_enum::kLstm){
      SHAPE_ASSIGN_CHECK(*in_shape,
                        rnn_enum::kCellStateIn,
                        Shape3(total_layers, batchSize, param_.state_size));
    }
    // infer weight size
    int weight_size = rnn_param_size(param_.num_layers, 
                                    inputSize, 
                                    param_.state_size, 
                                    param_.direction, 
                                    param_.mode);
    SHAPE_ASSIGN_CHECK(*in_shape, rnn_enum::kWeight, Shape1(weight_size));
    // infer output size
    TShape oshape = dshape;
    oshape[3] = numDirections * param_.state_size;
    // infer output state size   
    TShape outStateShape = dshape;
    outStateShape[0] = total_layers;
    outStateShape[1] = batchSize;
    outStateShape[2] = param_.state_size;

    out_shape->clear();   
    out_shape->push_back(oshape);
    out_shape->push_back(outStateShape);
    if (param_.mode == rnn_enum::kLstm) 
      out_shape->push_back(outStateShape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    out_type->push_back(dtype);
    if (param_.mode == rnn_enum::kLstm) 
      out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new RNNProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "RNN";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (param_.mode == rnn_enum::kLstm)
      return {out_grad[rnn_enum::kOut], in_data[rnn_enum::kData], in_data[rnn_enum::kWeight]};
    else
      return {out_grad[rnn_enum::kOut], in_data[rnn_enum::kData], in_data[rnn_enum::kWeight]};
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  RNNParam param_;
};  // class RNNProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_RNN_INL_H_
