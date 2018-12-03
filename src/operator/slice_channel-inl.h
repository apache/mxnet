/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file slice_channel-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_SLICE_CHANNEL_INL_H_
#define MXNET_OPERATOR_SLICE_CHANNEL_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./channel_op_common.h"
#include "./mxnet_op.h"

namespace mxnet {
namespace op {

namespace slice_enum {
enum SliceChannelOpInputs {kData};
}  // namespace slice_enum

struct SliceChannelParam : public dmlc::Parameter<SliceChannelParam> {
  dmlc::optional<int> num_outputs;
  dmlc::optional<TShape> indices;
  int axis;
  bool squeeze_axis;
  DMLC_DECLARE_PARAMETER(SliceChannelParam) {
    DMLC_DECLARE_FIELD(num_outputs).set_default(dmlc::optional<int>())
    .describe("Number of splits. Note that this should evenly divide the length of the `axis`.");
    DMLC_DECLARE_FIELD(indices).set_default(dmlc::optional<TShape>())
    .describe("Indices of splits. The elements should denote the boundaries of at which split"
              " is performed along the `axis`.");
    DMLC_DECLARE_FIELD(axis).set_default(1)
    .describe("Axis along which to split.");
    DMLC_DECLARE_FIELD(squeeze_axis).set_default(0)
    .describe("If true, Removes the axis with length 1 from the shapes of the output arrays."
              " **Note** that setting `squeeze_axis` to ``true`` removes axis with length 1"
              " only along the `axis` which it is split."
              " Also `squeeze_axis` can be set to ``true``"
              " only if ``input.shape[axis] == num_outputs``.");
  }
};  // struct SliceChannelParam

struct SplitKernel {
  /*!
   * \brief Map function for split operator indices option
   * \param i              global thread id
   * \param in_data        ptr to input buffer
   * \param out_data       ptr to ptr of outputs buffer
   * \param indices        ptr to indices buffer
   * \param num_sections   # of sections after split
   * \param axis_size      size of axis to be splitted on
   * \param trailing_size  step size within the data buffer of the axis to be splitted on
   */
  template<typename DType>
  static MSHADOW_XINLINE void Map(size_t i,
                                  const DType *in_data, DType** out_data, const size_t* indices,
                                  const size_t num_sections, const size_t axis_size,
                                  const size_t trailing_size) {
    size_t idx = i / trailing_size % axis_size;
    size_t target = 0;
    for (size_t section = 0; section < num_sections; target = section++) {
      if (indices[section] > idx) {
        break;
      }
    }
    DType* target_data = out_data[target];
    const size_t mid_idx = idx - indices[target];
    const size_t head_idx = i / (trailing_size * axis_size);
    const size_t tail_idx = i % trailing_size;
    const size_t section_size = indices[target + 1] - indices[target];
    const size_t target_idx =
      head_idx * trailing_size * section_size + mid_idx * trailing_size + tail_idx;
    target_data[target_idx] = in_data[i];
  }
};

struct ConcatenateKernel {
  /*!
   * \brief Map function for split operator indices option
   * \param i              global thread id
   * \param out_grad       ptr to ptr of out grads buffer
   * \param in_grad        ptr to input grad buffer
   * \param indices        ptr to indices buffer
   * \param num_sections   # of sections after split
   * \param axis_size      size of axis to be splitted on
   * \param trailing_size  step size within the data buffer of the axis to be splitted on
   */
  template<typename DType>
  static MSHADOW_XINLINE void Map(size_t i,
                                  DType** out_grad, DType* in_grad, const size_t* indices,
                                  const size_t num_sections, const size_t axis_size,
                                  const size_t trailing_size) {
    size_t idx = i / trailing_size % axis_size;
    size_t src = 0;
    for (size_t section = 0; section < num_sections; src = section++) {
      if (indices[section] > idx) {
        break;
      }
    }
    DType* src_grad = out_grad[src];
    const size_t mid_idx = idx - indices[src];
    const size_t head_idx = i / (trailing_size * axis_size);
    const size_t tail_idx = i % trailing_size;
    const size_t section_size = indices[src + 1] - indices[src];
    const size_t src_idx =
      head_idx * trailing_size * section_size + mid_idx * trailing_size + tail_idx;
    in_grad[i] = src_grad[src_idx];
  }
};

template<typename xpu, typename DType>
class SliceChannelOp : public Operator {
 public:
  explicit SliceChannelOp(SliceChannelParam param)
    : param_(param), axis_(param.axis) {
    CHECK(param.indices.has_value() ^ param.num_outputs.has_value())
      << "Only one of indices and num_outputs can have value";
    size_ = (param.num_outputs.has_value()) ?
            param.num_outputs.value() :
            param.indices.value().ndim() + 1;
    equal_split = param.num_outputs.has_value();
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), static_cast<size_t>(size_));
    Stream<xpu> *s = ctx.get_stream<xpu>();
    size_t leading = 1, trailing = 1;
    int real_axis = axis_;
    if (real_axis < 0) {
      real_axis += in_data[slice_enum::kData].ndim();
    }
    CHECK_LT(real_axis, in_data[slice_enum::kData].ndim());
    size_t mid = in_data[slice_enum::kData].shape_[real_axis];
    for (int i = 0; i < real_axis; ++i) {
      leading *= in_data[slice_enum::kData].shape_[i];
    }
    for (int i = real_axis + 1; i < in_data[slice_enum::kData].ndim(); ++i) {
      trailing *= in_data[slice_enum::kData].shape_[i];
    }
    Shape<3> dshape = Shape3(leading, mid, trailing);
    if (equal_split) {
      Shape<3> slice_shape = Shape3(leading, mid / size_, trailing);
      Tensor<xpu, 3, DType> data = in_data[slice_enum::kData].get_with_shape<xpu, 3, DType>(
          dshape, s);
      std::vector<Tensor<xpu, 3, DType> > outputs(size_);
      for (int i = 0; i < size_; ++i) {
        outputs[i] = out_data[i].get_with_shape<xpu, 3, DType>(slice_shape, s);
      }
      Split(data, &outputs, 1, req);
    } else {
      using namespace mxnet_op;
      const TBlob& input_data = in_data[slice_enum::kData];
      size_t workspace_size = 0;
      std::vector<size_t> indices;
      indices.push_back(0);
      for (const auto& section : param_.indices.value()) {
        indices.push_back(section);
      }
      indices.push_back(mid);
      workspace_size += indices.size() * sizeof(size_t);
      std::vector<DType*> output_data;
      for (const TBlob& data : out_data) {
        output_data.push_back(data.dptr<DType>());
      }
      workspace_size += output_data.size() * sizeof(DType*);
      Tensor<xpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
      Tensor<cpu, 1, size_t> indices_cpu_tensor(indices.data(), Shape1(indices.size()));
      Tensor<xpu, 1, size_t> indices_xpu_tensor(
        reinterpret_cast<size_t*>(workspace.dptr_), Shape1(indices.size()));
      Tensor<cpu, 1, DType*> ptrs_cpu_tensor(output_data.data(), Shape1(output_data.size()));
      Tensor<xpu, 1, DType*> ptrs_xpu_tensor(
        reinterpret_cast<DType**>(workspace.dptr_ + indices.size() * sizeof(size_t)),
        Shape1(output_data.size()));
      mshadow::Copy(indices_xpu_tensor, indices_cpu_tensor, s);
      mshadow::Copy(ptrs_xpu_tensor, ptrs_cpu_tensor, s);
      Kernel<SplitKernel, xpu>::Launch(
        s, input_data.Size(), input_data.dptr<DType>(), ptrs_xpu_tensor.dptr_,
        indices_xpu_tensor.dptr_, indices.size() - 1, mid, trailing);
    }
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
    CHECK_EQ(out_grad.size(), static_cast<size_t>(size_));
    CHECK_EQ(in_grad.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    size_t leading = 1, trailing = 1;
    int real_axis = axis_;
    if (real_axis < 0) {
        real_axis += in_grad[slice_enum::kData].ndim();
    }
    CHECK_LT(real_axis, in_grad[slice_enum::kData].ndim());
    size_t mid = in_grad[slice_enum::kData].shape_[real_axis];
    for (int i = 0; i < real_axis; ++i) {
      leading *= in_grad[slice_enum::kData].shape_[i];
    }
    for (int i = real_axis + 1; i < in_grad[slice_enum::kData].ndim(); ++i) {
      trailing *= in_grad[slice_enum::kData].shape_[i];
    }
    Shape<3> dshape = Shape3(leading, mid, trailing);
    if (equal_split) {
      Shape<3> slice_shape = Shape3(leading, mid / size_, trailing);
      Tensor<xpu, 3, DType> grad = in_grad[slice_enum::kData].get_with_shape<xpu, 3, DType>(
          dshape, s);
      std::vector<Tensor<xpu, 3, DType> > grad_out(size_);
      for (int i = 0; i < size_; ++i) {
        grad_out[i] = out_grad[i].get_with_shape<xpu, 3, DType>(slice_shape, s);
      }
      Concatenate(grad_out, &grad, 1, req[slice_enum::kData]);
    } else {
      CHECK(param_.indices.has_value())
        << "indices should have value!";
      using namespace mxnet_op;
      TBlob input_grad = in_grad[slice_enum::kData];
      size_t workspace_size = 0;
      std::vector<size_t> indices;
      indices.push_back(0);
      for (const auto& section : param_.indices.value()) {
        indices.push_back(section);
      }
      indices.push_back(mid);
      workspace_size += indices.size() * sizeof(size_t);
      std::vector<DType*> out_grads;
      for (const TBlob& output_grad : out_grad) {
        out_grads.push_back(output_grad.dptr<DType>());
      }
      workspace_size += out_grads.size() * sizeof(DType*);
      Tensor<xpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
      Tensor<cpu, 1, size_t> indices_cpu_tensor(indices.data(), Shape1(indices.size()));
      Tensor<xpu, 1, size_t> indices_xpu_tensor(
        reinterpret_cast<size_t*>(workspace.dptr_), Shape1(indices.size()));
      Tensor<cpu, 1, DType*> ptrs_cpu_tensor(out_grads.data(), Shape1(out_grads.size()));
      Tensor<xpu, 1, DType*> ptrs_xpu_tensor(
        reinterpret_cast<DType**>(workspace.dptr_ + indices.size() * sizeof(size_t)),
        Shape1(out_grads.size()));
      mshadow::Copy(indices_xpu_tensor, indices_cpu_tensor, s);
      mshadow::Copy(ptrs_xpu_tensor, ptrs_cpu_tensor, s);
      Kernel<ConcatenateKernel, xpu>::Launch(
        s, input_grad.Size(), ptrs_xpu_tensor.dptr_, input_grad.dptr<DType>(),
        indices_xpu_tensor.dptr_, indices.size() - 1, mid, trailing);
    }
  }

 private:
  SliceChannelParam param_;
  int size_;
  int axis_;
  bool equal_split;
};  // class SliceChannelOp


template<typename xpu>
Operator *CreateOp(SliceChannelParam param, int dtype);


#if DMLC_USE_CXX11
class SliceChannelProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListOutputs() const override {
    std::vector<std::string> ret;
    CHECK(param_.num_outputs.has_value() ^ param_.indices.has_value())
      << "Only one of num_outputs and indices should have value";
    int num_outputs = (param_.num_outputs.has_value()) ?
                      param_.num_outputs.value() :
                      param_.indices.value().ndim() + 1;
    CHECK_NE(num_outputs, -1);
    for (int i = 0; i < num_outputs; ++i) {
      std::ostringstream os;
      os << "output" << i;
      ret.push_back(os.str());
    }
    return ret;
  }

  int NumOutputs() const override {
    CHECK(param_.num_outputs.has_value() ^ param_.indices.has_value())
      << "Only one of indices and num_outputs can have value";
    return (param_.num_outputs.has_value()) ?
           param_.num_outputs.value() :
           param_.indices.value().ndim() + 1;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    out_type->clear();
    CHECK(param_.num_outputs.has_value() ^ param_.indices.has_value())
      << "Only one of num_outputs and indices should have value";
    int num_outputs = (param_.num_outputs.has_value()) ?
                      param_.num_outputs.value() :
                      param_.indices.value().ndim() + 1;
    out_type->reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      out_type->push_back(dtype);
    }
    aux_type->clear();
    return true;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1U);
    TShape dshape = in_shape->at(slice_enum::kData);
    TShape ishape = in_shape->at(slice_enum::kData);
    if (dshape.ndim() == 0) return false;
    if (param_.axis >= 0) {
      CHECK_LT(static_cast<size_t>(param_.axis), dshape.ndim());
    } else {
      CHECK_LT(param_.axis + dshape.ndim(), dshape.ndim());
    }
    int real_axis = param_.axis;
    if (real_axis < 0) {
      real_axis += dshape.ndim();
    }
    CHECK(param_.num_outputs.has_value() ^ param_.indices.has_value())
      << "Only one of num_outputs and indices should have value";
    int num_outputs = (param_.num_outputs.has_value()) ?
                      param_.num_outputs.value() :
                      param_.indices.value().ndim() + 1;
    if (param_.num_outputs.has_value()) {
      CHECK_EQ(dshape[real_axis] % num_outputs, 0U)
        << "You are trying to split the " << real_axis
        << "-th axis of input tensor with shape " << dshape
        << " into num_outputs=" << num_outputs
        << " evenly sized chunks, but this is not possible because "
        << num_outputs << " does not evenly divide "
        << dshape[real_axis];
      if (param_.squeeze_axis && ishape[real_axis] != 0) {
        CHECK_EQ(ishape[real_axis], static_cast<size_t>(num_outputs))
          << "If squeeze axis is True, the size of the sliced axis must be the same as num_outputs."
          << " Input shape=" << ishape << ", axis=" << real_axis
          << ", num_outputs=" << num_outputs << ".";
      }
      dshape[real_axis] /= num_outputs;
      if (param_.squeeze_axis && (dshape[real_axis] == 1 || ishape[real_axis] == 0)) {
        for (int d = real_axis; d < static_cast<int>(dshape.ndim()) - 1; ++d) {
          dshape[d] = dshape[d+1];
        }
        dshape = TShape(&dshape[0], &dshape[dshape.ndim()-1]);
      }
      CHECK_EQ(static_cast<int>((*out_shape).size()), num_outputs)
        << "Size of output shape mismatch!";
      for (int i = 0; i < num_outputs; ++i) {
        SHAPE_ASSIGN_CHECK(*out_shape, i, dshape);
        // Perform incomplete shape inference.
        // We can back-calculate the inshape based on the out_shape.
        TShape back_calculate_dshape = ishape;
        if (param_.squeeze_axis && (dshape.ndim() == ishape.ndim() - 1)) {
          for (int d = 0; d < real_axis; ++d) {
            back_calculate_dshape[d] = (*out_shape)[i][d];
          }
          back_calculate_dshape[real_axis] = num_outputs;
          for (int d = real_axis + 1; d < static_cast<int>(ishape.ndim()); ++d) {
            back_calculate_dshape[d] = (*out_shape)[i][d - 1];
          }
        } else {
          for (int d = 0; d < static_cast<int>(ishape.ndim()); ++d) {
            back_calculate_dshape[d] = (*out_shape)[i][d];
            if (d == real_axis) {
              back_calculate_dshape[d] *= num_outputs;
            }
          }
        }
        SHAPE_ASSIGN_CHECK(*in_shape, slice_enum::kData, back_calculate_dshape);
      }
    } else if (param_.indices.has_value()) {
      CHECK(!param_.squeeze_axis)
        << "squeeze_axis not implemented for indices option";
      const TShape& indices = param_.indices.value();
      for (int i = 0; i < num_outputs; ++i) {
        int start = (i == 0) ? 0 : indices[i-1];
        int end = (i == num_outputs - 1) ? ishape[real_axis] : indices[i];
        CHECK(start < end)
          << "start " << start << " is not less than end " << end << "for subarray " << i;
        CHECK(end <= ishape[real_axis])
          << "end " << end << " is no less than the size of the axis " << ishape[real_axis];
        dshape[real_axis] = (end - start);
        SHAPE_ASSIGN_CHECK(*out_shape, i, dshape);
      }
      TShape back_calculate_dshape = ishape;
      back_calculate_dshape[real_axis] = 0;
      for (int d = 0; d < static_cast<int>(ishape.ndim()); ++d) {
        if (d == real_axis) {
          for (int i = 0; i < num_outputs; ++i) {
            back_calculate_dshape[d] += (*out_shape)[i][d];
          }
        } else {
          back_calculate_dshape[d] = (*out_shape)[0][d];
        }
      }
      SHAPE_ASSIGN_CHECK(*in_shape, slice_enum::kData, back_calculate_dshape);
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SliceChannelProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SliceChannel";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return out_grad;
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return nullptr;
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  SliceChannelParam param_;
};  // class SliceChannelProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SLICE_CHANNEL_INL_H_
