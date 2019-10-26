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
#include <mshadow/tensor.h>
#include "./index_array-inl.h"


namespace mxnet {
namespace op {

void IndexArrayForwardCPU(const nnvm::NodeAttrs &attrs,
                          const OpContext &ctx,
                          const std::vector<TBlob> &inputs,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];

  const IndexArrayParam& param = nnvm::get<IndexArrayParam>(attrs.parsed);

  const TShape inshape = in_data.shape_;
  const int ndim = inshape.ndim();

  Stream<cpu> *stream = ctx.get_stream<cpu>();

  using namespace mxnet_op;

  if (param.axes.has_value()) {
    const mxnet::Tuple<int>& axes = param.axes.value();
    const int naxes = axes.ndim();

    std::vector<int64_t> index_products = IndexArrayComputeIndexProducts(inshape);

    Tensor<cpu, 1, int64_t> workspace =
        ctx.requested[0].get_space_typed<cpu, 1, int64_t>(Shape1(2 * naxes), stream);

    IndexArrayBuildSelectedAxesWorkspace(axes, index_products, workspace.dptr_, ndim);
    
    // Assumes param.target_axis is -1 or 0.
    const ptrdiff_t index_axis_offset = param.target_axis == -1 ? naxes : 1;
    const ptrdiff_t target_axis_offset = param.target_axis == -1 ? 1: in_data.Size();

    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<IndexArrayKernel<req_type>, cpu>::Launch(stream, in_data.Size(),
          out_data.dptr<int64_t>(), naxes, index_axis_offset, target_axis_offset, workspace.dptr_);
    });
  } else {
    // Assumes param.target_axis is -1 or 0.
    const ptrdiff_t index_axis_offset = param.target_axis == -1 ? ndim : 1;
    const ptrdiff_t target_axis_offset = param.target_axis == -1 ? 1: in_data.Size();

    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<IndexArrayDefaultKernel<req_type>, cpu>::Launch(stream, in_data.Size(),
          out_data.dptr<int64_t>(), ndim, index_axis_offset, target_axis_offset, inshape.data());
    });
  }
}

DMLC_REGISTER_PARAMETER(IndexArrayParam);

NNVM_REGISTER_OP(_contrib_index_array)
.describe(R"code(Returns an array of indexes of the input array.

For an input array with shape  :math:`(d_1, d_2, ..., d_n)`, `index_array` returns a
:math:`(d_1, d_2, ..., d_n, n)` array `idx`, where
:math:`idx[i_1, i_2, ..., i_n, :] = [i_1, i_2, ..., i_n]`.

Additionally, when the parameter `axes` is specified, `idx` will be a
:math:`(d_1, d_2, ..., d_n, m)` array where `m` is the length of `axes`, and the following
equality will hold: :math:`idx[i_1, i_2, ..., i_n, j] = i_{axes[j]}`.

The `target_axis` parameter can be used to move the index axis to the front of the shape.
For instance, if `target_axis` is set to 0, the `index_array` of
:math:`(d_1, d_2, ..., d_n)` will be a :math:`(n, d_1, d_2, ..., d_n)` array `idx`, where
:math:`idx[:, i_1, i_2, ..., i_n] = [i_1, i_2, ..., i_n]`.

Examples::

    x = mx.nd.ones((3, 2))

    mx.nd.contrib.index_array(x) = [[[0 0]
                                     [0 1]]

                                    [[1 0]
                                     [1 1]]

                                    [[2 0]
                                     [2 1]]]

    mx.nd.contrib.index_array(x, target_axis=0) = [[[0 0]
                                                    [1 1]
                                                    [2 2]]
                                                  
                                                   [[0 1]
                                                    [0 1]
                                                    [0 1]]]


    x = mx.nd.ones((3, 2, 2))

    mx.nd.contrib.index_array(x, axes=(1, 0)) = [[[[0 0]
                                                   [0 0]]

                                                  [[1 0]
                                                   [1 0]]]


                                                 [[[0 1]
                                                   [0 1]]

                                                  [[1 1]
                                                   [1 1]]]


                                                 [[[0 2]
                                                   [0 2]]

                                                  [[1 2]
                                                   [1 2]]]]

    mx.nd.contrib.index_array(x, axes=(1, 0), target_axis=0) = [[[[0 0]
                                                                  [1 1]]
                                                               
                                                                 [[0 0]
                                                                  [1 1]]
                                                               
                                                                 [[0 0]
                                                                  [1 1]]]
                                                               
                                                               
                                                                [[[0 0]
                                                                  [0 0]]
                                                               
                                                                 [[1 1]
                                                                  [1 1]]
                                                               
                                                                 [[2 2]
                                                                  [2 2]]]]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
                                 [](const NodeAttrs &attrs) {
                                   return std::vector<std::string>{ "data" };
                                 })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                  [](const NodeAttrs &attrs) {
                                    return std::vector<std::string>{ "output" };
                                  })
.set_attr_parser(ParamParser<IndexArrayParam>)
.set_attr<mxnet::FInferShape>("FInferShape", [](const nnvm::NodeAttrs &attrs,
                                                mxnet::ShapeVector *in_shape,
                                                mxnet::ShapeVector *out_shape) {
  const IndexArrayParam &param = nnvm::get<IndexArrayParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  CHECK_EQ(out_shape->size(), 1U);
  CHECK((param.target_axis == 0) || (param.target_axis == -1));

  const mxnet::TShape &inshape = (*in_shape)[index_array_enum::kIn];
  if (!mxnet::ndim_is_known(inshape)) return false;

  mxnet::TShape oshape = mxnet::TShape(inshape.ndim() + 1, 0);

  const int target_axis = (param.target_axis + inshape.ndim() + 1) % (inshape.ndim() + 1);

  for (int i = 0; i < inshape.ndim(); i++) {
    oshape[i + (target_axis <= i)] = inshape[i];
  }
  if (param.axes.has_value()) {
    oshape[target_axis] = param.axes.value().ndim();
  } else {
    oshape[target_axis] = inshape.ndim();
  }

  SHAPE_ASSIGN_CHECK(*out_shape, 0, oshape);
  return shape_is_known(oshape);
})
.set_attr<nnvm::FInferType>("FInferType", [](const nnvm::NodeAttrs &attrs,
                                             std::vector<int> *in_attrs,
                                             std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt64);
  return out_attrs->at(0) != -1;
})
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
  [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>(1, 0);
  })
.set_attr<FCompute>("FCompute<cpu>", IndexArrayForwardCPU)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.add_argument("data", "NDArray-or-Symbol", "Input data")
.add_arguments(IndexArrayParam::__FIELDS__());


}  // namespace op
}  // namespace mxnet

