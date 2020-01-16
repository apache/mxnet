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
 * \file upsampling_nearest.cc
 * \brief
 * \author Bing Xu, Da Zheng
*/

#include "./upsampling-inl.h"
#include <nnvm/op_attr_types.h>
#include "./deconvolution-inl.h"

namespace mxnet {
namespace op {

static bool UpSamplingShape(const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector *in_shape, mxnet::ShapeVector *out_shape) {
  const UpSamplingParam& param_ = nnvm::get<UpSamplingParam>(attrs.parsed);
  CHECK_GE(in_shape->size(), 1U);
  const mxnet::TShape &dshape = (*in_shape)[0];
  mxnet::TShape oshape = dshape;
  if (param_.sample_type == up_enum::kNearest) {
    CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args));
    oshape[1] = 0;
    for (auto& shape : *in_shape) {
      CHECK_EQ(shape.ndim(), 4U) << \
        "UpSamplingNearest: Input data should be 4D in (batch, channel, y, x)";
      int oh = dshape[2]*param_.scale, ow = dshape[3]*param_.scale;
      CHECK_EQ(oh%shape[2], 0U) << "UpSamplingNearest: input height of " << shape[2] << \
        "does not divide output height of " << oh;
      CHECK_EQ(ow%shape[3], 0U) << "UpSamplingNearest: input width of " << shape[3] << \
        "does not divide output width of " << ow;
      if (param_.multi_input_mode == up_enum::kSum) {
        CHECK(oshape[1] == 0 || oshape[1] == shape[1]) << \
                         "Number of channels must be the same when multi_input_mode==sum";
        oshape[1] = shape[1];
      } else {
        oshape[1] += shape[1];
      }
    }
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
    CHECK_EQ(dshape.ndim(), 4U) << \
      "UpSamplingBilinear: Input data should be 4D in (batch, channel, y, x)";
    if (!shape_is_known(dshape)) return false;
    int kernel = 2 * param_.scale - param_.scale % 2;
    SHAPE_ASSIGN_CHECK(*in_shape,
        up_enum::kWeight,
        mshadow::Shape4(dshape[1], 1, kernel, kernel));
    oshape = dshape;
  }
  oshape[2] = dshape[2] * param_.scale;
  oshape[3] = dshape[3] * param_.scale;
  out_shape->clear();
  out_shape->push_back(oshape);
  return true;
}

static inline std::vector<std::string> ListArguments(const UpSamplingParam& param) {
  if (param.sample_type == up_enum::kNearest) {
    std::vector<std::string> ret;
    for (int i = 0; i < param.num_args; ++i) {
      ret.push_back(std::string("arg") + std::to_string(i));
    }
    return ret;
  } else {
    return {"data", "weight"};
  }
}

static bool UpSamplingType(const nnvm::NodeAttrs& attrs,
                           std::vector<int> *in_type, std::vector<int> *out_type) {
  const UpSamplingParam& param = nnvm::get<UpSamplingParam>(attrs.parsed);
  CHECK_GE(in_type->size(), 1U);
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  for (size_t i = 0; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments(param)[i]);
    }
  }
  out_type->clear();
  out_type->push_back(dtype);
  return true;
}

struct UpSamplingGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    const UpSamplingParam& param_ = nnvm::get<UpSamplingParam>(n->attrs.parsed);
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    if (param_.sample_type != up_enum::kNearest) {
      heads.push_back(n->inputs[up_enum::kData]);
      heads.push_back(n->inputs[up_enum::kWeight]);
    }
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

DMLC_REGISTER_PARAMETER(UpSamplingParam);

NNVM_REGISTER_OP(UpSampling)
.describe(R"code(Upsamples the given input data.

Two algorithms (``sample_type``) are available for upsampling:

- Nearest Neighbor
- Bilinear

**Nearest Neighbor Upsampling**

Input data is expected to be NCHW.

Example::

  x = [[[[1. 1. 1.]
         [1. 1. 1.]
         [1. 1. 1.]]]]

  UpSampling(x, scale=2, sample_type='nearest') = [[[[1. 1. 1. 1. 1. 1.]
                                                     [1. 1. 1. 1. 1. 1.]
                                                     [1. 1. 1. 1. 1. 1.]
                                                     [1. 1. 1. 1. 1. 1.]
                                                     [1. 1. 1. 1. 1. 1.]
                                                     [1. 1. 1. 1. 1. 1.]]]]

**Bilinear Upsampling**

Uses `deconvolution` algorithm under the hood. You need provide both input data and the kernel.

Input data is expected to be NCHW.

`num_filter` is expected to be same as the number of channels.

Example::

  x = [[[[1. 1. 1.]
         [1. 1. 1.]
         [1. 1. 1.]]]]

  w = [[[[1. 1. 1. 1.]
         [1. 1. 1. 1.]
         [1. 1. 1. 1.]
         [1. 1. 1. 1.]]]]
  
  UpSampling(x, w, scale=2, sample_type='bilinear', num_filter=1) = [[[[1. 2. 2. 2. 2. 1.]
                                                                       [2. 4. 4. 4. 4. 2.]
                                                                       [2. 4. 4. 4. 4. 2.]
                                                                       [2. 4. 4. 4. 4. 2.]
                                                                       [2. 4. 4. 4. 4. 2.]
                                                                       [1. 2. 2. 2. 2. 1.]]]]
)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  const UpSamplingParam& params = nnvm::get<UpSamplingParam>(attrs.parsed);
  return params.sample_type == up_enum::kNearest ? params.num_args : 2;
})
.set_num_outputs(1)
.set_attr_parser(ParamParser<UpSamplingParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return ListArguments(nnvm::get<UpSamplingParam>(attrs.parsed));
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", UpSamplingShape)
.set_attr<nnvm::FInferType>("FInferType", UpSamplingType)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  const UpSamplingParam& param = nnvm::get<UpSamplingParam>(n.parsed);
  if (param.sample_type == up_enum::kNearest) {
    return std::vector<ResourceRequest>();
  } else {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  }
})
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", UpSamplingCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", UpSamplingGrad{"_backward_UpSampling"})
.set_attr<std::string>("key_var_num_args", "num_args")
.add_argument("data", "NDArray-or-Symbol[]", "Array of tensors to upsample. "
              "For bilinear upsampling, there should be 2 inputs - 1 data and 1 weight.")
.add_arguments(UpSamplingParam::__FIELDS__())
.set_attr<nnvm::FSetInputVarAttrOnCompose>("FSetInputVarAttrOnCompose",
    [](const nnvm::NodeAttrs& attrs, nnvm::NodePtr var, const int index) {
      if (var->attrs.dict.find("__init__") != var->attrs.dict.end()) return;
      if (index == 1) {
        var->attrs.dict["__init__"] = "[\"bilinear\", {}]";
      }
    });

NNVM_REGISTER_OP(_backward_UpSampling)
.set_num_inputs([](const NodeAttrs& attrs) {
  const UpSamplingParam& param_ = nnvm::get<UpSamplingParam>(attrs.parsed);
  if (param_.sample_type != up_enum::kNearest) {
    return 3;
  } else {
    return 1;
  }
})
.set_num_outputs([](const NodeAttrs& attrs) {
  const UpSamplingParam& params = nnvm::get<UpSamplingParam>(attrs.parsed);
  return params.sample_type == up_enum::kNearest ? params.num_args : 2;
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  const UpSamplingParam& param = nnvm::get<UpSamplingParam>(n.parsed);
  if (param.sample_type == up_enum::kNearest) {
    return std::vector<ResourceRequest>();
  } else {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  }
})
.set_attr_parser(ParamParser<UpSamplingParam>)
.set_attr<FCompute>("FCompute<cpu>", UpSamplingGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
