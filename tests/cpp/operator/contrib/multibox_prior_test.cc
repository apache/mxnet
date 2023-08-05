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

//
// The goal of this file multibox_prior_test.cc is to be the same as this test class
// MultiboxPriorTest.java
// https://github.com/deepjavalibrary/djl/blob/master/integration/src/main/java/ai/djl/integration/tests/modality/cv/MultiBoxPriorTest.java
//

#include <vector>
#include <iomanip>
#include <sstream>

#include "../include/test_op_runner.h"
#include "operator/contrib/multibox_prior-inl.h"
#include "mxnet/operator.h"

using namespace mxnet;
using DType = float;

TEST(CORE_OP_RUNNER, Multibox_prior) {
  std::vector<float> sizes   = {0.2f, 0.272f};
  std::vector<float> ratios  = {1.0f, 2.0f, 0.5f};
  mxnet::Tuple<float> steps  = mxnet::Tuple<float>({-1.0f, -1.0f});
  std::vector<float> offsets = {0.5f, 0.5f};

  mxnet::op::MultiBoxPriorParam p1 = mxnet::op::MultiBoxPriorParam();
  p1.sizes                         = sizes;
  p1.ratios                        = ratios;
  p1.steps                         = steps;
  p1.offsets                       = offsets;
  Operator* op2                    = mxnet::op::CreateOp<cpu>(p1, mshadow::kFloat32);

  std::vector<TBlob> in_data_fwd_, in_data_bwd_;
  std::vector<TBlob> aux_data_, out_data_, in_grad_, out_grad_;

  std::vector<TBlob> inputs;
  std::vector<TBlob> outputs;

  int in_width  = 512;
  int in_height = 512;

  Context ctx = Context();

  int arrangeVal      = 3.0f * 512.0f * 512.0f;
  TShape arrangeShape = TShape({arrangeVal});

  NDArray arangeNdArray = NDArray(arrangeShape, ctx);

  TShape reshapeShape = TShape({1, 3, in_height, in_width});
  NDArray ndArray     = arangeNdArray.Reshape(reshapeShape);

  TShape outShape    = TShape({1, 1048576, 4});
  NDArray outNdArray = NDArray(outShape, ctx);

  std::vector<TBlob> in_data;
  std::vector<OpReqType> req;
  std::vector<TBlob> out_data;
  std::vector<TBlob> aux_states;

  TBlob inData      = ndArray.data();
  inData.type_flag_ = mshadow::kFloat32;

  in_data = {inData};

  TBlob outData      = outNdArray.data();
  outData.type_flag_ = mshadow::kFloat32;
  out_data           = {outData};

  OpContext opContext = OpContext();

  op2->Forward(opContext, in_data, req, out_data, aux_states);

  TBlob anchors = out_data.front();

  int64_t resultShapeDim0 = anchors.size(0);
  int64_t resultShapeDim1 = anchors.size(1);
  int64_t resultShapeDim2 = anchors.size(2);
  assert(resultShapeDim0 == 1);
  assert(resultShapeDim1 == 1048576);
  assert(resultShapeDim2 == 4);

  float* anchorFloatArray = anchors.dptr<DType>();
  std::stringstream stream;
  stream << std::fixed << std::setprecision(8) << anchorFloatArray[0];
  std::string expectedVal = "-0.09902344";
  assert(expectedVal.compare(stream.str()) == 0);
}
