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
 * Copyright (c) 2019 by Contributors
 * \file np_choice_op.h
 * \brief Operator for random subset sampling
 */

#ifndef MXNET_OPERATOR_NUMPY_RANDOM_NP_CHOICE_OP_H_
#define MXNET_OPERATOR_NUMPY_RANDOM_NP_CHOICE_OP_H_

#include <mshadow/base.h>
#include <mxnet/operator_util.h>
#include <algorithm>
#include <string>
#include <vector>
#include "../../elemwise_op_common.h"
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

struct NumpyChoiceParam : public dmlc::Parameter<NumpyChoiceParam> {
  dmlc::optional<int64_t> a;
  std::string ctx;
  dmlc::optional<mxnet::Tuple<int64_t>> size;
  bool replace;
  bool weighted;
  DMLC_DECLARE_PARAMETER(NumpyChoiceParam) {
    DMLC_DECLARE_FIELD(a);
    DMLC_DECLARE_FIELD(size);
    DMLC_DECLARE_FIELD(ctx).set_default("cpu");
    DMLC_DECLARE_FIELD(replace).set_default(true);
    DMLC_DECLARE_FIELD(weighted).set_default(false);
  }
};

inline bool NumpyChoiceOpType(const nnvm::NodeAttrs &attrs,
                              std::vector<int> *in_attrs,
                              std::vector<int> *out_attrs) {
  (*out_attrs)[0] = mshadow::kInt64;
  return true;
}

inline bool NumpyChoiceOpShape(const nnvm::NodeAttrs &attrs,
                               std::vector<TShape> *in_attrs,
                               std::vector<TShape> *out_attrs) {
  const NumpyChoiceParam &param = nnvm::get<NumpyChoiceParam>(attrs.parsed);
  if (param.size.has_value()) {
    // Size declared.
    std::vector<dim_t> oshape_vec;
    const mxnet::Tuple<int64_t> &size = param.size.value();
    for (int i = 0; i < size.ndim(); ++i) {
      oshape_vec.emplace_back(size[i]);
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(oshape_vec));
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(0, -1))
  }
  return true;
}

template <typename xpu>
void _sort(float *key, int64_t *data, index_t length);

namespace mxnet_op {

// Uniform sample without replacement.
struct generate_samples {
  MSHADOW_XINLINE static void Map(index_t i, int64_t k, unsigned *rands) {
    rands[i] = rands[i] % (i + k + 1);
  }
};

template <typename xpu>
struct generate_reservoir {
  MSHADOW_XINLINE static void Map(index_t dummy_index, int64_t *indices,
                                  unsigned *samples, int64_t nb_iterations,
                                  int64_t k) {
    for (int64_t i = 0; i < nb_iterations; i++) {
      int64_t z = samples[i];
      if (z < k) {
        int64_t t = indices[z];
        indices[z] = indices[i + k];
        indices[i + k] = t;
      }
    }
  }
};

// Uniform sample with replacement.
struct random_indices {
  MSHADOW_XINLINE static void Map(index_t i, unsigned *samples, int64_t *outs,
                                  int64_t k) {
    outs[i] = samples[i] % k;
  }
};

// Weighted sample without replacement.
// Use perturbed Gumbel variates as keys.
template <typename IType>
struct generate_keys {
  MSHADOW_XINLINE static void Map(index_t i, float *uniforms, IType *weights) {
    uniforms[i] = -logf(-logf(uniforms[i])) + logf(weights[i]);
  }
};

// Weighted sample with replacement.
template <typename IType>
struct categorical_sampling {
  MSHADOW_XINLINE static void Map(index_t i, IType *weights, size_t length,
                                  float *uniforms, int64_t *outs) {
    outs[i] = 0;
    float acc = 0.0;
    float threshold = uniforms[i];
    for (size_t k = 0; k < length; k++) {
      acc += weights[k];
      if (acc < threshold) {
        outs[i] += 1;
      }
    }
  }
};

}  // namespace mxnet_op

template <typename xpu>
void NumpyChoiceForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                        const std::vector<TBlob> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const NumpyChoiceParam &param = nnvm::get<NumpyChoiceParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  bool replace = param.replace;
  bool weighted = param.weighted;
  int64_t input_size = 0;
  int weight_index = 0;
  if (param.a.has_value()) {
    input_size = param.a.value();
  } else {
    input_size = inputs[0].Size();
    weight_index += 1;
  }
  int64_t output_size = outputs[0].Size();
  if (weighted) {
    Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
    int64_t random_tensor_size = replace ? output_size : input_size;
    int64_t indices_size = replace ? 0 : input_size;
    Tensor<xpu, 1, char> workspace =
        ctx.requested[1].get_space_typed<xpu, 1, char>(
            Shape1(indices_size * sizeof(int64_t) +
                   (random_tensor_size * sizeof(float) / 7 + 1) * 8),
            s);
    // slice workspace
    char *workspace_ptr = workspace.dptr_;
    Tensor<xpu, 1, float> random_numbers =
        Tensor<xpu, 1, float>(reinterpret_cast<float *>(workspace_ptr),
                              Shape1(random_tensor_size), s);
    prnd->SampleUniform(&random_numbers, 0, 1);
    workspace_ptr += ((random_tensor_size * sizeof(float) / 7 + 1) * 8);
    if (replace) {
      MSHADOW_REAL_TYPE_SWITCH(inputs[weight_index].type_flag_, IType, {
        Kernel<categorical_sampling<IType>, xpu>::Launch(
            s, output_size, inputs[weight_index].dptr<IType>(), input_size,
            random_numbers.dptr_, outputs[0].dptr<int64_t>());
      });
    } else {
      Tensor<xpu, 1, int64_t> indices = Tensor<xpu, 1, int64_t>(
          reinterpret_cast<int64_t *>(workspace_ptr), Shape1(indices_size), s);
      indices = expr::range((int64_t)0, input_size);
      MSHADOW_REAL_TYPE_SWITCH(inputs[weight_index].type_flag_, IType, {
        Kernel<generate_keys<IType>, xpu>::Launch(s, input_size, random_numbers.dptr_,
                                           inputs[weight_index].dptr<IType>());
      });
      _sort<xpu>(random_numbers.dptr_, indices.dptr_, input_size);
      Copy(outputs[0].FlatTo1D<xpu, int64_t>(s), indices.Slice(0, output_size), s);
    }
  } else {
    Random<xpu, unsigned> *prnd = ctx.requested[0].get_random<xpu, unsigned>(s);
    int64_t random_tensor_size =
        (replace ? output_size
                 : std::min(output_size, input_size - output_size));
    int64_t indices_size = replace ? 0 : input_size;
    Tensor<xpu, 1, char> workspace =
        ctx.requested[1].get_space_typed<xpu, 1, char>(
            Shape1(indices_size * sizeof(int64_t) +
                   (random_tensor_size * sizeof(unsigned) / 7 + 1) * 8),
            s);
    // slice workspace
    char *workspace_ptr = workspace.dptr_;
    Tensor<xpu, 1, unsigned> random_numbers =
        Tensor<xpu, 1, unsigned>(reinterpret_cast<unsigned *>(workspace_ptr),
                                 Shape1(random_tensor_size), s);
    prnd->GetRandInt(random_numbers);
    workspace_ptr += ((random_tensor_size * sizeof(unsigned) / 7 + 1) * 8);
    if (replace) {
      Kernel<random_indices, xpu>::Launch(s, output_size, random_numbers.dptr_,
                                          outputs[0].dptr<int64_t>(),
                                          input_size);
    } else {
      Tensor<xpu, 1, int64_t> indices = Tensor<xpu, 1, int64_t>(
          reinterpret_cast<int64_t *>(workspace_ptr), Shape1(indices_size), s);
      indices = expr::range((int64_t)0, input_size);
      int64_t nb_iterations = random_tensor_size;
      int64_t split = input_size - nb_iterations;
      Kernel<generate_samples, xpu>::Launch(s, random_tensor_size, split,
                                            random_numbers.dptr_);
      // Reservoir sampling.
      Kernel<generate_reservoir<xpu>, xpu>::Launch(
          s, 1, indices.dptr_, random_numbers.dptr_, nb_iterations, split);
      index_t begin;
      index_t end;
      if (2 * output_size < input_size) {
        begin = input_size - output_size;
        end = input_size;
      } else {
        begin = 0;
        end = output_size;
      }
      Copy(outputs[0].FlatTo1D<xpu, int64_t>(s), indices.Slice(begin, end), s);
    }
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_RANDOM_NP_CHOICE_OP_H_
