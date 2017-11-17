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
 * \file channel_op_common.h
 * \brief common function used for concat and split channel
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_CHANNEL_OP_COMMON_H_
#define MXNET_OPERATOR_CHANNEL_OP_COMMON_H_
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <vector>
#include "./operator_common.h"

namespace mxnet {
namespace op {

template<typename xpu, int dim, int cdim, typename DType>
inline void concatenate_helper(const std::vector<mshadow::Tensor<xpu, dim, DType> > &input,
                               mshadow::Tensor<xpu, dim, DType> *output, const int dimension,
                               const OpReqType req) {
  using mshadow::expr::concat;
  using mshadow::expr::slice;

  if (dimension == cdim) {
    mshadow::Tensor<xpu, dim, DType> out = *output;
    size_t size = input.size();
    index_t begin = 0;
    for (index_t i = 0; i < size; ++i) {
      index_t end = begin + input[i].size(cdim);
      Assign(slice<cdim>(out, begin, end), req, input[i]);
      begin = end;
    }
  } else {
    concatenate_helper<xpu, dim, (cdim > 0 ? cdim - 1 : 0)>(input, output, dimension, req);
  }
}

template<typename xpu, int dim, typename DType>
inline void Concatenate(const std::vector<mshadow::Tensor<xpu, dim, DType> > &input,
                        mshadow::Tensor<xpu, dim, DType> *output, const int dimension,
                        const OpReqType req) {
  if (dimension < 0) {
    LOG(FATAL) << "dimension (" << dimension << ") must be greater than 0";
  } else if (dimension >= dim) {
    LOG(FATAL) << "dimension (" << dimension << ") must be smaller than dim (" << dim << ")";
  } else {
    concatenate_helper<xpu, dim, dim-1>(input, output, dimension, req);
  }
}


template<typename xpu, int dim, int cdim, typename DType>
void split_helper(const mshadow::Tensor<xpu, dim, DType> &input,
           std::vector<mshadow::Tensor<xpu, dim, DType> > *output,
           const int dimension, const std::vector<OpReqType> &req) {
  using mshadow::expr::concat;
  using mshadow::expr::slice;

  if (dimension == cdim) {
    std::vector<mshadow::Tensor<xpu, dim, DType> > out = *output;
    size_t size = out.size();
    index_t begin = 0;
    for (index_t i = 0; i < size; ++i) {
      index_t end = begin + out[i].size(cdim);
      Assign(out[i], req[i], slice<cdim>(input, begin, end));
      begin = end;
    }
  } else {
    split_helper<xpu, dim, (cdim > 0 ? cdim - 1 : 0)>(input, output, dimension, req);
  }
}

template<typename xpu, int dim, typename DType>
void Split(const mshadow::Tensor<xpu, dim, DType> &input,
           std::vector<mshadow::Tensor<xpu, dim, DType> > *output,
           const int dimension, const std::vector<OpReqType> &req) {
  if (dimension < 0) {
    LOG(FATAL) << "dimension (" << dimension << ") must be greater than 0";
  } else if (dimension >= dim) {
    LOG(FATAL) << "dimension (" << dimension << ") must be smaller than dim (" << dim << ")";
  } else {
    split_helper<xpu, dim, dim-1>(input, output, dimension, req);
  }
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CHANNEL_OP_COMMON_H_
