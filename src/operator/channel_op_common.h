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
    for (size_t i = 0; i < size; ++i) {
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
    for (size_t i = 0; i < size; ++i) {
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

template<typename cpu, int dim, typename DType>
void Split_2D(const mshadow::Tensor<cpu, dim, DType> &input,
           std::vector<mshadow::Tensor<cpu, dim, DType> > *output,
           const int dimension, const std::vector<OpReqType> &req) {
  if (dimension != 1) {
    LOG(FATAL) << "dimension (" << dimension << ") must == 1";
  }
  if (dim != 3) {
    LOG(FATAL) << "dimension (" << dim << ") must == 3";
  } else {
    std::vector<mshadow::Tensor<cpu, dim, DType> > out = *output;
    size_t size = out.size();
    std::vector<int>slice_len;
    std::vector<int>begin_pos;
    begin_pos.push_back(0);

    for (index_t i = 0; i < size; ++i) {
      slice_len.push_back(out[i].size(dimension));
      begin_pos.push_back(begin_pos[i] + out[i].size(dimension));
    }
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (int i = 0; i < input.shape_[0]; i++) {
      int iRow = i*input.shape_[1];
      for (int j = 0; j < size; j++) {
        int jRow = i*slice_len[j];
        int iPos = iRow + begin_pos[j];
        for (int k = 0; k < slice_len[j]; k++) {
          out[j].dptr_[jRow + k] = input.dptr_[iPos + k];
        }
      }
    }
  }
}


}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CHANNEL_OP_COMMON_H_
