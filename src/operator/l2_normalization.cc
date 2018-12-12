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
 * \file l2_normalization.cc
 * \brief l2 normalization operator
*/
#include "./l2_normalization-inl.h"

/* VisualStudio only supports openmp 2.0 */
#ifdef _MSC_VER
#define collapse(x)
#endif

namespace mxnet {
namespace op {

template<typename DType>
class L2NormalizationOpCPU : public L2NormalizationOp<cpu, DType> {
 public:
  explicit L2NormalizationOpCPU(L2NormalizationParam p)
      : L2NormalizationOp<cpu, DType>(p) {}
  void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data,
               const std::vector<TBlob> &aux_args) override {
    using namespace mshadow;
    using namespace mshadow::expr;
    if (req[l2_normalization::kOut] == kNullOp) return;
    CHECK_EQ(req[l2_normalization::kOut], kWriteTo);
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 2U);
    Stream<cpu> *s = ctx.get_stream<cpu>();
    TShape orig_shape = in_data[l2_normalization::kData].shape_;
    auto omp_threads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
    if (this->param_.mode == l2_normalization::kInstance) {
      Shape<2> dshape = Shape2(orig_shape[0],
        orig_shape.ProdShape(1, orig_shape.ndim()));
      Tensor<cpu, 2, DType> data = in_data[l2_normalization::kData]
        .get_with_shape<cpu, 2, DType>(dshape, s);
      Tensor<cpu, 2, DType> out = out_data[l2_normalization::kOut]
        .get_with_shape<cpu, 2, DType>(dshape, s);
      Tensor<cpu, 1, DType> norm = out_data[l2_normalization::kNorm].get<cpu, 1, DType>(s);
#pragma omp parallel for num_threads(omp_threads)
      for (int shape0 = 0; shape0 < static_cast<int>(dshape[0]); shape0++) {
        norm[shape0] = DType(this->param_.eps);
        for (int shape1 = 0; shape1 < static_cast<int>(dshape[1]); shape1++) {
          norm[shape0] += data[shape0][shape1] * data[shape0][shape1];
        }
        norm[shape0] = std::sqrt(norm[shape0]);
        for (int shape1 = 0; shape1 < static_cast<int>(dshape[1]); shape1++) {
          out[shape0][shape1] = data[shape0][shape1] / norm[shape0];
        }
      }
    } else if (this->param_.mode == l2_normalization::kChannel) {
      CHECK_GE(orig_shape.ndim(), 3U);
      Shape<3> dshape = Shape3(orig_shape[0], orig_shape[1],
        orig_shape.ProdShape(2, orig_shape.ndim()));
      Tensor<cpu, 3, DType> data = in_data[l2_normalization::kData]
        .get_with_shape<cpu, 3, DType>(dshape, s);
      Tensor<cpu, 3, DType> out = out_data[l2_normalization::kOut]
        .get_with_shape<cpu, 3, DType>(dshape, s);
      Shape<2> norm_shape = Shape2(dshape[0], dshape[2]);
      Tensor<cpu, 2, DType> norm = out_data[l2_normalization::kNorm]
        .get_with_shape<cpu, 2, DType>(norm_shape, s);
#pragma omp parallel for num_threads(omp_threads) collapse(2)
      for (int shape0 = 0; shape0 < static_cast<int>(dshape[0]); shape0++) {
        for (int shape2 = 0; shape2 < static_cast<int>(dshape[2]); shape2++) {
          norm[shape0][shape2] = DType(this->param_.eps);
          for (int shape1 = 0; shape1 < static_cast<int>(dshape[1]); shape1++) {
            norm[shape0][shape2] += data[shape0][shape1][shape2] * data[shape0][shape1][shape2];
          }
          norm[shape0][shape2] = std::sqrt(norm[shape0][shape2]);
          for (int shape1 = 0; shape1 < static_cast<int>(dshape[1]); shape1++) {
            out[shape0][shape1][shape2] = data[shape0][shape1][shape2] / norm[shape0][shape2];
          }
        }
      }
    } else if (this->param_.mode == l2_normalization::kSpatial) {
      CHECK_GE(orig_shape.ndim(), 3U);
      Shape<3> dshape = Shape3(orig_shape[0], orig_shape[1],
        orig_shape.ProdShape(2, orig_shape.ndim()));
      Tensor<cpu, 3, DType> data = in_data[l2_normalization::kData]
        .get_with_shape<cpu, 3, DType>(dshape, s);
      Tensor<cpu, 3, DType> out = out_data[l2_normalization::kOut]
        .get_with_shape<cpu, 3, DType>(dshape, s);
      Shape<2> norm_shape = Shape2(dshape[0], dshape[1]);
      Tensor<cpu, 2, DType> norm = out_data[l2_normalization::kNorm]
        .get_with_shape<cpu, 2, DType>(norm_shape, s);
#pragma omp parallel for num_threads(omp_threads) collapse(2)
      for (int shape0 = 0; shape0 < static_cast<int>(dshape[0]); shape0++) {
        for (int shape1 = 0; shape1 < static_cast<int>(dshape[1]); shape1++) {
          norm[shape0][shape1] = DType(this->param_.eps);
          for (int shape2 = 0; shape2 < static_cast<int>(dshape[2]); shape2++) {
            norm[shape0][shape1] += data[shape0][shape1][shape2] * data[shape0][shape1][shape2];
          }
          norm[shape0][shape1] = std::sqrt(norm[shape0][shape1]);
          for (int shape2 = 0; shape2 < static_cast<int>(dshape[2]); shape2++) {
            out[shape0][shape1][shape2] = data[shape0][shape1][shape2] / norm[shape0][shape1];
          }
        }
      }
    } else {
      LOG(FATAL) << "Unexpected mode in l2 normalization";
    }
  }
};

template<>
Operator* CreateOp<cpu>(L2NormalizationParam param, int dtype) {
  Operator* op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new L2NormalizationOpCPU<DType>(param);
  });
  return op;
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* L2NormalizationProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                                std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, this->param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(L2NormalizationParam);

MXNET_REGISTER_OP_PROPERTY(L2Normalization, L2NormalizationProp)
.describe(R"code(Normalize the input array using the L2 norm.

For 1-D NDArray, it computes::

  out = data / sqrt(sum(data ** 2) + eps)

For N-D NDArray, if the input array has shape (N, N, ..., N),

with ``mode`` = ``instance``, it normalizes each instance in the multidimensional
array by its L2 norm.::

  for i in 0...N
    out[i,:,:,...,:] = data[i,:,:,...,:] / sqrt(sum(data[i,:,:,...,:] ** 2) + eps)

with ``mode`` = ``channel``, it normalizes each channel in the array by its L2 norm.::

  for i in 0...N
    out[:,i,:,...,:] = data[:,i,:,...,:] / sqrt(sum(data[:,i,:,...,:] ** 2) + eps)

with ``mode`` = ``spatial``, it normalizes the cross channel norm for each position
in the array by its L2 norm.::

  for dim in 2...N
    for i in 0...N
      out[.....,i,...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out, indices=i, axis=dim) ** 2) + eps)
          -dim-

Example::

  x = [[[1,2],
        [3,4]],
       [[2,2],
        [5,6]]]

  L2Normalization(x, mode='instance')
  =[[[ 0.18257418  0.36514837]
     [ 0.54772252  0.73029673]]
    [[ 0.24077171  0.24077171]
     [ 0.60192931  0.72231513]]]

  L2Normalization(x, mode='channel')
  =[[[ 0.31622776  0.44721359]
     [ 0.94868326  0.89442718]]
    [[ 0.37139067  0.31622776]
     [ 0.92847669  0.94868326]]]

  L2Normalization(x, mode='spatial')
  =[[[ 0.44721359  0.89442718]
     [ 0.60000002  0.80000001]]
    [[ 0.70710677  0.70710677]
     [ 0.6401844   0.76822126]]]

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array to normalize.")
.add_arguments(L2NormalizationParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
