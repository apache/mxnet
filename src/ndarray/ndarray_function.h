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
 *  Copyright (c) 2015 by Contributors
 * \file ndarray_op.h
 * \brief the real execution functions of ndarray operations
 */
#ifndef MXNET_NDARRAY_NDARRAY_FUNCTION_H_
#define MXNET_NDARRAY_NDARRAY_FUNCTION_H_

#include <dmlc/logging.h>
#include <mshadow/tensor.h>
#include <mxnet/base.h>
#include <mxnet/resource.h>
#include <mxnet/ndarray.h>
#include <vector>
#include "../operator/mshadow_op.h"
#include "../operator/tensor/init_op.h"

namespace mxnet {
/*! \brief namespace to support all possible Ndarray operator */
namespace ndarray {
struct BinaryBase {
  inline static TShape GetShape(const TShape &lshape, const TShape &rshape) {
    CHECK(lshape == rshape) << "operands shape mismatch";
    CHECK(lshape.ndim() != 0) << "source operand have zero dimension shape";
    return lshape;
  }
};

// operators
struct Plus : public BinaryBase {
  typedef op::mshadow_op::plus mshadow_op;
};

struct Minus : public BinaryBase {
  typedef op::mshadow_op::minus mshadow_op;
};

struct Mul : public BinaryBase {
  typedef op::mshadow_op::mul mshadow_op;
};

struct Div : public BinaryBase {
  typedef op::mshadow_op::div mshadow_op;
};

struct Mod : public BinaryBase {
  typedef op::mshadow_op::mod mshadow_op;
};

struct ClipMin : public BinaryBase {
  struct mshadow_op {
    template<typename DType>
    MSHADOW_XINLINE static DType Map(DType a, DType b) {
      if (a < b) {
        return b;
      } else {
        return a;
      }
    }
  };
};

struct ClipMax : public BinaryBase {
  struct mshadow_op {
    template<typename DType>
    MSHADOW_XINLINE static DType Map(DType a, DType b) {
      if (a > b) {
        return b;
      } else {
        return a;
      }
    }
  };
};


struct OneHotEncode {
  inline static TShape GetShape(const TShape &index, const TShape &proptype) {
    CHECK(index.ndim() == 1 && proptype.ndim() == 2) << "OneHotEncode only support 1d index.";
    CHECK_EQ(index[0], proptype[0]) << "OneHotEncode shape inconsistent";
    return proptype;
  }
};

struct MatChooseRowElem {
  inline static TShape GetShape(const TShape &lshape, const TShape &rshape) {
    CHECK(lshape.ndim() == 2 && rshape.ndim() == 1)
        << "choose_row_element only support 2D Matrix and 1D index";
    CHECK_EQ(lshape[0], rshape[0]) << "choose_row_element index and matrix shape mismatch";
    return rshape;
  }
};

struct MatFillRowElem {
  inline static TShape GetShape(const TShape &lshape, const TShape &mshape, const TShape &rshape) {
    CHECK(lshape.ndim() == 2 && mshape.ndim() == 1 && rshape.ndim() == 1)
        << "fill_row_element only support 2D Matrix, 1D value and 1D index";
    CHECK((lshape[0] == mshape[0]) && (mshape[0] == rshape[0]))
        << "choose_row_element index vector, value vector and matrix shape mismatch";
    return lshape;
  }
};

// type holder for random number generators
struct UniformDistribution {};

struct GaussianDistribution {};

struct GammaDistribution {};

struct ExponentialDistribution {};

struct PoissonDistribution {};

struct NegBinomialDistribution {};

struct GenNegBinomialDistribution {};

template<typename Device>
void EvalClip(const TBlob &src, const real_t &a_min, const real_t &a_max,
              TBlob *ret, RunContext ctx);

template<typename Device, typename OP>
void Eval(const TBlob &lhs, const TBlob &mhs, const TBlob &rhs, TBlob *ret, RunContext ctx);

template<typename Device, typename OP>
void Eval(const TBlob &lhs, const TBlob &rhs, TBlob *ret, RunContext ctx);

template<typename Device, typename OP>
void Eval(const TBlob &src, TBlob *ret, RunContext ctx);

template<typename Device, typename OP, bool reverse>
void Eval(const TBlob &lhs, const real_t &rhs, TBlob *ret, RunContext ctx);

template<typename Device>
void Eval(const real_t &rhs, TBlob *ret, RunContext ctx);

template<typename Device, typename Distribution>
void EvalRandom(const real_t &a,
                const real_t &b,
                const Resource &resource,
                TBlob *ret,  RunContext ctx);

// copy function when only cpu is involved
template<typename DeviceFrom, typename DeviceTo>
void Copy(const TBlob &from, TBlob *to,
          Context from_ctx, Context to_ctx,
          RunContext ctx);

template<typename Device>
void ElementwiseSum(const std::vector<TBlob> source,
                    TBlob *out,
                    RunContext ctx);

/*!
 * \brief Interface for parallel impl of elemwise sum for sparse matrices
 */
template<typename xpu>
void ElementwiseSum(mshadow::Stream<xpu>* s,
                    const Resource& rsc,
                    const std::vector<NDArray>& nds,
                    NDArray* out);

/*!
 * \brief Set a row_sparse NDArray with val
 * \param s - The device stream
 * \param val - The value to be set
 * \param dst - NDArray which is to be set to val
 */
template<typename xpu>
void SetValueRspImpl(mshadow::Stream<xpu> *s,
                     const real_t val, NDArray *dst) {
  CHECK_EQ(dst->storage_type(), kRowSparseStorage);
  using namespace mxnet::op;
  nnvm::dim_t nnr = dst->shape()[0];
  dst->CheckAndAlloc({mshadow::Shape1(nnr)});
  MSHADOW_IDX_TYPE_SWITCH(dst->aux_type(rowsparse::kIdx), IType, {
    IType* idx = dst->aux_data(rowsparse::kIdx).dptr<IType>();
    mxnet_op::Kernel<PopulateFullIdxRspKernel, xpu>::Launch(s, nnr, idx);
  });
  Fill<false>(s, dst->data(), kWriteTo, val);
}

template<typename xpu>
void Eval(mshadow::Stream<xpu> *s,
          const real_t val, const NDArray& dst);

// broadcasting
template <typename Device>
void EvalBroadcast(TBlob const& src, TBlob* ret, int size, RunContext ctx);

}  // namespace ndarray
}  // namespace mxnet
#endif  // MXNET_NDARRAY_NDARRAY_FUNCTION_H_
