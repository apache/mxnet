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
 * \file reduceto1d.h
 * \brief support for sum_rows and sumall_except_dim
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_REDUCETO1D_H_
#define MSHADOW_EXTENSION_REDUCETO1D_H_
#include "../extension.h"
namespace mshadow {
namespace expr {
/*!
 * \brief reduction to 1 dimension tensor
 * input: Tensor<Device,k>: ishape
 * output: Tensor<Device,1> shape[0] = ishape[dimkeep];
 *
 * \tparam SrcExp type of expression to be reduced
 * \tparam DType the data type of the scalar
 * \tparam Reducer which reducer to use
 * \tparam m_dimkeep which dimension to be kept, encoded with dimsrc - dimkeep
 */
template<typename SrcExp, typename DType, typename Reducer, int m_dimkeep>
struct ReduceTo1DExp:
      public Exp<ReduceTo1DExp<SrcExp, DType, Reducer, m_dimkeep>,
                 DType, type::kComplex> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief source operand, scale of the  */
  DType scale_;
  /*! \brief construct a repmat expression from src and nrow */
  ReduceTo1DExp(const SrcExp& src, DType scale) : src_(src), scale_(scale) {}
};
/*!
 * \brief a sum over all dimensions, except dimkeep
 * \param exp input expression that must be a matrix Tensor<?,2>
 * \return a expresion with type Tensor<Device,1>
 * \tparam dimkeep the dimension that will be kept
 * \tparam SrcExp expression
 * \tparam etype type of expression
 */
template<int dimkeep,  typename SrcExp, typename DType, int etype>
inline ReduceTo1DExp<SrcExp, DType, red::sum,
                     ExpInfo<SrcExp>::kDim - dimkeep>
sumall_except_dim(const Exp<SrcExp, DType, etype> &exp) {
  return ReduceTo1DExp<SrcExp, DType, red::sum,
                       ExpInfo<SrcExp>::kDim - dimkeep>(exp.self(), DType(1));
}
/*!
 * \brief reduce over all dimensions, except dimkeep
 * \param exp input expression that must be a matrix Tensor<?,2>
 * \return a expresion with type Tensor<Device,1>
 * \tparam dimkeep the dimension that will be kept
 * \tparam SrcExp expression
 * \tparam etype type of expression
 */
template<int dimkeep, typename Reducer, typename SrcExp, typename DType, int etype>
inline ReduceTo1DExp<SrcExp, DType, Reducer,
                     ExpInfo<SrcExp>::kDim - dimkeep>
reduce_except_dim(const Exp<SrcExp, DType, etype> &exp) {
  return ReduceTo1DExp<SrcExp, DType, Reducer,
                       ExpInfo<SrcExp>::kDim - dimkeep>(exp.self(), DType(1));
}
/*!
 * \brief a expression that sum over rows of a matrix
 * \param exp input expression that must be a matrix Tensor<?, 2>
 * \return a expresion with type Tensor<Device, 1>
 * \tparam SrcExp expression
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int etype>
inline ReduceTo1DExp<SrcExp, DType, red::sum, 1>
sum_rows(const Exp<SrcExp, DType, etype> &exp) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim ==2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return sumall_except_dim<1>(exp);
}
template<typename SV, typename Device, typename DType,
         typename SrcExp, typename Reducer, int m_dimkeep>
struct ExpComplexEngine<SV,
                        Tensor<Device, 1, DType>,
                        ReduceTo1DExp<SrcExp, DType, Reducer, m_dimkeep>,
                        DType> {
  static const int dimkeep = ExpInfo<SrcExp>::kDim - m_dimkeep;
  inline static void Eval(Tensor<Device, 1, DType> *dst,
                          const ReduceTo1DExp<SrcExp, DType,
                                              Reducer, m_dimkeep> &exp) {
    TypeCheckPass<m_dimkeep != 1>
        ::Error_Expression_Does_Not_Meet_Dimension_Req();
    MapReduceKeepHighDim<SV, Reducer, dimkeep>(dst, exp.src_, exp.scale_);
  }
};
template<typename SV, typename Device, typename DType,
         typename SrcExp, typename Reducer>
struct ExpComplexEngine<SV,
                        Tensor<Device, 1, DType>,
                        ReduceTo1DExp<SrcExp, DType, Reducer, 1>, DType> {
  inline static void Eval(Tensor<Device, 1, DType> *dst,
                          const ReduceTo1DExp<SrcExp, DType, Reducer, 1> &exp) {
    MapReduceKeepLowest<SV, Reducer>(dst, exp.src_, exp.scale_);
  }
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_REDUCETO1D_H_
