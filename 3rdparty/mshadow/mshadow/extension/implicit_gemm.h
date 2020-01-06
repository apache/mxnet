/*!
 *  Copyright (c) 2014 by Contributors
 * \file implicit_gemm.h
 * \brief support for implicit GEMM operation
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_IMPLICIT_GEMM_H_
#define MSHADOW_EXTENSION_IMPLICIT_GEMM_H_

#include "../extension.h"
#include "../packet-inl.h"

namespace mshadow {
namespace expr {
/*!
 * \brief Matrix multiplication.
 * \tparam LhsExp type of lhs expression
 * \tparam LhsExp type of rhs expression
 * \tparam DType the type of elements
 */
template<typename LhsExp, typename RhsExp, typename DType>
struct ImplicitGEMMExp:
      public Exp<ImplicitGEMMExp<LhsExp, RhsExp, DType>,
                 DType, type::kChainer> {
  /*! \brief lhs operand */
  const LhsExp &lhs_;
  /*! \brief rhs operand */
  const RhsExp &rhs_;
  /*! \brief internal production size*/
  index_t prod_size_;
  /*! \brief the shape of this expression */
  Shape<2> shape_;
  /*! \brief constructor */
  ImplicitGEMMExp(const LhsExp &lhs, const RhsExp &rhs)
      : lhs_(lhs), rhs_(rhs) {
    Shape<2> slhs = ShapeCheck<2, LhsExp>::Check(lhs_);
    Shape<2> srhs = ShapeCheck<2, RhsExp>::Check(rhs_);
    this->shape_ = mshadow::Shape2(slhs[0], srhs[1]);
    prod_size_ = slhs[1];
  }
};


template<typename LhsExp, typename RhsExp, typename DType, int e1, int e2>
inline ImplicitGEMMExp<LhsExp, RhsExp, DType>
implicit_dot(const Exp<LhsExp, DType, e1> &lhs,
             const Exp<RhsExp, DType, e2> &rhs) {
  TypeCheckPass<ExpInfo<LhsExp>::kDim == 2 && ExpInfo<RhsExp>::kDim == 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return ImplicitGEMMExp<LhsExp, RhsExp, DType>(lhs.self(), rhs.self());
}

//----------------------
// Execution plan
//----------------------
template<typename LhsExp, typename RhsExp, typename DType>
struct Plan<ImplicitGEMMExp<LhsExp, RhsExp, DType>, DType> {
 public:
  explicit Plan(const ImplicitGEMMExp<LhsExp, RhsExp, DType> &e)
      : lhs_(MakePlan(e.lhs_)),
        rhs_(MakePlan(e.rhs_)),
        prod_size_(e.prod_size_),
        prod_size_lower_align_(packet::LowerAlign<DType, MSHADOW_DEFAULT_PACKET>(e.prod_size_)) {
  }

  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    typedef packet::Packet<DType> Packet;
    Packet sum = Packet::Fill(0);

    const size_t packetSize = Packet::size;
    DType lhs_temp[packetSize], rhs_temp[packetSize];

    for (index_t i = 0; i < prod_size_lower_align_; i += packetSize) {
      // unroll
      for (index_t j = 0; j < packetSize; ++j) {
        lhs_temp[j] = lhs_.Eval(y, i + j);
      }
      for (index_t j = 0; j < packetSize; ++j) {
        rhs_temp[j] = rhs_.Eval(i + j, x);
      }
      sum = sum + Packet::LoadUnAligned(lhs_temp) * Packet::LoadUnAligned(rhs_temp);
    }
    DType ret_result = sum.Sum();

    for (index_t i =  prod_size_lower_align_; i < prod_size_; ++i) {
      ret_result += lhs_.Eval(y, i) * rhs_.Eval(i, x);
    }
    return ret_result;
  }

 private:
  expr::Plan<LhsExp, DType> lhs_;
  expr::Plan<RhsExp, DType> rhs_;
  const index_t prod_size_;
  const index_t prod_size_lower_align_;
};

template<typename LhsExp, typename RhsExp, typename DType>
inline Plan<ImplicitGEMMExp<LhsExp, RhsExp, DType>, DType>
MakePlan(const ImplicitGEMMExp<LhsExp, RhsExp, DType> &exp) {
  return Plan<ImplicitGEMMExp<LhsExp, RhsExp, DType>, DType>(exp);
}


template<int dim, typename LhsExp, typename RhsExp, typename DType>
struct ShapeCheck<dim, ImplicitGEMMExp<LhsExp, RhsExp, DType> > {
  inline static Shape<dim>
  Check(const ImplicitGEMMExp<LhsExp, RhsExp, DType> &t) {
    CHECK(dim == 2)
        << "ImplicitGEMMExp only support 2 dimension";
    Shape<dim> shape1 = ShapeCheck<dim, LhsExp>::Check(t.lhs_);
    Shape<dim> shape2 = ShapeCheck<dim, RhsExp>::Check(t.rhs_);
    CHECK_EQ(shape1[1], shape2[0])
      << "implicit_dot The matrix shape do  not match";
    return t.shape_;
  }
};

template<typename LhsExp, typename RhsExp, typename DType>
struct ExpInfo<ImplicitGEMMExp<LhsExp, RhsExp, DType> > {
  static const int kDim = 2;
  static const int kDevMask = ExpInfo<LhsExp>::kDevMask & ExpInfo<RhsExp>::kDevMask;
};

}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_IMPLICIT_GEMM_H_

