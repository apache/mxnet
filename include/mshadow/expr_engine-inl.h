/*!
 *  Copyright (c) 2014 by Contributors
 * \file expr_engine-inl.h
 * \brief definitions of how expressions should be evaluated
 * \author Tianqi Chen, Bing Xu
 */
#ifndef MSHADOW_EXPR_ENGINE_INL_H_
#define MSHADOW_EXPR_ENGINE_INL_H_
#include <utility>
#include <algorithm>
#include "./logging.h"
#include "./expression.h"
#include "./tensor.h"

namespace mshadow {
namespace expr {
/*!
 * \brief a general class that allows extension that makes tensors of some shape
 * \tparam SubType type of subclass
 * \tparam SrcExp source expression of the MakeTensorExp, the source of operation
 * \tparam dim dimension of the expression
 * \tparam DType the type of elements
 */
template<typename SubType, typename SrcExp, int dim, typename DType>
struct MakeTensorExp
    : public Exp<MakeTensorExp<SubType, SrcExp, dim, DType>,
                 DType, type::kChainer> {
  /*! \brief the shape of this expression */
  Shape<dim> shape_;
  /*! \brief true self of subtype */
  inline const SubType& real_self(void) const{
    return *static_cast<const SubType*>(this);
  }
};
//----------------------------------------------------------------------
// This part of code gives plan that can be used to carry out execution
//---------------------------------------------------------------------
// Declarations of plans
template<typename ExpType, typename DType>
class Plan {
 public:
  /*!
   * \brief evaluate the expression at index [y][x]
   *  to be implemented by SubType, for RValue, the return type will be DType &
   */
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const;
};
// tensor plan
template <typename Device, int dim, typename DType>
class Plan<Tensor<Device, dim, DType>, DType> {
 public:
  explicit Plan(const Tensor<Device, dim, DType> &t)
      : dptr_(t.dptr_), stride_(t.stride_) {}
  // for RValue, the return type should be reference
  MSHADOW_XINLINE DType &REval(index_t y, index_t x) {
    return dptr_[y * stride_ + x];
  }
  // const evaluation
  MSHADOW_XINLINE const DType &Eval(index_t y, index_t x) const {
    return dptr_[y * stride_ + x];
  }

 private:
  DType  *dptr_;
  index_t stride_;
};
// special evaluation case for 1d tensor, no stride
template <typename Device, typename DType>
class Plan<Tensor<Device, 1, DType>, DType> {
 public:
  explicit Plan(const Tensor<Device, 1, DType> &t) : dptr_(t.dptr_) {}
  MSHADOW_XINLINE DType &REval(index_t y, index_t x) {
    return dptr_[x];
  }
  MSHADOW_XINLINE const DType &Eval(index_t y, index_t x) const {
    return dptr_[x];
  }

 private:
  DType  *dptr_;
};
// scalar
template<typename DType>
class Plan<ScalarExp<DType>, DType> {
 public:
  explicit Plan(DType scalar) : scalar_(scalar) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return scalar_;
  }

 private:
  DType scalar_;
};
// unary expression
template<typename DstDType, typename SrcDType,
         typename EType, int etype>
class Plan<TypecastExp<DstDType, SrcDType, EType, etype>, DstDType> {
 public:
  explicit Plan(const Plan<EType, SrcDType> &src) : src_(src) {}
  MSHADOW_XINLINE DstDType Eval(index_t y, index_t x) const {
    return DstDType(src_.Eval(y, x));  // NOLINT(*)
  }

 private:
  Plan<EType, SrcDType> src_;
};

// ternary expression
template<typename OP, typename TA, typename TB, typename TC, int etype, typename DType>
class Plan<TernaryMapExp<OP, TA, TB, TC, DType, etype>, DType> {
 public:
  explicit Plan(const Plan<TA, DType> &item1, const Plan<TB, DType> &item2,
       const Plan<TC, DType> &item3)
      : item1_(item1), item2_(item2), item3_(item3) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return OP::Map(item1_.Eval(y, x), item2_.Eval(y, x), item3_.Eval(y, x));
  }

 private:
  Plan<TA, DType> item1_;
  Plan<TB, DType> item2_;
  Plan<TC, DType> item3_;
};
// binary expression
template<typename OP, typename TA, typename TB, int etype, typename DType>
class Plan<BinaryMapExp<OP, TA, TB, DType, etype>, DType> {
 public:
  explicit Plan(const Plan<TA, DType> &lhs, const Plan<TB, DType> &rhs)
      : lhs_(lhs), rhs_(rhs) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return OP::Map(lhs_.Eval(y, x), rhs_.Eval(y, x));
  }

 private:
  Plan<TA, DType> lhs_;
  Plan<TB, DType> rhs_;
};
// unary expression
template<typename OP, typename TA, int etype, typename DType>
class Plan<UnaryMapExp<OP, TA, DType, etype>, DType> {
 public:
  explicit Plan(const Plan<TA, DType> &src) : src_(src) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return OP::Map(src_.Eval(y, x));
  }

 private:
  Plan<TA, DType> src_;
};
// remaps map tensor expression to subtype's plan
template<typename SubType, typename SrcExp, int dim, typename DType>
struct Plan<MakeTensorExp<SubType, SrcExp, dim, DType>, DType> {
 public:
  Plan(const Plan<SubType, DType> &src) : src_(src) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return src_.Eval(y, x);
  }

 private:
  Plan<SubType, DType> src_;
};
// tranpsoe
template<typename EType, typename DType>
class Plan<TransposeExp<EType, DType>, DType> {
 public:
  explicit Plan(const Plan<EType, DType> &src) : src_(src) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return src_.Eval(x, y);
  }

 private:
  Plan<EType, DType> src_;
};
//----------------------------------------------------------------------
// Mappings from expression to plans
//---------------------------------------------------------------------
template<typename OP, typename TA, typename TB, typename DType, int etype>
inline Plan<BinaryMapExp<OP, TA, TB, DType, etype>, DType>
MakePlan(const BinaryMapExp<OP, TA, TB, DType, etype> &e);

template<typename OP, typename TA, typename TB, typename TC, typename DType, int etype>
inline Plan<TernaryMapExp<OP, TA, TB, TC, DType, etype>, DType>
MakePlan(const TernaryMapExp<OP, TA, TB, TC, DType, etype> &e);

template<typename DType>
inline Plan<ScalarExp<DType>, DType> MakePlan(const ScalarExp<DType> &e) {
  return Plan<ScalarExp<DType>, DType>(e.scalar_);
}

template<typename DstDType, typename SrcDType, typename EType, int etype>
inline Plan<TypecastExp<DstDType, SrcDType, EType, etype>, DstDType>
MakePlan(const TypecastExp<DstDType, SrcDType, EType, etype> &e) {
  return Plan<TypecastExp<DstDType, SrcDType, EType, etype>, DstDType>(MakePlan(e.exp));
}

template<typename T, typename DType>
inline Plan<T, DType> MakePlan(const RValueExp<T, DType> &e) {
  return Plan<T, DType>(e.self());
}

template<typename T, typename DType>
inline Plan<TransposeExp<T, DType>, DType>
MakePlan(const TransposeExp<T, DType> &e) {
  return Plan<TransposeExp<T, DType>, DType>(MakePlan(e.exp));
}

template<typename T, typename SrcExp, int dim, typename DType>
inline Plan<T, DType>
MakePlan(const MakeTensorExp<T, SrcExp, dim, DType> &e) {
  return Plan<T, DType>(e.real_self());
}

template<typename OP, typename TA, typename DType, int etype>
inline Plan<UnaryMapExp<OP, TA, DType, etype>, DType>
MakePlan(const UnaryMapExp<OP, TA, DType, etype> &e) {
  return Plan<UnaryMapExp<OP, TA, DType, etype>, DType>(MakePlan(e.src_));
}

template<typename OP, typename TA, typename TB, typename DType, int etype>
inline Plan<BinaryMapExp<OP, TA, TB, DType, etype>, DType>
MakePlan(const BinaryMapExp<OP, TA, TB, DType, etype> &e) {
  return Plan<BinaryMapExp<OP, TA, TB, DType, etype>,
              DType>(MakePlan(e.lhs_), MakePlan(e.rhs_));
}

// Ternary
template<typename OP, typename TA, typename TB, typename TC, typename DType, int etype>
inline Plan<TernaryMapExp<OP, TA, TB, TC, DType, etype>, DType>
MakePlan(const TernaryMapExp<OP, TA, TB, TC, DType, etype> &e) {
  return Plan<TernaryMapExp<OP, TA, TB, TC, DType, etype>,
              DType>(MakePlan(e.item1_), MakePlan(e.item2_), MakePlan(e.item3_));
}
//----------------------------------------------------------------
// Static Type inference and Type Checking
//----------------------------------------------------------------
/*!
 * \brief static type inference template,
 *        used to get the dimension of each expression,
 *        if ExpInfo<E>::kDim == -1, this means here are mismatch in expression
 *        if (ExpInfo<E>::kDevMask & cpu::kDevMask) != 0, this means this expression can be assigned to cpu
 * \tparam E expression
 */
template<typename E>
struct ExpInfo {
  static const int kDim = -1;
  static const int kDevMask = 0;
};
template<typename DType>
struct ExpInfo< ScalarExp<DType> > {
  static const int kDim = 0;
  static const int kDevMask = 0xffff;
};
template<typename E, typename DType>
struct ExpInfo<TransposeExp<E, DType> > {
  static const int kDim = ExpInfo<E>::kDim;
  static const int kDevMask = ExpInfo<E>::kDevMask;
};
template<typename DstDType, typename SrcDType, typename EType, int etype>
struct ExpInfo<TypecastExp<DstDType, SrcDType, EType, etype> > {
  static const int kDim = ExpInfo<EType>::kDim;
  static const int kDevMask = ExpInfo<EType>::kDevMask;
};
template<typename Device, int dim, typename DType>
struct ExpInfo<Tensor<Device, dim, DType> > {
  static const int kDim = dim;
  static const int kDevMask = Device::kDevMask;
};
template<typename T, typename SrcExp, int dim, typename DType>
struct ExpInfo<MakeTensorExp<T, SrcExp, dim, DType> > {
  static const int kDimSrc = ExpInfo<SrcExp>::kDim;
  static const int kDim = kDimSrc >= 0 ? dim : -1;
  static const int kDevMask = ExpInfo<SrcExp>::kDevMask;
};
template<typename OP, typename TA, typename DType, int etype>
struct ExpInfo<UnaryMapExp<OP, TA, DType, etype> > {
  static const int kDim = ExpInfo<TA>::kDim;
  static const int kDevMask = ExpInfo<TA>::kDevMask;
};
template<typename OP, typename TA, typename TB, typename DType, int etype>
struct ExpInfo<BinaryMapExp<OP, TA, TB, DType, etype> > {
  static const int kDimLhs = ExpInfo<TA>::kDim;
  static const int kDimRhs = ExpInfo<TB>::kDim;
  static const int kDim = (kDimLhs >= 0 && kDimRhs >= 0) ?\
      (kDimLhs == 0 ?\
       kDimRhs :\
       ((kDimRhs == 0 || kDimLhs == kDimRhs) ? kDimLhs : -1)) : -1;
  static const int kDevMask = ExpInfo<TA>::kDevMask & ExpInfo<TB>::kDevMask;
};
template<typename OP, typename TA, typename TB, typename TC, typename DType, int etype>
struct ExpInfo<TernaryMapExp<OP, TA, TB, TC, DType, etype> > {
  static const int kDimItem1 = ExpInfo<TA>::kDim;
  static const int kDimItem2 = ExpInfo<TB>::kDim;
  static const int kDimItem3 = ExpInfo<TC>::kDim;
  static const int kDim = kDimItem1;
  static const int kDevMask = ExpInfo<TA>::kDevMask & ExpInfo<TB>::kDevMask & ExpInfo<TC>::kDevMask;
};

/*! \brief template to do type check */
template<typename Device, int dim, typename DType, typename E>
struct TypeCheck {
  /*! \brief dimension of expression*/
  static const int kExpDim = ExpInfo<E>::kDim;
  /*! \brief whether the expression device type matches */
  static const bool kDevPass = (ExpInfo<E>::kDevMask & Device::kDevMask) != 0;
  /*! \brief whether the expression can be mapped to expression of dim */
  static const bool kMapPass = (kExpDim == 0 || kExpDim == dim) && kDevPass;
  /*! \brief whether the expression can be reduced to expression of dim */
  static const bool kRedPass = (kExpDim > dim) && kDevPass;
};
/*! \brief used to help static type check*/
template<bool kPass>
struct TypeCheckPass;
// Todo : add static assert using C++11
template<>
struct TypeCheckPass<false> {};
template<>
struct TypeCheckPass<true> {
  inline static void Error_All_Tensor_in_Exp_Must_Have_Same_Type(void) {}
  inline static void Error_TypeCheck_Not_Pass_For_Reduce_Exp(void) {}
  inline static void Error_Expression_Does_Not_Meet_Dimension_Req(void) {}
};

//----------------------------------------------------------------
// Runtime Stream Getting
//----------------------------------------------------------------
template<typename Device, typename E>
struct StreamInfo {
  inline static Stream<Device> *Get(const E &t);
};
template<int dim, typename Device, typename DType>
struct StreamInfo<Device, Tensor<Device, dim, DType> > {
  inline static Stream<Device> *Get(const Tensor<Device, dim, DType> &t) {
    return t.stream_;
  }
};
//----------------------------------------------------------------
// Runtime Shape Checking
//----------------------------------------------------------------
/*!
 * \brief runtime shape checking template
 *    get the shape of an expression, report error if shape mismatch
 * \tparam dim the dimension of the shape
 * \tparam E expression
 */
template<int dim, typename E>
struct ShapeCheck {
  inline static Shape<dim> Check(const E &t);
};
template<int dim, typename DType>
struct ShapeCheck<dim, ScalarExp<DType> > {
  inline static Shape<dim> Check(const ScalarExp<DType> &exp) {
    // use lowest dimension to mark scalar exp
    Shape<dim> shape;
    for (int i = 0; i < dim; ++i) {
      shape[i] = 0;
    }
    return shape;
  }
};
template<int dim, typename DstDType, typename SrcDType, typename EType, int etype>
struct ShapeCheck<dim, TypecastExp<DstDType, SrcDType, EType, etype> > {
  inline static Shape<dim>
  Check(const TypecastExp<DstDType, SrcDType, EType, etype> &exp) {
    return ShapeCheck<dim, EType>::Check(exp.exp);
  }
};
template<int dim, typename E, typename DType>
struct ShapeCheck<dim, TransposeExp<E, DType> > {
  inline static Shape<dim> Check(const TransposeExp<E, DType> &e) {
    // swap the lowest two dimensions
    Shape<dim> s = ShapeCheck<dim, E>::Check(e.exp);
    std::swap(s[0], s[1]);
    return s;
  }
};
template<int dim, typename Device, typename DType>
struct ShapeCheck<dim, Tensor<Device, dim, DType> > {
  inline static Shape<dim> Check(const Tensor<Device, dim, DType> &t) {
    return t.shape_;
  }
};
template<int dim, typename SrcExp, typename T, typename DType>
struct ShapeCheck<dim, MakeTensorExp<T, SrcExp, dim, DType> > {
  inline static Shape<dim>
  Check(const MakeTensorExp<T, SrcExp, dim, DType> &t) {
    return t.shape_;
  }
};
template<int dim, typename OP, typename TA, typename DType, int etype>
struct ShapeCheck<dim, UnaryMapExp<OP, TA, DType, etype> > {
  inline static Shape<dim> Check(const UnaryMapExp<OP, TA, DType, etype> &t) {
    Shape<dim> s = ShapeCheck<dim, TA>::Check(t.src_);
    return s;
  }
};

template<int dim, typename OP, typename TA, typename TB,
         typename DType, int etype>
struct ShapeCheck<dim, BinaryMapExp<OP, TA, TB, DType, etype> > {
  inline static Shape<dim>
  Check(const BinaryMapExp<OP, TA, TB, DType, etype> &t) {
    Shape<dim> shape1 = ShapeCheck<dim, TA>::Check(t.lhs_);
    Shape<dim> shape2 = ShapeCheck<dim, TB>::Check(t.rhs_);
    if (shape1[0] == 0) return shape2;
    if (shape2[0] == 0) return shape1;
    CHECK_EQ(shape1, shape2) << "BinaryMapExp: Shapes of operands are not the same, " <<
      "Shape1=" << shape1 << ", Shape2=" << shape2;
    return shape1;
  }
};

template<int dim, typename OP, typename TA, typename TB, typename TC,
         typename DType, int etype>
struct ShapeCheck<dim, TernaryMapExp<OP, TA, TB, TC, DType, etype> > {
  inline static Shape<dim>
  Check(const TernaryMapExp<OP, TA, TB, TC, DType, etype> &t) {
    Shape<dim> shape1 = ShapeCheck<dim, TA>::Check(t.item1_);
    Shape<dim> shape2 = ShapeCheck<dim, TB>::Check(t.item2_);
    Shape<dim> shape3 = ShapeCheck<dim, TC>::Check(t.item3_);
    bool same = (shape1 == shape2) && (shape2 == shape3);
    CHECK(same) << "TernaryMapExp: Shapes of operands are not the same, " <<
      "Shape1=" << shape1 << ", Shape2=" << shape2 << ", Shape3=" << shape3;

    return shape1;
  }
};
}  // namespace expr

}  // namespace mshadow
// include definition of dot engine
#include "./dot_engine-inl.h"

namespace mshadow {
namespace expr {
/*! \brief some engine that evaluate complex expression */
template<typename SV, typename RV, typename E, typename DType>
struct ExpComplexEngine {
  inline static void Eval(RV *dst, const E &exp);
};
/*! \brief the engine that dispatches simple operations*/
template<typename SV, typename RV, typename DType>
struct ExpEngine {
  template<typename E>
  inline static void Eval(RV *dst,
                          const Exp<E, DType, type::kMapper> &exp) {
    MapExp<SV>(dst, exp);
  }
  template<typename E>
  inline static void Eval(RV *dst,
                          const Exp<E, DType, type::kChainer> &exp) {
    MapExp<SV>(dst, exp);
  }
  template<typename E>
  inline static void Eval(RV *dst,
                          const Exp<E, DType, type::kRValue> &exp) {
    MapExp<SV>(dst, exp);
  }
  template<typename E>
  inline static void Eval(RV *dst,
                          const Exp<E, DType, type::kComplex> &exp) {
    ExpComplexEngine<SV, RV, E, DType>::Eval(dst->ptrself(), exp.self());
  }
};
template<typename SV, typename Device, int dim, int ldim,
         int rdim, bool ltrans, bool rtrans, typename DType>
struct ExpComplexEngine<SV,
                        Tensor<Device, dim, DType>,
                        DotExp<Tensor<Device, ldim, DType>,
                               Tensor<Device, rdim, DType>,
                               ltrans, rtrans, DType>,
                        DType> {
  inline static void Eval(Tensor<Device, dim, DType> *dst,
                          const DotExp<Tensor<Device, ldim, DType>,
                                       Tensor<Device, rdim, DType>,
                                       ltrans, rtrans, DType> &exp) {
    DotEngine<SV, Device, dim, ldim, rdim,
              ltrans, rtrans, DType>::Eval(dst, exp.lhs_, exp.rhs_, exp.scale_);
  }
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXPR_ENGINE_INL_H_
