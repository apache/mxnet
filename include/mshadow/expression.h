/*!
 *  Copyright (c) 2014 by Contributors
 * \file expression.h
 * \brief definitions of abstract expressions and expressions template
 * \author Tianqi Chen, Bing Xu
 */
#ifndef MSHADOW_EXPRESSION_H_
#define MSHADOW_EXPRESSION_H_
#include "./base.h"

namespace mshadow {
/*!
 * \brief namespace for abstract expressions and expressions template,
 *        have no dependency on tensor.h,
 *        These data structure takes no charge in computations,
 *        they are only used to define operations and represent expression in a symbolic way
 */
namespace expr {
/*! \brief type of expressions */
namespace type {
// type expression type are defined as bitmask
// subtype relationshop kRValue < kMapper < kPull < kComplex
/*!
 * \brief this expression directly correspnds to a data class,
 *   can be used to assign data
 */
const int kRValue = 0;
/*!
 * \brief expression contains element-wise tensor operations,
 *   map a expression to same shape
 */
const int kMapper = 1;
/*!
 * \brief expression that can be chained with other expressiones
 *    Usually it have function Eval(i,j) defined, which pulls the result (i, j) from input
 *    expression and output the result at certain position.
 */
const int kChainer = 3;
/*! \brief othercase: e.g dot product */
const int kComplex = 7;
}  // namespace type
/*!
 * \brief expression engine that actually interprets these expressions
 *   this is a function template that needed to be implemented for specific expressions
 * \tparam Saver the save method
 * \tparam RValue the type of RValue to be saved
 * \sa namespace sv
 */
template<typename Saver, typename RValue, typename DType>
struct ExpEngine;
/*! \brief defines how expression exp can be evaluated and stored into dst */
// template<typename EType>
// inline static void Eval(RValue *dst, const EType &exp);
/*!
 * \brief base class for expression
 * \tparam SubType inheritated class must put their type into this parameter
 * \tparam DType the data type of each element in the expression
 * \tparam exp_type expression type, see namespace type
 */
template<typename SubType, typename DType, int exp_type>
struct Exp {
 public:
  /*! \return  subtype instance of current class */
  inline const SubType& self(void) const {
    return *static_cast<const SubType*>(this);
  }
  /*! \return reference of subtype instance of current class */
  inline SubType* ptrself(void) {
    return static_cast<SubType*>(this);
  }
};
/*!
 * \brief scalar expression
 * \tparam DType the data type of the scalar
 */
template<typename DType>
struct ScalarExp: public Exp<ScalarExp<DType>, DType, type::kMapper> {
  /*! \brief scalar value */
  DType scalar_;
  /*! \brief implicit constructor, MUST NOT BE explicit */
  ScalarExp(DType scalar) : scalar_(scalar) {}  // NOLINT(*)
};
/*! \brief create an scalar expression */
template<typename DType>
inline ScalarExp<DType> scalar(DType s) {
  return ScalarExp<DType>(s);
}
/*!
 * \brief typecast expression, cast the type of elements
 * \tparam DstDType the target type we want to cast into
 * \tparam SrcDType the target type we want to cast from
 * \tparam EType the type of the source expression
 * \tparam etype the type of expression after cast
 */
template<typename DstDType, typename SrcDType, typename EType, int etype>
struct TypecastExp:
      public Exp<TypecastExp<DstDType, SrcDType, EType, etype>,
                 DstDType, etype> {
  /*! \brief expression to be typecasted */
  const EType &exp;
  /*! \brief constructor */
  explicit TypecastExp(const EType &e) : exp(e) {}
};
/*! \brief create an scalar expression */
template<typename DstDType, typename SrcDType,
         typename EType, int etype>
inline TypecastExp<DstDType, SrcDType, EType, (etype|type::kMapper)>
tcast(const Exp<EType, SrcDType, etype> &exp) {
  return TypecastExp<DstDType, SrcDType, EType, (etype|type::kMapper)>(exp.self());
}
/*! \brief represent a transpose expression of a container */
template<typename EType, typename DType>
struct TransposeExp: public Exp<TransposeExp<EType, DType>,
                                DType, type::kChainer> {
  /*! \brief expression to be transposed */
  const EType &exp;
  /*! \brief constructor */
  explicit TransposeExp(const EType &e) : exp(e) {}
  /*! \brief transpose expression */
  inline const EType &T(void) const {
    return exp;
  }
};
/*!
 * \brief base class of all rvalues
 * \tparam Container the actually class of data container, e.g. Tensor1D
 * \tparam DataType the element data type of each element in the container
 */
template<typename Container, typename DType>
class RValueExp: public Exp<Container, DType, type::kRValue> {
 public:
  /*!
   *\brief transpose of a matrix
   *\return transpose of current expression
   */
  inline const TransposeExp<Container, DType> T(void) const {
    return TransposeExp<Container, DType>(this->self());
  }
  /*! \brief operator overload */
  inline Container &operator+=(DType s) {
    ExpEngine<sv::plusto, Container, DType>::Eval(this->ptrself(), scalar<DType>(s));
    return *(this->ptrself());
  }
  /*! \brief operator overload */
  inline Container &operator-=(DType s) {
    ExpEngine<sv::minusto, Container, DType>::Eval(this->ptrself(), scalar<DType>(s));
    return *(this->ptrself());
  }
  /*! \brief operator overload */
  inline Container &operator*=(DType s) {
    ExpEngine<sv::multo, Container, DType>::Eval(this->ptrself(), scalar<DType>(s));
    return *(this->ptrself());
  }
  /*! \brief operator overload */
  inline Container &operator/=(DType s) {
    ExpEngine<sv::divto, Container, DType>::Eval(this->ptrself(), scalar<DType>(s));
    return *(this->ptrself());
  }
  /*! \brief operator overload */
  inline Container &__assign(DType s) {
    ExpEngine<sv::saveto, Container, DType>::Eval(this->ptrself(), scalar<DType>(s));
    return *(this->ptrself());
  }
  /*! \brief  we can not define container = container */
  template<typename E, int etype>
  inline Container &__assign(const Exp<E, DType, etype> &exp) {
    ExpEngine<sv::saveto, Container, DType>::Eval(this->ptrself(), exp.self());
    return *(this->ptrself());
  }
  /*! \brief operator overload, assign */
  inline Container &__assign(const Exp<Container, DType, type::kRValue> &exp);
  /*! \brief implementation of operator+= */
  template<typename E, int etype>
  inline Container &operator+=(const Exp<E, DType, etype> &exp) {
    ExpEngine<sv::plusto, Container, DType>::Eval(this->ptrself(), exp.self());
    return *(this->ptrself());
  }
  /*! \brief implementation of operator-= */
  template<typename E, int etype>
  inline Container &operator-=(const Exp<E, DType, etype> &exp) {
    ExpEngine<sv::minusto, Container, DType>::Eval(this->ptrself(), exp.self());
    return *(this->ptrself());
  }
  /*! \brief implementation of operator*= */
  template<typename E, int etype>
  inline Container &operator*=(const Exp<E, DType, etype> &exp) {
    ExpEngine<sv::multo, Container, DType>::Eval(this->ptrself(), exp.self());
    return *(this->ptrself());
  }
  /*! \brief implementation of operator/= */
  template<typename E, int etype>
  inline Container &operator/=(const Exp<E, DType, etype> &exp) {
    ExpEngine<sv::divto, Container, DType>::Eval(this->ptrself(), exp.self());
    return *(this->ptrself());
  }
};
/*!
 * \brief matrix multiplication expression dot(lhs[.T], rhs[.T])
 * \tparam TA type of lhs
 * \tparam TB type of rhs
 * \tparam ltrans whether lhs is transposed
 * \tparam rtrans whether rhs is transposed
 * \tparam DType the data type of the scalar
 */
template<typename TA, typename TB, bool ltrans, bool rtrans, typename DType>
struct DotExp: public Exp<DotExp<TA, TB, ltrans, rtrans, DType>,
                          DType, type::kComplex> {
  /*! \brief left operand */
  const TA &lhs_;
  /*! \brief right operand */
  const TB &rhs_;
  /*! \brief scale over result */
  DType scale_;
  /*! \brief constructor */
  explicit DotExp(const TA &lhs, const TB &rhs, DType scale)
      : lhs_(lhs), rhs_(rhs), scale_(scale) {}
};
// definition of dot expression
/*! \brief dot operator def */
template<typename TA, typename TB, typename DType>
inline DotExp<TA, TB, false, false, DType>
dot(const RValueExp<TA, DType> &lhs, const RValueExp<TB, DType> &rhs) {
  return DotExp<TA, TB, false, false, DType>(lhs.self(), rhs.self(), DType(1.0f));
}
/*! \brief dot operator def */
template<typename TA, typename TB, typename DType>
inline DotExp<TA, TB, true, false, DType>
dot(const TransposeExp<TA, DType> &lhs, const RValueExp<TB, DType> &rhs) {
  return DotExp<TA, TB, true, false, DType>(lhs.exp, rhs.self(), DType(1.0f));
}
/*! \brief dot operator def */
template<typename TA, typename TB, typename DType>
inline DotExp<TA, TB, false, true, DType>
dot(const RValueExp<TA, DType> &lhs, const TransposeExp<TB, DType> &rhs) {
  return DotExp<TA, TB, false, true, DType>(lhs.self(), rhs.exp, DType(1.0f));
}
/*! \brief dot operator def */
template<typename TA, typename TB, typename DType>
inline DotExp<TA, TB, true, true, DType>
dot(const TransposeExp<TA, DType> &lhs, const TransposeExp<TB, DType> &rhs) {
  return DotExp<TA, TB, true, true, DType>(lhs.exp, rhs.exp, DType(1.0f));
}
/*! \brief batch_dot operator def */
template<bool transpose_left, bool transpose_right, typename TA, typename TB, typename DType>
inline DotExp<TA, TB, transpose_left, transpose_right, DType>
batch_dot(const RValueExp<TA, DType> &lhs, const RValueExp<TB, DType> &rhs) {
  return DotExp<TA, TB, transpose_left, transpose_right, DType>(
    lhs.self(), rhs.self(), DType(1.0f));
}
//---------------
// TernaryMapExp
// --------------
/*!
 * \brief ternary map expression
 * \tparam OP operator
 * \tparam TA type of item1
 * \tparam TB type of item2
 * \tparam etype expression type, sa namespace::type
 */
template<typename OP, typename TA, typename TB, typename TC, typename DType, int etype>
struct TernaryMapExp: public Exp<TernaryMapExp<OP, TA, TB, TC, DType, etype>,
                                DType, etype> {
  /*! \brief first operand */
  const TA &item1_;
  /*! \brief second operand */
  const TB &item2_;
  /*! \brief third  operand */
  const TC &item3_;
  /*! \brief constructor */
  explicit TernaryMapExp(const TA &item1, const TB &item2, const TC &item3)
      :item1_(item1), item2_(item2), item3_(item3) {}
};

/*! \brief make expression */
template<typename OP, typename TA, typename TB, typename TC, typename DType, int ta, int tb, int tc>
inline TernaryMapExp<OP, TA, TB, TC, DType, (ta|tb|tc|type::kMapper)>
MakeExp(const Exp<TA, DType, ta> &item1, const Exp<TB, DType, tb> &item2,
 const Exp<TC, DType, tc> &item3) {
  return TernaryMapExp<OP, TA, TB, TC, DType,
                      (ta|tb|tc|type::kMapper)>(item1.self(), item2.self(), item3.self());
}
/*!
 * \brief short hand for MakeExp, usage F<op>(item1,item2,item3). create a ternary operation expression
 * \param item1 first operand
 * \param item2 second operand
 * \param item3 third operand
 * \return the result expression
 * \tparam ternary operator
 * \tparam TA item1 expression
 * \tparam ta item1 expression type
 * \tparam TB item2 expression
 * \tparam tb item2 expression type
 * \tparam TC item3 expression
 * \tparam tc item3 expression type
 * \sa mshadow::op
 */

// Ternary
template<typename OP, typename TA, typename TB, typename TC, typename DType, int ta, int tb, int tc>
inline TernaryMapExp<OP, TA, TB, TC, DType, (ta|tb|tc|type::kMapper)>
F(const Exp<TA, DType, ta> &item1, const Exp<TB, DType, tb> &item2,
 const Exp<TC, DType, tc> &item3) {
  return MakeExp<OP>(item1, item2, item3);
}
//---------------
// BinaryMapExp
// --------------
/*!
 * \brief binary map expression lhs [op] rhs
 * \tparam OP operator
 * \tparam TA type of lhs
 * \tparam TB type of rhs
 * \tparam etype expression type, sa namespace::type
 */
template<typename OP, typename TA, typename TB, typename DType, int etype>
struct BinaryMapExp: public Exp<BinaryMapExp<OP, TA, TB, DType, etype>,
                                DType, etype> {
  /*! \brief left operand */
  const TA &lhs_;
  /*! \brief right operand */
  const TB &rhs_;
  /*! \brief constructor */
  explicit BinaryMapExp(const TA &lhs, const TB &rhs)
      :lhs_(lhs), rhs_(rhs) {}
};

/*! \brief make expression */
template<typename OP, typename TA, typename TB, typename DType, int ta, int tb>
inline BinaryMapExp<OP, TA, TB, DType, (ta|tb|type::kMapper)>
MakeExp(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return BinaryMapExp<OP, TA, TB, DType,
                      (ta|tb|type::kMapper)>(lhs.self(), rhs.self());
}
/*!
 * \brief short hand for MakeExp, usage F<op>(lhs, rhs). create a binary operation expression
 * \param lhs left operand
 * \param rhs right operand
 * \return the result expression
 * \tparam binary operator
 * \tparam TA lhs expression
 * \tparam ta lhs expression type
 * \tparam TB rhs expression
 * \tparam tb rhs expression type
 * \sa mshadow::op
 */
template<typename OP, typename TA, typename TB, typename DType, int ta, int tb>
inline BinaryMapExp<OP, TA, TB, DType, (ta|tb|type::kMapper)>
F(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return MakeExp<OP>(lhs, rhs);
}
// operator rules
/*! \brief operator overload */
template<typename TA, typename TB, typename DType, int ta, int tb>
inline BinaryMapExp<op::plus, TA, TB, DType, (ta|tb|type::kMapper)>
operator+(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return MakeExp<op::plus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, typename TB, typename DType, int ta, int tb>
inline BinaryMapExp<op::minus, TA, TB, DType, (ta|tb|type::kMapper)>
operator-(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return MakeExp<op::minus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, typename TB, typename DType, int ta, int tb>
inline BinaryMapExp<op::mul, TA, TB, DType, (ta|tb|type::kMapper)>
operator*(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return MakeExp<op::mul>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, typename TB, typename DType, int ta, int tb>
inline BinaryMapExp<op::div, TA, TB, DType, (ta|tb|type::kMapper)>
operator/(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return MakeExp<op::div>(lhs, rhs);
}
//---------------
// UnaryMapExp
// --------------
/*!
 * \brief unary map expression op(src)
 * \tparam OP operator
 * \tparam TA type of src
 * \tparam etype expression type, sa namespace::type
 */
template<typename OP, typename TA, typename DType, int etype>
struct UnaryMapExp: public Exp<UnaryMapExp<OP, TA, DType, etype>,
                               DType, etype> {
  /*! \brief source expression */
  const TA &src_;
  /*! \brief constructor */
  explicit UnaryMapExp(const TA &src) : src_(src) {}
};

/*! \brief make expression */
template<typename OP, typename TA, typename DType, int ta>
inline UnaryMapExp<OP, TA, DType, (ta|type::kMapper)>
MakeExp(const Exp<TA, DType, ta> &src) {
  return UnaryMapExp<OP, TA, DType, (ta|type::kMapper)>(src.self());
}
/*!
 * \brief short hand for MakeExp, usage F<op>(src), create a unary operation expression
 * \param src source expression
 * \return the result expression
 * \tparam operator
 * \tparam TA source expression
 * \tparam ta source expression type
 * \sa mshadow::op
 */
template<typename OP, typename TA, typename DType, int ta>
inline UnaryMapExp<OP, TA, DType, (ta|type::kMapper)>
F(const Exp<TA, DType, ta> &src) {
  return MakeExp<OP>(src);
}
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXPRESSION_H_
