/*!
 *  Copyright (c) 2014 by Contributors
 * \file broadcast.h
 * \brief support for broadcast and repmat
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_BROADCAST_H_
#define MSHADOW_EXTENSION_BROADCAST_H_
#include "../extension.h"
namespace mshadow {
namespace expr {
/*!
 * \brief broadcast Tensor1D into a higher dimension Tensor
 * input: Tensor<Device,1>: ishape[0]
 * output: Tensor<Device,dimdst> : oshape[dimcast] = ishape[0]
 * \tparam SrcExp type of input expression
 * \tparam DType the type of elements
 * \tparam dimdst  target tensor dimension
 * \tparam dimcast_m_dst  dimdst - dimcast
 */
template<typename SrcExp, typename DType, int dimdst, int dimdst_m_cast>
struct Broadcast1DExp:
      public MakeTensorExp<Broadcast1DExp<SrcExp, DType, dimdst, dimdst_m_cast>,
                           SrcExp, dimdst, DType> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief constructor */
  Broadcast1DExp(const SrcExp &src, Shape<dimdst> shape)
      : src_(src) {
    this->shape_ = shape;
  }
};

/*!
 * \brief broadcast scalar into a higher dimension Tensor
 * input: Tensor<Device,1>: ishape = {1}
 * output: Tensor<Device, dimdst> : oshape[dimcast] = ishape[0]
 * \tparam SrcExp type of input expression
 * \tparam DType the type of elements
 * \tparam dimdst  target tensor dimension
 */
template<typename SrcExp, typename DType, int dimdst>
struct BroadcastScalarExp:
      public MakeTensorExp<BroadcastScalarExp<SrcExp, DType, dimdst>,
                           SrcExp, dimdst, DType> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief constructor */
  BroadcastScalarExp(const SrcExp &src, Shape<dimdst> shape)
      : src_(src) {
    this->shape_ = shape;
  }
};

/*!
 * \brief a expression that replicate a 1 dimension tensor in dimension dimcast
 * \param src Tensor<Device,1>: shape[0]
 * \param shape shape of output
 * \return a expresion with type Tensor<Device,dimdst>
 * \tparam dimcast target dimension where the 1D tensor will be broadcasted
 * \tparam SrcExp type of input expression
 * \tparam DType the type of elements
 * \tparam dimdst dimension of destination tensor
 * \tparam dimcast_lowest the dimension we want to cast the data into
 */
template<int dimcast, typename SrcExp, typename DType,
         int etype, int dimdst>
inline Broadcast1DExp<SrcExp, DType, dimdst, dimdst - dimcast>
broadcast(const expr::Exp<SrcExp, DType, etype> &src, Shape<dimdst> shape) {
  TypeCheckPass<dimcast < dimdst && ExpInfo<SrcExp>::kDim == 1>
                ::Error_Expression_Does_Not_Meet_Dimension_Req();
  typedef ShapeCheck<1, SrcExp> ShapeCheckDim1SrcExp;
  CHECK_EQ(ShapeCheckDim1SrcExp::Check(src.self())[0], shape[dimcast])
    << "broadcast, shape mismatch";
  return Broadcast1DExp<SrcExp, DType, dimdst,
                        dimdst - dimcast>(src.self(), shape);
}

/*!
 * \brief a expression that replicate a scalar tensor to target dimension.
 * \param src Tensor<Device,1>: shape[0] == 1
 * \param shape shape of output
 * \return a expresion with type Tensor<Device, dimdst>
 * \tparam dimcast target dimension where the 1D tensor will be broadcasted
 * \tparam SrcExp type of input expression
 * \tparam DType the type of elements
 * \tparam dimdst dimension of destination tensor
 */
template<typename SrcExp, typename DType, int etype, int dimdst>
inline BroadcastScalarExp<SrcExp, DType, dimdst>
broadcast_scalar(const expr::Exp<SrcExp, DType, etype> &src, Shape<dimdst> shape) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim == 1>
                ::Error_Expression_Does_Not_Meet_Dimension_Req();
  typedef ShapeCheck<1, SrcExp> ShapeCheckDim1SrcExp;
  CHECK_EQ(ShapeCheckDim1SrcExp::Check(src.self())[0], 1U)
      << "broadcast_scalar, source need to be scalar expression";
  return BroadcastScalarExp<SrcExp, DType, dimdst>(src.self(), shape);
}
// short cut functions
/*!
 * \brief a expression that replicate a 1 dimension tensor for nrow times
 * \param src Tensor<Device,1>: shape[0]
 * \param nrow number of rows to replicate
 * \return a expresion with type Tensor<Device,2> size(1), size(0) = nrow
 * \tparam Device which device it lies
 */
template<typename SrcExp, typename DType, int etype>
inline Broadcast1DExp<SrcExp, DType, 2, 1>
repmat(const expr::Exp<SrcExp, DType, etype> &src, index_t nrow) {
  return broadcast<1>
      (src, Shape2(nrow, ShapeCheck<1, SrcExp>::Check(src.self())[0]));
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int dimdst, int dimdst_m_cast>
struct Plan<Broadcast1DExp<SrcExp, DType, dimdst, dimdst_m_cast>, DType> {
 public:
  static const int dimcast = dimdst - dimdst_m_cast;
  explicit Plan(const Broadcast1DExp<SrcExp, DType, dimdst, dimdst_m_cast> &e)
      : src_(MakePlan(e.src_)),
        ystride_(e.shape_.ProdShape(dimcast + 1, dimdst - 1)),
        length_(e.shape_[dimcast]) {
    TypeCheckPass<dimcast != dimdst - 1>
        ::Error_Expression_Does_Not_Meet_Dimension_Req();
  }
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return src_.Eval(0, (y / ystride_) % length_);
  }

 private:
  expr::Plan<SrcExp, DType> src_;
  const index_t  ystride_, length_;
};

/*! \brief execution plan of Broadcast1DExp */
template<typename SrcExp, typename DType, int dimdst>
struct Plan<Broadcast1DExp<SrcExp, DType, dimdst, 1>, DType>{
 public:
  explicit Plan(const Broadcast1DExp<SrcExp, DType, dimdst, 1> &e)
      : src_(MakePlan(e.src_)) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return src_.Eval(0, x);
  }

 private:
  expr::Plan<SrcExp, DType> src_;
};

/*! \brief execution plan of Broadcast1DExp */
template<typename SrcExp, typename DType, int dimdst>
struct Plan<BroadcastScalarExp<SrcExp, DType, dimdst>, DType>{
 public:
  explicit Plan(const BroadcastScalarExp<SrcExp, DType, dimdst> &e)
      : src_(MakePlan(e.src_)) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return src_.Eval(0, 0);
  }

 private:
  expr::Plan<SrcExp, DType> src_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_BROADCAST_H_
