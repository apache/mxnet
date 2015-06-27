/*!
 *  Copyright (c) 2015 by Contributors
 * \file narray_op.h
 * \brief the real execution functions of narray operations
 */
#ifndef MXNET_NARRAY_OP_H_
#define MXNET_NARRAY_OP_H_
#include <dmlc/logging.h>
#include <mshadow/tensor.h>
#include <mxnet/base.h>
#include <mxnet/tensor_blob.h>

namespace mxnet {
/*! \brief namespace to support all possible NArray operator */
namespace narray {
struct BinaryBase {
  inline static TShape GetShape(const TShape &lshape, const TShape &rshape) {
    CHECK(lshape == rshape) << "operands shape mismatch";
    CHECK(lshape.ndim() != 0) << "source operand have zero dimension shape";
    return lshape;
  }
};
// operators
struct Plus : public BinaryBase {
  typedef mshadow::op::plus mshadow_op;
};
struct Minus : public BinaryBase {
  typedef mshadow::op::minus mshadow_op;
};
struct Mul : public BinaryBase {
  typedef mshadow::op::mul mshadow_op;
};
struct Div : public BinaryBase {
  typedef mshadow::op::div mshadow_op;
};
template<typename Device, typename OP>
void Eval(const TBlob &lhs, const TBlob &rhs, TBlob ret, RunContext ctx);

}  // namespace narray
}  // namespace mxnet
#endif  // MXNET_NARRAY_OP_H_
