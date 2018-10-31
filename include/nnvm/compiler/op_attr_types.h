/*!
 *  Copyright (c) 2017 by Contributors
 * \file nnvm/compiler/op_attr_types.h
 * \brief The Expr and related elements in DataFlow construction.
 */
#ifndef NNVM_COMPILER_OP_ATTR_TYPES_H_
#define NNVM_COMPILER_OP_ATTR_TYPES_H_

#include <tvm/expr.h>
#include <tvm/tensor.h>
#include <tvm/schedule.h>
#include <tvm/packed_func_ext.h>
#include <tvm/runtime/registry.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/graph.h>
#include <vector>
#include <string>
#include "packed_func_ext.h"

namespace nnvm {
namespace compiler {

using ::tvm::Array;
using ::tvm::Tensor;
using ::tvm::Schedule;

/*! \brief operator pattern used in graph fusion */
enum OpPatternKind {
  // Elementwise operation
  kElemWise = 0,
  // Broadcasting operator, can always map output axis to the input in order.
  // for example :code:`out[i, ax1, j, ax2] = input[i, j]`.
  // Note that the axis need to be in order so transpose is not a bcast operator.
  kBroadcast = 1,
  // Injective operator, can always injectively map output axis to a single input axis.
  // All injective operator can still be safely fused to injective and reduction.
  kInjective = 2,
  // Communicative reduction operator.
  kCommReduce = 3,
  // Complex operation, can still fuse elemwise operations into its output.
  // but cannot chain another complex op
  kOutEWiseFusable = 4,
  // Opaque operation, cannot fuse anything.
  kOpaque = 8
};

/*! \brief the operator pattern */
using TOpPattern = int;

/*!
 * \brief Computation description interface
 * \param attrs The attribute of the node.
 * \param inputs The input tensors(placeholders)
 * \param out_info Tensors holding shape/type information about output,
 &                 these are always placeholders.
 * \return The output description of the tensor.
 */
using FTVMCompute = std::function<
  Array<Tensor>(const NodeAttrs& attrs,
                const Array<Tensor>& inputs,
                const Array<Tensor>& out_info)>;

/*!
 * \brief Build the computation schedule for
 *  op whose root is at current op.
 * \param attrs The attribute of the node.
 * \param outs The output tensors.
 * \param target The build target.
 * \return schedule The computation schedule.
 */
using FTVMSchedule = std::function<
  Schedule(const NodeAttrs& attrs,
           const Array<Tensor>& outs,
           const std::string& target)>;

/*!
 * \brief Modify the op node to alter its input layout.
 *  it is invoked in AlterOpLayout pass.
 * \param attrs The attribute of the original node.
 * \param inputs The input symbols of the original node.
 * \param tinfos The inferred shape and dtype of the inputs.
 * \param ret The replaced operator.
 * \return Whether to replace current operator.
 */
using FTVMAlterOpLayout = std::function<
  bool(const NodeAttrs& attrs,
       const Symbol& inputs,
       const Array<Tensor>& tinfos,
       Symbol* ret)>;

/*!
 * \brief Transform from normal operator to vectorized operator
 * \param node The source node.
 * \return Transformed vectorized op.
 */
using FTVMVectorizedOp = std::function<nnvm::NodePtr (const nnvm::Node* node)>;

}  // namespace compiler
}  // namespace nnvm
#endif  // NNVM_COMPILER_OP_ATTR_TYPES_H_
