/*!
 * Copyright (c) 2015 by Contributors
 * \file  operator_common.h
 * \brief common internal header of most operators
 *   this header includes utility functions operator can use
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_OPERATOR_COMMON_H_
#define MXNET_OPERATOR_OPERATOR_COMMON_H_

#include <dmlc/json.h>
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <mxnet/base.h>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace mxnet {
namespace op {
/*!
 * \brief assign the expression to out according to request
 * \param out the data to be assigned
 * \param req the assignment request
 * \param exp the expression
 * \tparam OType output type
 * \tparam Exp expression type
 */
#define Assign(out, req, exp)           \
  {                                     \
    switch (req) {                      \
      case kNullOp:                     \
        break;                          \
      case kWriteTo:                    \
      case kWriteInplace:               \
        (out) = (exp);                  \
        break;                          \
      case kAddTo:                      \
        (out) += (exp);                 \
        break;                          \
      default:                          \
        LOG(FATAL) << "not reached";    \
    }                                   \
  }


/*! \brief exception throwed by InferShape error */
struct InferShapeError {
  /*! \brief analyze message */
  std::string msg;
  /*! \brief corresponding input index */
  int index;
  // constructor
  InferShapeError(std::string msg, int index)
    : msg(msg), index(index) {}
};

/*! \brief exception throwed by InferShape error */
struct InferTypeError {
  /*! \brief analyze message */
  std::string msg;
  /*! \brief corresponding input index */
  int index;
  // constructor
  InferTypeError(std::string msg, int index)
    : msg(msg), index(index) {}
};

/*!
 * \brief macro assign shape to out if out is unknown otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param shape_array the shape array to store the result
 * \param index the index of in the array
 * \param shape the inferred shape
 */
#define SHAPE_ASSIGN_CHECK(shape_array, index, shape)                   \
  {                                                                     \
    auto &local_out = (shape_array)[index];                             \
    if (local_out.ndim() == 0) {                                        \
      local_out = shape;                                                \
    } else {                                                            \
      if (local_out != shape) {                                         \
        std::ostringstream os;                                          \
        os << "Shape inconsistent, Provided=" << local_out << ','       \
           << " inferred shape=" << shape;                              \
        throw ::mxnet::op::InferShapeError(os.str(), index);            \
      }                                                                 \
    }                                                                   \
  }

/*!
 * \brief macro assign type to out if out is unknown (-1) otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param type_array the type array to store the result
 * \param index the index of in the array
 * \param type the inferred type
 */
#define TYPE_ASSIGN_CHECK(type_array, index, type)                      \
  {                                                                     \
    auto &local_out = (type_array)[index];                              \
    if (local_out == -1) {                                              \
      local_out = type;                                                 \
    } else {                                                            \
      if (local_out != type) {                                          \
        std::ostringstream os;                                          \
        os << "Type inconsistent, Provided=" << local_out << ','        \
           << " inferred type=" << type;                                \
        throw ::mxnet::op::InferTypeError(os.str(), index);             \
      }                                                                 \
    }                                                                   \
  }

// helper macro to implement bind dispatch
#if MXNET_USE_CUDA
#define DO_BIND_DISPATCH(Method, ...)                                \
  if (ctx.dev_mask() == cpu::kDevMask) {                             \
      return Method<cpu>(__VA_ARGS__);                               \
    } else {                                                         \
      return Method<gpu>(__VA_ARGS__);                               \
    }
#else
#define DO_BIND_DISPATCH(Method, ...)                                \
  if (ctx.dev_mask() == cpu::kDevMask) {                             \
    return Method<cpu>(__VA_ARGS__);                                 \
  } else {                                                           \
    LOG(FATAL) << "GPU is not enabled";                              \
    return nullptr;                                                  \
  }
#endif

// describe op registration point
// TODO(eric): move to dmlc-core
#define STRINGIZE_DETAIL(x) #x
#define STRINGIZE(x) STRINGIZE_DETAIL(x)
#define MXNET_DESCRIBE(...) describe(__VA_ARGS__ "\n\nFrom:" __FILE__ ":" STRINGIZE(__LINE__))

// quick helper to make node
inline std::vector<nnvm::NodeEntry> MakeGradNode(
    const char* op_name,
    const nnvm::NodePtr& n,
    std::vector<nnvm::NodeEntry> inputs,
    std::unordered_map<std::string, std::string> dict) {
  nnvm::NodePtr p = nnvm::Node::Create();
  p->attrs.op = nnvm::Op::Get(op_name);
  p->attrs.name = n->attrs.name + "_backward";
  p->attrs.dict = std::move(dict);
  if (p->op()->attr_parser != nullptr) {
    p->op()->attr_parser(&(p->attrs));
  }
  p->control_deps.emplace_back(n);
  p->inputs = std::move(inputs);
  std::vector<nnvm::NodeEntry> ret;
  for (index_t i = 0; i < p->num_outputs(); ++i) {
    ret.emplace_back(nnvm::NodeEntry{p, i, 0});
  }
  return ret;
}

/*! \brief Parse keyword arguments as PType arguments and save to parsed */
template<typename PType>
inline void ParamParser(nnvm::NodeAttrs* attrs) {
  PType param;
  try {
    param.Init(attrs->dict);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  attrs->parsed = std::move(param);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_OPERATOR_COMMON_H_
