/*!
* Copyright (c) 2016 by Contributors
* \file elemwise_op_common.h
* \brief common function used for broadcasting and reducing
* \author Xingjian Shi
*/
#ifndef MXNET_OPERATOR_ELEMWISE_OP_COMMON_H_
#define MXNET_OPERATOR_ELEMWISE_OP_COMMON_H_
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "../common/utils.h"

namespace mxnet {
namespace op {
template<typename AttrType, bool (*is_none)(const AttrType&),
         bool (*assign)(AttrType*, const AttrType&), bool reverse_infer,
         std::string (*attr_string)(const AttrType&)>
inline bool ElemwiseAttr(const nnvm::NodeAttrs& attrs,
                         std::vector<AttrType> *in_attrs,
                         std::vector<AttrType> *out_attrs,
                         const AttrType& none) {
  AttrType dattr = none;
  auto deduce = [&](std::vector<AttrType> *vec, const char *name) {
      for (size_t i = 0; i < vec->size(); ++i) {
        CHECK(assign(&dattr, (*vec)[i]))
          << "Incompatible attr in node " << attrs.name << " at " << i << "-th "
          << name << ": " << "expected " << attr_string(dattr)
          << ", got " << attr_string((*vec)[i]);
      }
    };
  deduce(in_attrs, "input");
  if (reverse_infer) deduce(out_attrs, "output");

  auto write = [&](std::vector<AttrType> *vec, const char *name) {
      for (size_t i = 0; i < vec->size(); ++i) {
        CHECK(assign(&(*vec)[i], dattr))
          << "Incompatible attr in node " << attrs.name << " at " << i << "-th "
          << name << ": " << "expected " << attr_string(dattr)
          << ", got " << attr_string((*vec)[i]);
      }
    };
  write(in_attrs, "input");
  write(out_attrs, "output");
  if (is_none(dattr)) return false;
  return true;
}

// Only inferring output storage types from input for now
template<typename AttrType, bool (*is_none)(const AttrType&),
         bool (*assign)(AttrType*, const AttrType&), bool reverse_infer,
         bool enable_fallback>
inline bool ElemwiseStorageAttr(const nnvm::NodeAttrs& attrs,
                         std::vector<AttrType> *in_attrs,
                         std::vector<AttrType> *out_attrs) {
  auto deduce = [&](std::vector<AttrType> *vec, const char *name, AttrType& result,
                    bool fallback) {
      auto &v = *vec;
      for (size_t i = 0; i < vec->size(); ++i) {
        if (v[i] == kUndefinedStorage) {
          // if input type is unknown, assume it's default storage
          CHECK(assign(&v[i], kDefaultStorage));
        } else if (assign(&result, v[i]) == false && fallback) {
          result = kDefaultStorage;
        }
      }
    };
  AttrType dattr = kUndefinedStorage;
  deduce(in_attrs, "input", dattr, enable_fallback);
  if (reverse_infer) {
    LOG(FATAL) << "not implemented yet";
  }
  auto write = [&](std::vector<AttrType> *vec, const char *name) {
      for (size_t i = 0; i < vec->size(); ++i) {
        CHECK(assign(&(*vec)[i], dattr))
          << "Incompatible attr in node " << attrs.name << " at " << i << "-th "
          << name << ": " << "expected " << dattr << ", got " << (*vec)[i];
      }
    };
  if (is_none(dattr)) dattr = kDefaultStorage;
  write(out_attrs, "output");
  return true;
}

template<int n_in, int n_out>
inline bool ElemwiseShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  return ElemwiseAttr<TShape, shape_is_none, shape_assign, true, shape_string>(
    attrs, in_attrs, out_attrs, TShape());
}

template<int n_in, int n_out>
inline bool ElemwiseType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
    attrs, in_attrs, out_attrs, -1);
}

template<int n_in, int n_out>
inline bool ElemwiseStorageType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  return ElemwiseStorageAttr<int, type_is_none, type_assign, false, true>(
    attrs, in_attrs, out_attrs);
}

inline bool IdentityAttrLikeRhsStorageType(const nnvm::NodeAttrs& attrs,
                                           std::vector<int> *in_attrs,
                                           std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(2)) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), static_cast<size_t>(1)) << " in operator " << attrs.name;
  auto &in = *in_attrs;
  auto &out = *out_attrs;
  CHECK_NE(in[1], kUndefinedStorage) << "rhs storage type must be known";
  if (in[0] == kUndefinedStorage) in[0] = in[1];
  if (out[0] == kUndefinedStorage) out[0] = in[1];
  return true;
}

// Transfer gradient and input to FGradient function
struct ElemwiseGradUseIn {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode(op_name, n, ograds, n->inputs, n->attrs.dict);
  }
};

// Transfer gradient and output to FGradient function
struct ElemwiseGradUseOut {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    std::vector<nnvm::NodeEntry> heads;
    index_t n_out = n->num_outputs();
    for (index_t i = 0; i < n_out; ++i) {
      heads.emplace_back(nnvm::NodeEntry{n, i, 0});
    }
    return MakeNonlossGradNode(op_name, n, ograds, heads, n->attrs.dict);
  }
};

// Transfer only gradient to FGradient function
struct ElemwiseGradUseNone {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode(op_name, n, ograds, {}, n->attrs.dict);
  }
};

// TODO(haibin) this is a temporary function for debugging purpose. Remove later.
template <int dim, typename DType>
void print_info(const mshadow::Tensor<cpu, dim, DType>& tensor, const std::string& name) {
  std::cout << "Tensor " << name << " with shape (";
  int len = 1;
  for (int i = 0; i < dim; i++) {
    len *= tensor.shape_[i];
    std::cout << tensor.shape_[i] << ",";
    if (i == dim - 1) std::cout << ")";
  }
  std::cout << std::endl;
  for (int j = 0; j < len; j ++) std::cout << tensor.dptr_[j] << " ";
  std::cout << std::endl;
}


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_ELEMWISE_OP_COMMON_H_
