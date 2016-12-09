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
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {
template<typename AttrType,
         bool (*is_none)(const AttrType&), bool reverse_infer>
inline bool ElemwiseAttr(const nnvm::NodeAttrs& attrs,
                         std::vector<AttrType> *in_attrs,
                         std::vector<AttrType> *out_attrs) {
  size_t n_in = in_attrs->size();
  size_t n_out = out_attrs->size();
  bool found = false;
  AttrType dattr;
  for (size_t i = 0; i < n_in; ++i) {
    if (!is_none((*in_attrs)[i])) {
      dattr = (*in_attrs)[i];
      found = true;
      break;
    }
  }
  if (reverse_infer && !found) {
    for (size_t i = 0; i < n_out; ++i) {
      if (!is_none((*out_attrs)[i])) {
        dattr = (*out_attrs)[i];
        found = true;
        break;
      }
    }
  }
  if (!found) {
    return false;
  }
  for (size_t i = 0; i < n_in; ++i) {
    if (is_none((*in_attrs)[i])) {
      (*in_attrs)[i] = dattr;
    } else if ((*in_attrs)[i] != dattr) {
      LOG(FATAL) << "Incompatible attr in node " << attrs.name << " at " << i << "-th input: "
                 << "expected " << dattr << ", got " << (*in_attrs)[i];
    }
  }
  for (size_t i = 0; i < n_out; ++i) {
    if (is_none((*out_attrs)[i])) {
      (*out_attrs)[i] = dattr;
    } else if ((*out_attrs)[i] != dattr) {
      LOG(FATAL) << "Incompatible attr in node " << attrs.name << " at " << i << "-th output: "
                 << "expected " << dattr << ", got " << (*out_attrs)[i];
    }
  }
  return true;
}

inline bool shape_is_none(const TShape& x) {
  return  x.ndim() == 0;
}

template<int n_in, int n_out>
inline bool ElemwiseShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), n_in) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), n_out);
  return ElemwiseAttr<TShape, shape_is_none, true>(
    attrs, in_attrs, out_attrs);
}

inline bool type_is_none(const int& x) {
  return x == -1;
}

template<int n_in, int n_out>
inline bool ElemwiseType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), n_in) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), n_out);
  return ElemwiseAttr<int, type_is_none, true>(
    attrs, in_attrs, out_attrs);
}

struct ElemwiseGradUseIn {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    for (auto& h : n->inputs) {
      heads.push_back(h);
    }
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

struct ElemwiseGradUseOut {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    index_t n_out = n->num_outputs();
    for (index_t i = 0; i < n_out; ++i) {
      heads.emplace_back(nnvm::NodeEntry{n, i, 0});
    }
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

struct ElemwiseGradUseNone {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeGradNode(op_name, n, ograds, n->attrs.dict);
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_ELEMWISE_OP_COMMON_H_
