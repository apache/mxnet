/*!
 *  Copyright (c) 2015 by Contributors
 * \file legacy_op_util.cc
 * \brief Utility to adapt OpProperty to the new NNVM registery
 */
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <memory>

namespace mxnet {
namespace op {

using nnvm::Op;
using nnvm::NodeAttrs;
using nnvm::array_view;
using nnvm::FInferShape;
using nnvm::FInferType;
using nnvm::FListInputNames;
using nnvm::FListOutputNames;

// function to use operator property to infer attr
template<typename AttrType, typename FInfer>
bool OpPropInferAttr(const NodeAttrs& attrs,
                     array_view<AttrType*> iattr,
                     array_view<AttrType*> oattr,
                     FInfer finfer) {
  auto& prop = nnvm::get<std::shared_ptr<OperatorProperty> >(attrs.parsed);
  std::vector<AttrType> in_attr(iattr.size());
  std::vector<AttrType> out_attr, aux_attr;

  for (size_t i = 0; i < iattr.size(); ++i) {
    in_attr[i] = *iattr[i];
  }
  if (!finfer(prop.get(), &in_attr, &out_attr, &aux_attr)) return false;

  for (size_t i = 0; i < iattr.size(); ++i) {
    *iattr[i] = in_attr[i];
  }

  CHECK_EQ(oattr.size(), out_attr.size());
  for (size_t i = 0; i < oattr.size(); ++i) {
    *oattr[i] = out_attr[i];
  }
  CHECK_EQ(aux_attr.size(), 0)
      << "not implemented adapter with aux state";
  return true;
}

bool OpPropInferShape(const NodeAttrs& attrs,
                      array_view<TShape*> iattr,
                      array_view<TShape*> oattr) {
  auto finfer = [](const OperatorProperty* op,
                   std::vector<TShape> *in,
                   std::vector<TShape> *out,
                   std::vector<TShape> *aux) {
    return op->InferShape(in, out, aux);
  };
  return OpPropInferAttr(attrs, iattr, oattr, finfer);
}

bool OpPropInferType(const NodeAttrs& attrs,
                      array_view<int*> iattr,
                      array_view<int*> oattr) {
  auto finfer = [](const OperatorProperty* op,
                   std::vector<int> *in,
                   std::vector<int> *out,
                   std::vector<int> *aux) {
    return op->InferType(in, out, aux);
  };
  return OpPropInferAttr(attrs, iattr, oattr, finfer);
}

inline uint32_t OpPropNumInputs(const NodeAttrs& attrs) {
  auto& prop = nnvm::get<std::shared_ptr<OperatorProperty> >(attrs.parsed);
  return prop->ListArguments().size() + prop->ListAuxiliaryStates().size();
}

inline uint32_t OpPropNumOutputs(const NodeAttrs& attrs) {
  auto& prop = nnvm::get<std::shared_ptr<OperatorProperty> >(attrs.parsed);
  return prop->NumOutputs();
}

std::vector<std::string> OpPropListInputNames(const NodeAttrs& attrs) {
  auto& prop = nnvm::get<std::shared_ptr<OperatorProperty> >(attrs.parsed);
  return prop->ListArguments();
}

std::vector<std::string> OpPropListOutputNames(const NodeAttrs& attrs) {
  auto& prop = nnvm::get<std::shared_ptr<OperatorProperty> >(attrs.parsed);
  return prop->ListOutputs();
}

// register the legacy operator properties under NNVM registry.
void RegisterLegacyOpProp() {
  for (auto reg : dmlc::Registry<OperatorPropertyReg>::List()) {
    Op& op = ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(reg->name);
    if (op.attr_parser != nullptr) continue;
    auto creator = reg->body;
    // attribute parser
    op.set_attr_parser([creator](NodeAttrs* attrs) {
        if (attrs->parsed.empty()) {
          std::shared_ptr<OperatorProperty> ptr(creator());
          std::vector<std::pair<std::string, std::string> > kwargs(
              attrs->dict.begin(), attrs->dict.end());
          ptr->Init(kwargs);
          attrs->parsed = std::move(ptr);
        }
      });
    // numer of inputs
    op.set_num_inputs(OpPropNumInputs);
    op.set_num_outputs(OpPropNumOutputs);
    op.attr<FListInputNames>("FListInputNames", OpPropListInputNames);
    op.attr<FListOutputNames>("FListOutputNames", OpPropListOutputNames);
    op.attr<FInferShape>("FInferShape", OpPropInferShape);
    op.attr<FInferType>("FInferType", OpPropInferType);
    if (reg->key_var_num_args.length() != 0) {
      op.attr<std::string>("key_var_num_args", reg->key_var_num_args);
    }
  }
}

}  // namespace op
}  // namespace mxnet
