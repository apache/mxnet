/*!
 *  Copyright (c) 2015 by Contributors
 * \file legacy_op_util.cc
 * \brief Utility to adapt OpProperty to the new NNVM registery
 */
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/node.h>
#include <memory>

namespace mxnet {
namespace op {

using nnvm::Op;
using nnvm::Node;
using nnvm::NodeAttrs;

class ParsedOpProp {
 public:
  std::shared_ptr<OperatorProperty> ptr;
  std::vector<std::string> arguments;
  std::vector<std::string> aux_states;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<std::pair<int, int> > forward_inplace;
  // initializer
  void Init(const NodeAttrs& attrs) {
    std::vector<std::pair<std::string, std::string> > kwargs(
        attrs.dict.begin(), attrs.dict.end());
    ptr->Init(kwargs);
    arguments = ptr->ListArguments();
    aux_states = ptr->ListAuxiliaryStates();
    outputs = ptr->ListOutputs();
    inputs = arguments;
    inputs.insert(
        inputs.end(), aux_states.begin(), aux_states.end());

    std::vector<int> in_data(arguments.size());
    std::vector<int> out_data(outputs.size());
    std::vector<void*> out_addr(outputs.size());
    for (size_t i = 0; i < in_data.size(); ++i) {
      in_data[i] = static_cast<int>(i);
    }
    for (size_t i = 0; i < out_data.size(); ++i) {
      out_data[i] = static_cast<int>(i);
      out_addr[i] = &out_data[i];
    }
    for (auto& kv : ptr->ForwardInplaceOption(in_data, out_addr)) {
      forward_inplace.push_back(
          std::make_pair(kv.first, *static_cast<int*>(kv.second)));
    }
  }
};

// function to use operator property to infer attr
template<typename AttrType, typename FInfer>
bool OpPropInferAttr(const Node& n,
                     std::vector<AttrType> *iattr,
                     std::vector<AttrType> *oattr,
                     FInfer finfer) {
  auto& prop = nnvm::get<ParsedOpProp>(n.attrs.parsed);
  CHECK_EQ(prop.inputs.size(), iattr->size());
  std::vector<AttrType> in_attr(prop.arguments.size());
  std::vector<AttrType> aux_attr(prop.aux_states.size());

  for (size_t i = 0; i < prop.arguments.size(); ++i) {
    in_attr[i] = (*iattr)[i];
  }
  for (size_t i = 0; i < prop.aux_states.size(); ++i) {
    aux_attr[i] = (*iattr)[i + prop.arguments.size()];
  }

  if (!finfer(prop.ptr.get(), &in_attr, oattr, &aux_attr)) return false;

  for (size_t i = 0; i < prop.arguments.size(); ++i) {
    (*iattr)[i] = in_attr[i];
  }
  for (size_t i = 0; i < prop.aux_states.size(); ++i) {
    (*iattr)[i + prop.arguments.size()] = aux_attr[i];
  }
  return true;
}

bool OpPropInferShape(const Node& n,
                      std::vector<TShape> *iattr,
                      std::vector<TShape>* oattr) {
  auto finfer = [](const OperatorProperty* op,
                   std::vector<TShape> *in,
                   std::vector<TShape> *out,
                   std::vector<TShape> *aux) {
    return op->InferShape(in, out, aux);
  };
  return OpPropInferAttr(n, iattr, oattr, finfer);
}

bool OpPropInferType(const Node& n,
                     std::vector<int> *iattr,
                     std::vector<int>* oattr) {
  auto finfer = [](const OperatorProperty* op,
                   std::vector<int> *in,
                   std::vector<int> *out,
                   std::vector<int> *aux) {
    return op->InferType(in, out, aux);
  };
  return OpPropInferAttr(n, iattr, oattr, finfer);
}

inline uint32_t OpPropNumInputs(const Node& n) {
  auto& prop = nnvm::get<ParsedOpProp>(n.attrs.parsed);
  return prop.inputs.size();
}

inline uint32_t OpPropNumOutputs(const Node& n) {
  auto& prop = nnvm::get<ParsedOpProp>(n.attrs.parsed);
  return prop.outputs.size();
}

std::vector<std::string> OpPropListInputNames(const Node& n) {
  auto& prop = nnvm::get<ParsedOpProp>(n.attrs.parsed);
  return prop.inputs;
}

std::vector<std::string> OpPropListOutputNames(const Node& n) {
  auto& prop = nnvm::get<ParsedOpProp>(n.attrs.parsed);
  return prop.outputs;
}

std::vector<uint32_t> OpPropMutateInputs(const Node& n) {
  auto& prop = nnvm::get<ParsedOpProp>(n.attrs.parsed);
  std::vector<uint32_t> ret;
  for (uint32_t i = 0; i < prop.aux_states.size(); ++i) {
    ret.push_back(static_cast<uint32_t>(i + prop.arguments.size()));
  }
  return ret;
}

Operator* OpPropCreateLayerOp(const Node& n,
                              Context ctx,
                              const std::vector<TShape>& ishape,
                              const std::vector<int>& itype) {
  auto& prop = nnvm::get<ParsedOpProp>(n.attrs.parsed);
  std::vector<TShape> is = ishape;
  std::vector<int> it = itype;
  return prop.ptr->CreateOperatorEx(ctx, &is, &it);
}

std::vector<std::pair<int, int> > OpPropInplaceOption(const Node& n) {
  auto& prop = nnvm::get<ParsedOpProp>(n.attrs.parsed);
  return prop.forward_inplace;
}

// register the legacy operator properties under NNVM registry.
void RegisterLegacyOpProp() {
  using nnvm::FInferShape;
  using nnvm::FInferType;
  using nnvm::FMutateInputs;
  using nnvm::FListInputNames;
  using nnvm::FListOutputNames;
  using nnvm::FInplaceOption;
  using mxnet::FCreateLayerOp;

  for (auto reg : dmlc::Registry<OperatorPropertyReg>::List()) {
    Op& op = ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(reg->name);
    if (op.attr_parser != nullptr) continue;
    auto creator = reg->body;
    // attribute parser
    op.set_attr_parser([creator](NodeAttrs* attrs) {
        if (attrs->parsed.empty()) {
          ParsedOpProp op;
          op.ptr.reset(creator());
          op.Init(*attrs);
          attrs->parsed = std::move(op);
        }
      });
    // numer of inputs
    op.set_num_inputs(OpPropNumInputs);
    op.set_num_outputs(OpPropNumOutputs);
    op.attr<FListInputNames>("FListInputNames", OpPropListInputNames);
    op.attr<FListOutputNames>("FListOutputNames", OpPropListOutputNames);
    op.attr<FInferShape>("FInferShape", OpPropInferShape);
    op.attr<FInferType>("FInferType", OpPropInferType);
    op.attr<FMutateInputs>("FMutateInputs", OpPropMutateInputs);
    op.attr<FInplaceOption>("FInplaceOption", OpPropInplaceOption);
    op.attr<FCreateLayerOp>("FCreateLayerOp", OpPropCreateLayerOp);
    if (reg->key_var_num_args.length() != 0) {
      op.attr<std::string>("key_var_num_args", reg->key_var_num_args);
    }
  }
}

}  // namespace op
}  // namespace mxnet
