#include <mxnet/runtime/ffi_helper.h>
#include <mxnet/runtime/container.h>
#include <mxnet/runtime/packed_func.h>
#include <mxnet/api_registry.h>
#include <mxnet/base.h>
#include <nnvm/c_api.h>
#include <iostream>

#include "../operator/tensor/init_op.h"
#include "../operator/numpy/np_tensordot_op-inl.h"
#include "../imperative/imperative_utils.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.zeros1")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  const static nnvm::Op* op = Op::Get("_npi_zeros");
  nnvm::NodeAttrs attrs;
  op::InitOpParam param;
  param.shape = args[0].operator TShape();
  if (args[1].type_code() == kNull) {
    param.dtype = mshadow::kFloat32;
  } else {
    param.dtype = runtime::String2MXNetTypeWithBool(args[1].operator std::string());
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  if (args[2].type_code() != kNull) {
    attrs.dict["ctx"] = args[2].operator std::string();
  }

  const int num_inputs = 0;
  int infered_num_outputs;
  int num_visible_outputs;
  imperative::SetNumOutputs(op, attrs, num_inputs, &infered_num_outputs, &num_visible_outputs);

  std::vector<NDArray*> ndoutputs(1, nullptr), ndinputs;
  ndoutputs[0] = static_cast<NDArray*>(new NDArray());

  auto state = Imperative::Get()->Invoke(Context::CPU(), attrs, ndinputs, ndoutputs);
  if (Imperative::Get()->is_recording()) {
    Imperative::Get()->RecordOp(std::move(attrs), ndinputs, ndoutputs, state);
  }
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.tensordot_dispatcher")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  bool isscalar = args[2].type_code() == kDLInt;
  const nnvm::Op* op = Op::Get(isscalar ?
                               "_npi_tensordot_int_axes" :
                               "_npi_tensordot");
  nnvm::NodeAttrs attrs;
  attrs.op = op;
  if (isscalar) {
    mxnet::op::TensordotIntAxesParam param;
    param.axes = args[2].operator int();
    attrs.parsed = std::move(param);
  } else {
    mxnet::op::TensordotParam param;
    const runtime::ObjectRef ref = args[2].operator runtime::ObjectRef();
    const runtime::ADTObj* obj = ref.as<runtime::ADTObj>();
    if (obj->operator[](0).get()->IsInstance<::mxnet::runtime::IntegerObj>()) {
      param.a_axes_summed = Tuple<int>(1,
        obj->operator[](0).as<::mxnet::runtime::IntegerObj>()->value);
      param.b_axes_summed = Tuple<int>(1,
        obj->operator[](1).as<::mxnet::runtime::IntegerObj>()->value);
    } else {
      const runtime::ADTObj* a_axes_summed = obj->operator[](0).as<runtime::ADTObj>();
      const runtime::ADTObj* b_axes_summed = obj->operator[](1).as<runtime::ADTObj>();
      param.a_axes_summed = Tuple<int>(a_axes_summed->size, 0);
      param.b_axes_summed = Tuple<int>(b_axes_summed->size, 0);
      for (uint32_t i = 0; i < a_axes_summed->size; ++i) {
        param.a_axes_summed[i] = a_axes_summed->operator[](i).as<::mxnet::runtime::IntegerObj>()->value;
        param.b_axes_summed[i] = b_axes_summed->operator[](i).as<::mxnet::runtime::IntegerObj>()->value;
      }
    }
    attrs.parsed = std::move(param);
  }

  int num_inputs = 2;
  int infered_num_outputs;
  int num_visible_outputs;
  mxnet::imperative::SetNumOutputs(op, attrs, num_inputs, &infered_num_outputs, &num_visible_outputs);

  std::vector<mxnet::NDArray*> ndoutputs(1, nullptr), ndinputs(2, nullptr);
  ndoutputs[0] = reinterpret_cast<mxnet::NDArray*>(new mxnet::NDArray());
  ndinputs[0] = args[0].operator mxnet::NDArray*();
  ndinputs[1] = args[1].operator mxnet::NDArray*();
  auto state = mxnet::Imperative::Get()->Invoke(Context::CPU(), attrs, ndinputs, ndoutputs);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
});

MXNET_REGISTER_API("_npi.nop")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
});

}  // namespace mxnet
