#include "reduce_op-inl.h"
#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "../../operator_common.h"
#include "../../elemwise_op_common.h"


namespace mxnet {
namespace op {

void NCCLCommContainer::Init(const NCCLCommContainer::Param& param) {
  std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
  if (NCCLCommContainer::comm_map.count(param.num_gpus) == 0) {
    auto [it, inserted] = NCCLCommContainer::comm_map.emplace(param.num_gpus, // NOLINT(*)
        std::make_unique<ncclComm_t>());
    CHECK(inserted) << "Could not insert new NCCL communicator!";
    ncclComm_t* comm = it->second.get();
    ncclUniqueId id = *(reinterpret_cast<ncclUniqueId*>(
          reinterpret_cast<void*>(param.nccl_unique_id)));
    auto result = ncclCommInitRank(comm, param.num_gpus, id, param.rank);
    CHECK_EQ(result, ncclSuccess) << "ncclCommInitRank failed!";
  }
}

bool NCCLReduceShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0).ndim() != 0U && out_attrs->at(0).Size() != 0U;
}

inline bool NCCLReduceType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}

DMLC_REGISTER_PARAMETER(NCCLReduceParam);

NNVM_REGISTER_OP(_contrib_NCCLReduce)
.describe(R"code(Reduce operation
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<NCCLReduceParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", NCCLReduceShape)
.set_attr<nnvm::FInferType>("FInferType", NCCLReduceType)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_NCCLReduce"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    const NCCLReduceParam& param = nnvm::get<NCCLReduceParam>(attrs.parsed);
    if (param.num_gpus == 1) {
      return std::vector<bool>{true};
    } else {
      return std::vector<bool>{false};
    }
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(NCCLReduceParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_NCCLReduce)
.set_attr_parser(ParamParser<NCCLReduceParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", NCCLReduceBackward<cpu>);
}
}