/*!
 * Copyright (c) 2020 by Contributors
 * \file spatial_parallel_support.cu
 * \brief Support operators for spatial parallelism
 * \author Przemyslaw Tredak
*/

#include "reduce_op-inl.h"
#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <mutex>
#include <vector>
#include "../../operator_common.h"
#include "../../../common/utils.h"
#include "../../tensor/elemwise_binary_op.h"

namespace mxnet {
namespace op {


void NCCLReduceCompute(const nnvm::NodeAttrs& attrs,
                                     const OpContext& ctx,
                                     const std::vector<TBlob>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<TBlob>& outputs) {
  const NCCLReduceParam& param = nnvm::get<NCCLReduceParam>(attrs.parsed);
  if (req[0] == OpReqType::kNullOp) return;
  if (param.num_gpus == 1 && req[0] == OpReqType::kWriteInplace) return;
  NCCLCommContainer::Param p = {param.num_gpus,
                                param.rank,
                                param.nccl_unique_id};
  NCCLCommContainer::Init(p);

  std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
  ncclComm_t comm = *(NCCLCommContainer::comm_map.at(param.num_gpus));
  const index_t size = inputs[0].shape_.Size() *
                       common::mshadow_type_info(inputs[0].type_flag_).size;
  if (req[0] != OpReqType::kAddTo) {

    ncclResult_t result = ncclReduce(inputs[0].dptr_,
                                        outputs[0].dptr_,
                                        size, ncclInt8, ncclSum,  param.root_rank,
                                        comm,
                                        mshadow::Stream<gpu>::GetStream(ctx.get_stream<gpu>()));


    CHECK_EQ(result, ncclSuccess) << "NCCL Reduce failed!";
  } else {
    LOG(FATAL) << "kAddTo not supported yet!";
  }
}

NNVM_REGISTER_OP(_contrib_NCCLReduce)
.set_attr<FCompute>("FCompute<gpu>", NCCLReduceCompute);

NNVM_REGISTER_OP(_backward_NCCLReduce)
.set_attr<FCompute>("FCompute<gpu>", NCCLReduceBackward<gpu>);

}  // namespace op
}  // namespace mxnet