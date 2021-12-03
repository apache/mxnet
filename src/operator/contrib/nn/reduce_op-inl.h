#ifndef MXNET_OPERATOR_CONTRIB_NCCLREDUCE_H_
#define MXNET_OPERATOR_CONTRIB_NCCLREDUCE_H_

#include <mxnet/base.h>
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../elemwise_op_common.h"
#include "../../tensor/init_op.h"

#if MXNET_USE_NCCL
#include <nccl.h>
#include <unordered_map>
#include <memory>

namespace mxnet {
namespace op {

struct NCCLReduceParam : public dmlc::Parameter<NCCLReduceParam> {
  int32_t num_gpus;
  int32_t root_rank;
  int32_t rank;
  uintptr_t nccl_unique_id;

  DMLC_DECLARE_PARAMETER(NCCLReduceParam) {
    DMLC_DECLARE_FIELD(num_gpus)
      .set_default(1)
      .describe("Number of all gpus.");
    DMLC_DECLARE_FIELD(root_rank)
      .set_default(0)
      .describe("root rank of reduce operation");
    DMLC_DECLARE_FIELD(rank)
      .set_default(0)
      .describe("rank of current process");
    DMLC_DECLARE_FIELD(nccl_unique_id)
      .describe("NCCL unique ID");
  }
};

template <int req>
struct ncclreduce_backward {
  template <typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* in_grad,
                                  const DType* out_grad) {
    KERNEL_ASSIGN(in_grad[i], req, out_grad[i] * 1);
  }
};

template <typename xpu>
void NCCLReduceBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu>* s     = ctx.get_stream<xpu>();
  const TBlob& out_grad       = inputs[0];
  const TBlob& in_grad        = outputs[0];
  const NCCLReduceParam& param = nnvm::get<NCCLReduceParam>(attrs.parsed);
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<ncclreduce_backward<req_type>, xpu>::Launch(s,
                                                        in_grad.Size(),
                                                        in_grad.dptr<DType>(),
                                                        out_grad.dptr<DType>());
    });
  });
}





class NCCLCommContainer {
 public:
  struct Param {
    int num_gpus;
    int rank;
    uintptr_t nccl_unique_id;
  };
  static inline std::unordered_map<int, std::unique_ptr<ncclComm_t>> comm_map;

  static void Init(const Param& param);
};

}  // namespace op
}  // namespace mxnet

#else
static_assert(false, "You need to compile with NCCL support to use reduce operation!");
#endif  // MXNET_USE_NCCL

#endif  // MXNET_OPERATOR_CONTRIB_SPATIAL_PARALLEL_SUPPORT_H_