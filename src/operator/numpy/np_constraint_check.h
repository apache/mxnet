#include <algorithm>
#include <string>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "./np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {

template<typename xpu>
void GetReduceOutput(mshadow::Stream<xpu> *s, const TBlob &output_blob, bool *red_output);

struct ConstraintCheckParam : public dmlc::Parameter<ConstraintCheckParam> {
  std::string msg;
  DMLC_DECLARE_PARAMETER(ConstraintCheckParam) {
    DMLC_DECLARE_FIELD(msg)
    .set_default("Constraint violated!")
    .describe("Error message raised when constraint violated");
  }
};

template <typename xpu>
void ConstraintCheckForward(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  // ???
  CHECK(false);
  const ConstraintCheckParam& param =
      nnvm::get<ConstraintCheckParam>(attrs.parsed);
  ReduceAxesComputeImpl<xpu, mshadow_op::product, false, false,
                        op::mshadow_op::identity>(ctx, inputs, req, outputs,
                                                  outputs[0].shape_);
  std::string msg = param.msg;
  bool red_output = true;
  GetReduceOutput(ctx.get_stream<xpu>(), outputs[0], &red_output);
  CHECK_EQ(red_output, true) << msg;
}

}  // namespace op
}  // namespace mxnet