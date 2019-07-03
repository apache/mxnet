#ifndef MXNET_OPERATOR_NUMPY_AROUND_OP_H_
#define MXNET_OPERATOR_NUMPY_AROUND_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <cmath>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

struct AroundParam : public dmlc::Parameter<AroundParam> {
  int decimals;
  DMLC_DECLARE_PARAMETER(QuadraticParam) {
    DMLC_DECLARE_FIELD(decimals)
      .set_default(0)
      .describe("Number of decimal places to round to.");
  }
};

template<int req>
struct around_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data,
                                  const int decimals) {
    int d = 0;
    double temp = in_data[i];
    while(d != decimals){
        if(decimals > 0){
            d++;
            temp *= 10;
        }
        else{
            d--;
            temp /= 10;
        }
    }
    temp = round(temp);
    while(d != 0){
        if(decimals > 0){
            d--;
            temp /= 10;
        }
        else{
            d++;
            temp *= 10;
        }
    }
    KERNEL_ASSIGN(out_data[i], req, (DType)temp);
  }
};

template<typename xpu>
void AroundOpForward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  const AroundParam& param = nnvm::get<AroundParam>(attrs.parsed);
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<around_forward<req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>(),
          param.decimals);
    });
  });
}

}// namespace mxnet
}// namespace op
#endif