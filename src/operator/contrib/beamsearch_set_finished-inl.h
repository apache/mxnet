#include <dmlc/parameter.h>

#include "../operator_common.h"
namespace mxnet {
namespace op {

namespace beamsearch_set_finished {
enum BeamsearchSetFinishedInputs {kDist, kScores, kFin, kOverMax};
enum BeamsearchSetFinishedOutputs {kOut};
}


//template<int score_idx, int eos_idx>
struct beamsearch_set_finished_forward {
    template<typename DType, typename IType>
    MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data,
                                    const DType* scores, const IType* fin, const IType* over_max,
                                    const DType mask_val, const int score_idx, const int eos_idx,
                                    int V) {
        int j = i / V;
        int k = i % V;
        bool f = static_cast<bool>(fin[j]);
        bool o = static_cast<bool>(over_max[j]);
        bool s = k == score_idx;
        bool e = k == eos_idx;
        bool input = !f && (!o || e);
        bool score = f && s;
        if (input) out_data[i] = in_data[i];
        else if (score) out_data[i] = scores[j];
        else out_data[i] = mask_val;
    }
};

struct BeamsearchSetFinishedParam : public dmlc::Parameter<BeamsearchSetFinishedParam> {
    int score_idx;
    int eos_idx;
    float mask_val;
    DMLC_DECLARE_PARAMETER(BeamsearchSetFinishedParam) {
        DMLC_DECLARE_FIELD(score_idx)
            .set_default(0)
            .describe("Index to set the score of finished beams.");
        DMLC_DECLARE_FIELD(eos_idx)
            .describe("Index of the EOS token.");
        DMLC_DECLARE_FIELD(mask_val)
            .set_default(std::numeric_limits<float>::lowest())
            .describe("Padding value used to mask out unwanted tokens in beams.");
    }
};

inline bool BeamsearchSetFinishedShape(const nnvm::NodeAttrs& attrs,
                                       mxnet::ShapeVector* in_attrs,
                                       mxnet::ShapeVector* out_attrs) {
    const BeamsearchSetFinishedParam& param = nnvm::get<BeamsearchSetFinishedParam>(attrs.parsed);
    CHECK_EQ(in_attrs->size(), 4U);
    CHECK_EQ(out_attrs->size(), 1U);

    auto dist = in_attrs->at(beamsearch_set_finished::kDist);
    auto scores = in_attrs->at(beamsearch_set_finished::kScores);
    auto fin = in_attrs->at(beamsearch_set_finished::kFin);
    auto over_max = in_attrs->at(beamsearch_set_finished::kOverMax);
    CHECK_EQ(dist.ndim(), 2U);
    CHECK_EQ(scores.ndim(), 2U);
    CHECK_EQ(fin.ndim(), 1U);
    CHECK_EQ(over_max.ndim(), 1U);

    CHECK_EQ(dist[0], scores[0]);
    CHECK_EQ(dist[0], fin[0]);
    CHECK_EQ(dist[0], over_max[0]);
    CHECK_EQ(scores[1], 1);

    mxnet::TShape score_shape(dist.ndim(), -1);
    score_shape[0] = dist[0];
    score_shape[1] = 1;

    mxnet::TShape bool_shape(dist.ndim() - 1, -1);
    bool_shape[0] = dist[0];

    SHAPE_ASSIGN_CHECK(*out_attrs, 0, dist);
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(beamsearch_set_finished::kOut));
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, score_shape);
    SHAPE_ASSIGN_CHECK(*in_attrs, 2, bool_shape);
    SHAPE_ASSIGN_CHECK(*in_attrs, 3, bool_shape);

    return true;
}

inline bool BeamsearchSetFinishedType(const nnvm::NodeAttrs& attrs,
                                      std::vector<int>* in_attrs,
                                      std::vector<int>* out_attrs) {
    CHECK_EQ(in_attrs->size(), 4U);
    CHECK_EQ(out_attrs->size(), 1U);

    TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
    TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
    TYPE_ASSIGN_CHECK(*in_attrs, 1, (*out_attrs)[0]);
    TYPE_ASSIGN_CHECK(*in_attrs, 2, mshadow::kInt32);
    TYPE_ASSIGN_CHECK(*in_attrs, 3, mshadow::kInt32);
    return (*in_attrs)[0] != -1 && (*in_attrs)[1] != -1;
}

template<typename xpu>
void NoopGrad(const nnvm::NodeAttrs& attrs,
              const OpContext& ctx,
              const std::vector<TBlob>& inputs,
              const std::vector<OpReqType>& req,
              const std::vector<TBlob>& outputs) {
    LOG(FATAL) << "This operator should only be used for inference";
}

template<typename xpu>
void BeamsearchSetFinishedForward(const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const std::vector<TBlob>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<TBlob>& outputs) {
    if (req[beamsearch_set_finished::kOut] == mxnet::kNullOp) return;
    const BeamsearchSetFinishedParam& param = nnvm::get<BeamsearchSetFinishedParam>(attrs.parsed);
    CHECK_EQ(inputs.size(), 4U);
    CHECK_EQ(outputs.size(), 1U);
    CHECK_EQ(req.size(), 1U);

    const mxnet::TShape& out_shape = outputs[beamsearch_set_finished::kOut].shape_;
    const mxnet::TShape& batch_beam_shape = inputs[beamsearch_set_finished::kFin].shape_;

    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    using namespace mxnet_op;
    MSHADOW_TYPE_SWITCH(outputs[beamsearch_set_finished::kOut].type_flag_, DType, {
        DType mask_val = param.mask_val;
        const int score_idx = param.score_idx;
        const int eos_idx = param.eos_idx;
        Kernel<beamsearch_set_finished_forward, xpu>::Launch(s, out_shape.Size(),
            outputs[beamsearch_set_finished::kOut].dptr<DType>(),
            inputs[beamsearch_set_finished::kDist].dptr<DType>(),
            inputs[beamsearch_set_finished::kScores].dptr<DType>(),
            inputs[beamsearch_set_finished::kFin].dptr<int>(),
            inputs[beamsearch_set_finished::kOverMax].dptr<int>(),
            mask_val, score_idx, eos_idx, out_shape.Size()/batch_beam_shape.Size());
    });

}


}
}
