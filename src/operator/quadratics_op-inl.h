//
// Created by Zai, Alexander on 5/22/18.
//


#include "operator_common.h"

struct QuadraticParam : public dmlc::Parameter<QuadraticParam> {
    float a, b, c;
    DMLC_DECLARE_PARAMETER(QuadraticParam) {
        DMLC_DECLARE_FIELD(a)
           .set_default(0.0)
           .describe("Coefficient of the quadratic term in the quadratic function.");
        DMLC_DECLARE_FIELD(b)
            .set_default(0.0)
            .describe("Coefficinet of the linear term in the quadratic function.");
        DMLC_DECLARE_FIELD(c)
           .set_default(0.0)
           .describe("Constant term in the quadratic function.");
    }
};


inline bool QuadraticOpShape(const nnvm::NodeAttrs& attrs, std::vector<TShape>* in_attrs, std::vector<TShape>* out_attrs) {
    CHECK_EQ(in_attrs->size(), 1U);
    CHECK_EQ(out_attrs->size(), 1U);

    SHAPE_ASSIGN_CHECK
}

#endif //QUADRATIC_OPERATOR_QUADRATIC_OP_INL_H
