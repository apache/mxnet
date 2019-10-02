/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * Copyright (c) 2019 by Contributors
 * \file legacy_nnvm_attrs.cc
 * \author Junru Shao
 */
#if MXNET_USE_TVM_OP && !defined MXNET_AMALGAMATION
#include "../../include/ir.h"
#include "../../include/op/attrs/legacy_nnvm_attrs.h"

namespace mxnet {
namespace v3 {
namespace op {
namespace attrs {
namespace {
// Skip empty attribute LegacyCopytoAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyAllFiniteAttrs);
// Skip empty attribute LegacyNpiDeg2radAttrs
// Skip empty attribute LegacyNpiRad2degAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyIdentityAttachKLSparseRegAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLeakyReLUAttrs);
// Skip empty attribute LegacySoftmaxCrossEntropyAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyActivationAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyBatchNormAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyConvolutionAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyCTCLossAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyDeconvolutionAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyDropoutAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyFullyConnectedAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyGroupNormAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLayerNormAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLogSoftmaxAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLRNAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyMomentsAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyPoolingAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySoftmaxAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySoftmaxActivationAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySoftminAttrs);
// Skip empty attribute LegacyNpLinalgSvdAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiBooleanMaskAssignScalarAttrs);
// Skip empty attribute LegacyNpiBooleanMaskAssignTensorAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiArgmaxAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpSumAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpMaxAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpMinAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpProdAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiMeanAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiStdAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiVarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpBroadcastToAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpCumsumAttrs);
// Skip empty attribute LegacyNpDotAttrs
// Skip empty attribute LegacyNpiAddAttrs
// Skip empty attribute LegacyNpiSubtractAttrs
// Skip empty attribute LegacyNpiMultiplyAttrs
// Skip empty attribute LegacyNpiModAttrs
// Skip empty attribute LegacyNpiPowerAttrs
// Skip empty attribute LegacyNpiCopysignAttrs
// Skip empty attribute LegacyNpiLcmAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiAddScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiSubtractScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiRsubtractScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiMultiplyScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiModScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiRmodScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiPowerScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiRpowerScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiCopysignScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiRcopysignScalarAttrs);
// Skip empty attribute LegacyNpiArctan2Attrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiArctan2ScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiRarctan2ScalarAttrs);
// Skip empty attribute LegacyNpiHypotAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiLcmScalarAttrs);
// Skip empty attribute LegacyNpiLdexpAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiLdexpScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiRldexpScalarAttrs);
// Skip empty attribute LegacyNpxReluAttrs
// Skip empty attribute LegacyNpxSigmoidAttrs
// Skip empty attribute LegacyNpCopyAttrs
// Skip empty attribute LegacyNpiNegativeAttrs
// Skip empty attribute LegacyNpiReciprocalAttrs
// Skip empty attribute LegacyNpiAbsoluteAttrs
// Skip empty attribute LegacyNpiSignAttrs
// Skip empty attribute LegacyNpiRintAttrs
// Skip empty attribute LegacyNpiCeilAttrs
// Skip empty attribute LegacyNpiFloorAttrs
// Skip empty attribute LegacyNpiTruncAttrs
// Skip empty attribute LegacyNpiFixAttrs
// Skip empty attribute LegacyNpiSquareAttrs
// Skip empty attribute LegacyNpiSqrtAttrs
// Skip empty attribute LegacyNpiCbrtAttrs
// Skip empty attribute LegacyNpiExpAttrs
// Skip empty attribute LegacyNpiLogAttrs
// Skip empty attribute LegacyNpiLog10Attrs
// Skip empty attribute LegacyNpiLog2Attrs
// Skip empty attribute LegacyNpiLog1pAttrs
// Skip empty attribute LegacyNpiExpm1Attrs
// Skip empty attribute LegacyNpiLogicalNotAttrs
// Skip empty attribute LegacyNpiSinAttrs
// Skip empty attribute LegacyNpiCosAttrs
// Skip empty attribute LegacyNpiTanAttrs
// Skip empty attribute LegacyNpiArcsinAttrs
// Skip empty attribute LegacyNpiArccosAttrs
// Skip empty attribute LegacyNpiArctanAttrs
// Skip empty attribute LegacyNpiDegreesAttrs
// Skip empty attribute LegacyNpiRadiansAttrs
// Skip empty attribute LegacyNpiSinhAttrs
// Skip empty attribute LegacyNpiCoshAttrs
// Skip empty attribute LegacyNpiTanhAttrs
// Skip empty attribute LegacyNpiArcsinhAttrs
// Skip empty attribute LegacyNpiArccoshAttrs
// Skip empty attribute LegacyNpiArctanhAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiAroundAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiZerosAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiOnesAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiIdentityAttrs);
// Skip empty attribute LegacyNpZerosLikeAttrs
// Skip empty attribute LegacyNpOnesLikeAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiArangeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiIndicesAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpTransposeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpReshapeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpSqueezeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpRollAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiFlipAttrs);
// Skip empty attribute LegacyNpxNonzeroAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiTensordotAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiTensordotIntAxesAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpTraceAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiTrilAttrs);
// Skip empty attribute LegacyNpiTrueDivideAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiTrueDivideScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiRtrueDivideScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiUniqueAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiHanningAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiHammingAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiBlackmanAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiNormalAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiUniformAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyPadAttrs);
// Skip empty attribute LegacyFlattenAttrs
MX_V3_REGISTER_NODE_TYPE(LegacySampleUniformAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySampleNormalAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySampleGammaAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySampleExponentialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySamplePoissonAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySampleNegativeBinomialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySampleGeneralizedNegativeBinomialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPdfUniformAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPdfNormalAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPdfGammaAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPdfExponentialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPdfPoissonAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPdfNegativeBinomialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPdfGeneralizedNegativeBinomialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPdfDirichletAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySampleMultinomialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomUniformAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomNormalAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomGammaAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomExponentialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPoissonAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomNegativeBinomialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomGeneralizedNegativeBinomialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomRandintAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomUniformLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomNormalLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomGammaLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomExponentialLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPoissonLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomNegativeBinomialLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomGeneralizedNegativeBinomialLikeAttrs);
// Skip empty attribute LegacyShuffleAttrs
MX_V3_REGISTER_NODE_TYPE(LegacySampleUniqueZipfianAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLinearRegressionOutputAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyMAERegressionOutputAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLogisticRegressionOutputAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRNNAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyROIPoolingAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySequenceMaskAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySliceChannelAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySoftmaxOutputAttrs);
// Skip empty attribute LegacySgMkldnnConvAttrs
// Skip empty attribute LegacySgMkldnnFullyConnectedAttrs
MX_V3_REGISTER_NODE_TYPE(LegacySwapAxisAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyMaxAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyMinAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNormAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyArgmaxAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyArgminAttrs);
// Skip empty attribute LegacyArgmaxChannelAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyPickAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyBroadcastAxisAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyBroadcastToAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyBroadcastLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyProdAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNanprodAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySumAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyMeanAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNansumAttrs);
// Skip empty attribute LegacyWhereAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyDiagAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyDotAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyBatchDotAttrs);
// Skip empty attribute LegacyBroadcastAddAttrs
// Skip empty attribute LegacyBroadcastSubAttrs
// Skip empty attribute LegacyBroadcastMulAttrs
// Skip empty attribute LegacyBroadcastDivAttrs
// Skip empty attribute LegacyBroadcastModAttrs
// Skip empty attribute LegacyBroadcastPowerAttrs
// Skip empty attribute LegacyBroadcastMaximumAttrs
// Skip empty attribute LegacyBroadcastMinimumAttrs
// Skip empty attribute LegacyBroadcastHypotAttrs
// Skip empty attribute LegacyBroadcastEqualAttrs
// Skip empty attribute LegacyBroadcastNotEqualAttrs
// Skip empty attribute LegacyBroadcastGreaterAttrs
// Skip empty attribute LegacyBroadcastGreaterEqualAttrs
// Skip empty attribute LegacyBroadcastLesserAttrs
// Skip empty attribute LegacyBroadcastLesserEqualAttrs
// Skip empty attribute LegacyBroadcastLogicalAndAttrs
// Skip empty attribute LegacyBroadcastLogicalOrAttrs
// Skip empty attribute LegacyBroadcastLogicalXorAttrs
// Skip empty attribute LegacyElemwiseAddAttrs
// Skip empty attribute LegacyGradAddAttrs
// Skip empty attribute LegacyElemwiseSubAttrs
// Skip empty attribute LegacyElemwiseMulAttrs
// Skip empty attribute LegacyElemwiseDivAttrs
// Skip empty attribute LegacyModAttrs
// Skip empty attribute LegacyPowerAttrs
// Skip empty attribute LegacyMaximumAttrs
// Skip empty attribute LegacyMinimumAttrs
// Skip empty attribute LegacyHypotAttrs
// Skip empty attribute LegacyEqualAttrs
// Skip empty attribute LegacyNotEqualAttrs
// Skip empty attribute LegacyGreaterAttrs
// Skip empty attribute LegacyGreaterEqualAttrs
// Skip empty attribute LegacyLesserAttrs
// Skip empty attribute LegacyLesserEqualAttrs
// Skip empty attribute LegacyLogicalAndAttrs
// Skip empty attribute LegacyLogicalOrAttrs
// Skip empty attribute LegacyLogicalXorAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyPlusScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyMinusScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRminusScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyMulScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyDivScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRdivScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyModScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRmodScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyMaximumScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyMinimumScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyPowerScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRpowerScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyHypotScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySmoothL1Attrs);
MX_V3_REGISTER_NODE_TYPE(LegacyEqualScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNotEqualScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyGreaterScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyGreaterEqualScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLesserScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLesserEqualScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLogicalAndScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLogicalOrScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLogicalXorScalarAttrs);
// Skip empty attribute LegacyScatterElemwiseDivAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyScatterPlusScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyScatterMinusScalarAttrs);
// Skip empty attribute LegacyReluAttrs
// Skip empty attribute LegacySigmoidAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyHardSigmoidAttrs);
// Skip empty attribute LegacySoftsignAttrs
// Skip empty attribute LegacyCopyAttrs
// Skip empty attribute LegacyMakeLossAttrs
// Skip empty attribute LegacyIdentityWithAttrLikeRhsAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyReshapeLikeAttrs);
// Skip empty attribute LegacyShapeArrayAttrs
// Skip empty attribute LegacySizeArrayAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyCastAttrs);
// Skip empty attribute LegacyNegativeAttrs
// Skip empty attribute LegacyAbsAttrs
// Skip empty attribute LegacySignAttrs
// Skip empty attribute LegacyRoundAttrs
// Skip empty attribute LegacyRintAttrs
// Skip empty attribute LegacyCeilAttrs
// Skip empty attribute LegacyFloorAttrs
// Skip empty attribute LegacyTruncAttrs
// Skip empty attribute LegacyFixAttrs
// Skip empty attribute LegacyErfAttrs
// Skip empty attribute LegacyErfinvAttrs
// Skip empty attribute LegacyGammaAttrs
// Skip empty attribute LegacyGammalnAttrs
// Skip empty attribute LegacyLogicalNotAttrs
// Skip empty attribute LegacyExpAttrs
// Skip empty attribute LegacyLogAttrs
// Skip empty attribute LegacyLog10Attrs
// Skip empty attribute LegacyLog2Attrs
// Skip empty attribute LegacyLog1pAttrs
// Skip empty attribute LegacyExpm1Attrs
// Skip empty attribute LegacyReciprocalAttrs
// Skip empty attribute LegacySquareAttrs
// Skip empty attribute LegacySqrtAttrs
// Skip empty attribute LegacyRsqrtAttrs
// Skip empty attribute LegacyCbrtAttrs
// Skip empty attribute LegacyRcbrtAttrs
// Skip empty attribute LegacySinAttrs
// Skip empty attribute LegacyCosAttrs
// Skip empty attribute LegacyTanAttrs
// Skip empty attribute LegacyArcsinAttrs
// Skip empty attribute LegacyArccosAttrs
// Skip empty attribute LegacyArctanAttrs
// Skip empty attribute LegacyDegreesAttrs
// Skip empty attribute LegacyRadiansAttrs
// Skip empty attribute LegacySinhAttrs
// Skip empty attribute LegacyCoshAttrs
// Skip empty attribute LegacyTanhAttrs
// Skip empty attribute LegacyArcsinhAttrs
// Skip empty attribute LegacyArccoshAttrs
// Skip empty attribute LegacyArctanhAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyEmbeddingAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyTakeAttrs);
// Skip empty attribute LegacyBatchTakeAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyOneHotAttrs);
// Skip empty attribute LegacyGatherNdAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyScatterNdAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyScatterSetNdAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyZerosWithoutDtypeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyZerosAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyEyeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyOnesAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyFullAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyArangeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLinspaceAttrs);
// Skip empty attribute LegacyZerosLikeAttrs
// Skip empty attribute LegacyOnesLikeAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgGemmAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgGemm2Attrs);
// Skip empty attribute LegacyLinalgPotrfAttrs
// Skip empty attribute LegacyLinalgPotriAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgTrmmAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgTrsmAttrs);
// Skip empty attribute LegacyLinalgSumlogdiagAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgExtractdiagAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgMakediagAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgExtracttrianAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgMaketrianAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgSyrkAttrs);
// Skip empty attribute LegacyLinalgGelqfAttrs
// Skip empty attribute LegacyLinalgSyevdAttrs
// Skip empty attribute LegacyLinalgInverseAttrs
// Skip empty attribute LegacyLinalgDetAttrs
// Skip empty attribute LegacyLinalgSlogdetAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyReshapeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyTransposeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyExpandDimsAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySliceAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySliceAssignAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySliceAssignScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySliceAxisAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySliceLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyClipAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRepeatAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyTileAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyReverseAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySqueezeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyDepthToSpaceAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySpaceToDepthAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySplitV2Attrs);
MX_V3_REGISTER_NODE_TYPE(LegacySortAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyArgsortAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRavelMultiIndexAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyUnravelIndexAttrs);
// Skip empty attribute LegacySparseRetainAttrs
MX_V3_REGISTER_NODE_TYPE(LegacySquareSumAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyBilinearSamplerAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyCorrelationAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyInstanceNormAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyL2NormalizationAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySequenceLastAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySequenceReverseAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySpatialTransformerAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySetValueAttrs);
// Skip empty attribute LegacyOnehotEncodeAttrs
}  // namespace
}  // namespace attrs
}  // namespace op
}  // namespace v3
}  // namespace mxnet
#endif
