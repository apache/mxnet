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
#include "../../include/op/attrs/legacy_nnvm_attrs.h"

#include "../../include/ir.h"

namespace mxnet {
namespace v3 {
namespace op {
namespace attrs {
namespace {
// Skip empty attribute LegacyAbsAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyActivationAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyArangeAttrs);
// Skip empty attribute LegacyArccosAttrs
// Skip empty attribute LegacyArccoshAttrs
// Skip empty attribute LegacyArcsinAttrs
// Skip empty attribute LegacyArcsinhAttrs
// Skip empty attribute LegacyArctanAttrs
// Skip empty attribute LegacyArctanhAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyArgmaxAttrs);
// Skip empty attribute LegacyArgmaxChannelAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyArgminAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyArgsortAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyBatchDotAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyBatchNormAttrs);
// Skip empty attribute LegacyBatchTakeAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyBilinearSamplerAttrs);
// Skip empty attribute LegacyBroadcastAddAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyBroadcastAxisAttrs);
// Skip empty attribute LegacyBroadcastDivAttrs
// Skip empty attribute LegacyBroadcastEqualAttrs
// Skip empty attribute LegacyBroadcastGreaterAttrs
// Skip empty attribute LegacyBroadcastGreaterEqualAttrs
// Skip empty attribute LegacyBroadcastHypotAttrs
// Skip empty attribute LegacyBroadcastLesserAttrs
// Skip empty attribute LegacyBroadcastLesserEqualAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyBroadcastLikeAttrs);
// Skip empty attribute LegacyBroadcastLogicalAndAttrs
// Skip empty attribute LegacyBroadcastLogicalOrAttrs
// Skip empty attribute LegacyBroadcastLogicalXorAttrs
// Skip empty attribute LegacyBroadcastMaximumAttrs
// Skip empty attribute LegacyBroadcastMinimumAttrs
// Skip empty attribute LegacyBroadcastModAttrs
// Skip empty attribute LegacyBroadcastMulAttrs
// Skip empty attribute LegacyBroadcastNotEqualAttrs
// Skip empty attribute LegacyBroadcastPowerAttrs
// Skip empty attribute LegacyBroadcastSubAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyBroadcastToAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyCTCLossAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyCastAttrs);
// Skip empty attribute LegacyCbrtAttrs
// Skip empty attribute LegacyCeilAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyClipAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyConvolutionAttrs);
// Skip empty attribute LegacyCopyAttrs
// Skip empty attribute LegacyCopytoAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyCorrelationAttrs);
// Skip empty attribute LegacyCosAttrs
// Skip empty attribute LegacyCoshAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyDeconvolutionAttrs);
// Skip empty attribute LegacyDegreesAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyDepthToSpaceAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyDiagAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyDivScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyDotAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyDropoutAttrs);
// Skip empty attribute LegacyElemwiseAddAttrs
// Skip empty attribute LegacyElemwiseDivAttrs
// Skip empty attribute LegacyElemwiseMulAttrs
// Skip empty attribute LegacyElemwiseSubAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyEmbeddingAttrs);
// Skip empty attribute LegacyEqualAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyEqualScalarAttrs);
// Skip empty attribute LegacyErfAttrs
// Skip empty attribute LegacyErfinvAttrs
// Skip empty attribute LegacyExpAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyExpandDimsAttrs);
// Skip empty attribute LegacyExpm1Attrs
MX_V3_REGISTER_NODE_TYPE(LegacyEyeAttrs);
// Skip empty attribute LegacyFixAttrs
// Skip empty attribute LegacyFlattenAttrs
// Skip empty attribute LegacyFloorAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyFullAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyFullyConnectedAttrs);
// Skip empty attribute LegacyGammaAttrs
// Skip empty attribute LegacyGammalnAttrs
// Skip empty attribute LegacyGatherNdAttrs
// Skip empty attribute LegacyGradAddAttrs
// Skip empty attribute LegacyGreaterAttrs
// Skip empty attribute LegacyGreaterEqualAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyGreaterEqualScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyGreaterScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyGroupNormAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyHardSigmoidAttrs);
// Skip empty attribute LegacyHypotAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyHypotScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyIdentityAttachKLSparseRegAttrs);
// Skip empty attribute LegacyIdentityWithAttrLikeRhsAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyInstanceNormAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyL2NormalizationAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLRNAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLayerNormAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLeakyReLUAttrs);
// Skip empty attribute LegacyLesserAttrs
// Skip empty attribute LegacyLesserEqualAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyLesserEqualScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLesserScalarAttrs);
// Skip empty attribute LegacyLinalgDetAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgExtractdiagAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgExtracttrianAttrs);
// Skip empty attribute LegacyLinalgGelqfAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgGemmAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgGemm2Attrs);
// Skip empty attribute LegacyLinalgInverseAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgMakediagAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgMaketrianAttrs);
// Skip empty attribute LegacyLinalgPotrfAttrs
// Skip empty attribute LegacyLinalgPotriAttrs
// Skip empty attribute LegacyLinalgSlogdetAttrs
// Skip empty attribute LegacyLinalgSumlogdiagAttrs
// Skip empty attribute LegacyLinalgSyevdAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgSyrkAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgTrmmAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLinalgTrsmAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLinearRegressionOutputAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLinspaceAttrs);
// Skip empty attribute LegacyLogAttrs
// Skip empty attribute LegacyLog10Attrs
// Skip empty attribute LegacyLog1pAttrs
// Skip empty attribute LegacyLog2Attrs
MX_V3_REGISTER_NODE_TYPE(LegacyLogSoftmaxAttrs);
// Skip empty attribute LegacyLogicalAndAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyLogicalAndScalarAttrs);
// Skip empty attribute LegacyLogicalNotAttrs
// Skip empty attribute LegacyLogicalOrAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyLogicalOrScalarAttrs);
// Skip empty attribute LegacyLogicalXorAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyLogicalXorScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyLogisticRegressionOutputAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyMAERegressionOutputAttrs);
// Skip empty attribute LegacyMakeLossAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyMaxAttrs);
// Skip empty attribute LegacyMaximumAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyMaximumScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyMeanAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyMinAttrs);
// Skip empty attribute LegacyMinimumAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyMinimumScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyMinusScalarAttrs);
// Skip empty attribute LegacyModAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyModScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyMomentsAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyMulScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNanprodAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNansumAttrs);
// Skip empty attribute LegacyNegativeAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNormAttrs);
// Skip empty attribute LegacyNotEqualAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNotEqualScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpBroadcastToAttrs);
// Skip empty attribute LegacyNpCopyAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpCumsumAttrs);
// Skip empty attribute LegacyNpDotAttrs
// Skip empty attribute LegacyNpLinalgSvdAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpMaxAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpMinAttrs);
// Skip empty attribute LegacyNpOnesLikeAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpProdAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpReshapeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpRollAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpSqueezeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpSumAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpTraceAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpTransposeAttrs);
// Skip empty attribute LegacyNpZerosLikeAttrs
// Skip empty attribute LegacyNpiAbsoluteAttrs
// Skip empty attribute LegacyNpiAddAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiAddScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiArangeAttrs);
// Skip empty attribute LegacyNpiArccosAttrs
// Skip empty attribute LegacyNpiArccoshAttrs
// Skip empty attribute LegacyNpiArcsinAttrs
// Skip empty attribute LegacyNpiArcsinhAttrs
// Skip empty attribute LegacyNpiArctanAttrs
// Skip empty attribute LegacyNpiArctan2Attrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiArctan2ScalarAttrs);
// Skip empty attribute LegacyNpiArctanhAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiArgmaxAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiAroundAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiBooleanMaskAssignScalarAttrs);
// Skip empty attribute LegacyNpiBooleanMaskAssignTensorAttrs
// Skip empty attribute LegacyNpiCbrtAttrs
// Skip empty attribute LegacyNpiCeilAttrs
// Skip empty attribute LegacyNpiCopysignAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiCopysignScalarAttrs);
// Skip empty attribute LegacyNpiCosAttrs
// Skip empty attribute LegacyNpiCoshAttrs
// Skip empty attribute LegacyNpiDeg2radAttrs
// Skip empty attribute LegacyNpiDegreesAttrs
// Skip empty attribute LegacyNpiEqualAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiEqualScalarAttrs);
// Skip empty attribute LegacyNpiExpAttrs
// Skip empty attribute LegacyNpiExpm1Attrs
// Skip empty attribute LegacyNpiFixAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiFlipAttrs);
// Skip empty attribute LegacyNpiFloorAttrs
// Skip empty attribute LegacyNpiGreaterAttrs
// Skip empty attribute LegacyNpiGreaterEqualAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiGreaterEqualScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiGreaterScalarAttrs);
// Skip empty attribute LegacyNpiHypotAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiIdentityAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiIndicesAttrs);
// Skip empty attribute LegacyNpiLcmAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiLcmScalarAttrs);
// Skip empty attribute LegacyNpiLdexpAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiLdexpScalarAttrs);
// Skip empty attribute LegacyNpiLessAttrs
// Skip empty attribute LegacyNpiLessEqualAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiLessEqualScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiLessScalarAttrs);
// Skip empty attribute LegacyNpiLogAttrs
// Skip empty attribute LegacyNpiLog10Attrs
// Skip empty attribute LegacyNpiLog1pAttrs
// Skip empty attribute LegacyNpiLog2Attrs
// Skip empty attribute LegacyNpiLogicalNotAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiMeanAttrs);
// Skip empty attribute LegacyNpiModAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiModScalarAttrs);
// Skip empty attribute LegacyNpiMultiplyAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiMultiplyScalarAttrs);
// Skip empty attribute LegacyNpiNegativeAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiNormalAttrs);
// Skip empty attribute LegacyNpiNotEqualAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiNotEqualScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiOnesAttrs);
// Skip empty attribute LegacyNpiPowerAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiPowerScalarAttrs);
// Skip empty attribute LegacyNpiRad2degAttrs
// Skip empty attribute LegacyNpiRadiansAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiRarctan2ScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiRcopysignScalarAttrs);
// Skip empty attribute LegacyNpiReciprocalAttrs
// Skip empty attribute LegacyNpiRintAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiRldexpScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiRmodScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiRpowerScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiRsubtractScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiRtrueDivideScalarAttrs);
// Skip empty attribute LegacyNpiSignAttrs
// Skip empty attribute LegacyNpiSinAttrs
// Skip empty attribute LegacyNpiSinhAttrs
// Skip empty attribute LegacyNpiSqrtAttrs
// Skip empty attribute LegacyNpiSquareAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiStdAttrs);
// Skip empty attribute LegacyNpiSubtractAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiSubtractScalarAttrs);
// Skip empty attribute LegacyNpiTanAttrs
// Skip empty attribute LegacyNpiTanhAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiTensordotAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiTensordotIntAxesAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiTrilAttrs);
// Skip empty attribute LegacyNpiTrueDivideAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiTrueDivideScalarAttrs);
// Skip empty attribute LegacyNpiTruncAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyNpiUniformAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiUniqueAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiVarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyNpiZerosAttrs);
// Skip empty attribute LegacyNpxNonzeroAttrs
// Skip empty attribute LegacyNpxReluAttrs
// Skip empty attribute LegacyNpxSigmoidAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyOneHotAttrs);
// Skip empty attribute LegacyOnehotEncodeAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyOnesAttrs);
// Skip empty attribute LegacyOnesLikeAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyPadAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyPickAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyPlusScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyPoolingAttrs);
// Skip empty attribute LegacyPowerAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyPowerScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyProdAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRNNAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyROIPoolingAttrs);
// Skip empty attribute LegacyRadiansAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyRandomExponentialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomExponentialLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomGammaAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomGammaLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomGeneralizedNegativeBinomialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomGeneralizedNegativeBinomialLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomNegativeBinomialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomNegativeBinomialLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomNormalAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomNormalLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPdfDirichletAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPdfExponentialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPdfGammaAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPdfGeneralizedNegativeBinomialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPdfNegativeBinomialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPdfNormalAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPdfPoissonAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPdfUniformAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPoissonAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomPoissonLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomRandintAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomUniformAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRandomUniformLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRavelMultiIndexAttrs);
// Skip empty attribute LegacyRcbrtAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyRdivScalarAttrs);
// Skip empty attribute LegacyReciprocalAttrs
// Skip empty attribute LegacyReluAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyRepeatAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyReshapeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyReshapeLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyReverseAttrs);
// Skip empty attribute LegacyRintAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyRminusScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyRmodScalarAttrs);
// Skip empty attribute LegacyRoundAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyRpowerScalarAttrs);
// Skip empty attribute LegacyRsqrtAttrs
MX_V3_REGISTER_NODE_TYPE(LegacySampleExponentialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySampleGammaAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySampleGeneralizedNegativeBinomialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySampleMultinomialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySampleNegativeBinomialAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySampleNormalAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySamplePoissonAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySampleUniformAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySampleUniqueZipfianAttrs);
// Skip empty attribute LegacyScatterElemwiseDivAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyScatterMinusScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyScatterNdAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyScatterPlusScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyScatterSetNdAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySequenceLastAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySequenceMaskAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySequenceReverseAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySetValueAttrs);
// Skip empty attribute LegacySgMkldnnConvAttrs
// Skip empty attribute LegacySgMkldnnFullyConnectedAttrs
// Skip empty attribute LegacyShapeArrayAttrs
// Skip empty attribute LegacyShuffleAttrs
// Skip empty attribute LegacySigmoidAttrs
// Skip empty attribute LegacySignAttrs
// Skip empty attribute LegacySinAttrs
// Skip empty attribute LegacySinhAttrs
// Skip empty attribute LegacySizeArrayAttrs
MX_V3_REGISTER_NODE_TYPE(LegacySliceAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySliceAssignAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySliceAssignScalarAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySliceAxisAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySliceChannelAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySliceLikeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySmoothL1Attrs);
MX_V3_REGISTER_NODE_TYPE(LegacySoftmaxAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySoftmaxActivationAttrs);
// Skip empty attribute LegacySoftmaxCrossEntropyAttrs
MX_V3_REGISTER_NODE_TYPE(LegacySoftmaxOutputAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySoftminAttrs);
// Skip empty attribute LegacySoftsignAttrs
MX_V3_REGISTER_NODE_TYPE(LegacySortAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySpaceToDepthAttrs);
// Skip empty attribute LegacySparseRetainAttrs
MX_V3_REGISTER_NODE_TYPE(LegacySpatialTransformerAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySplitV2Attrs);
// Skip empty attribute LegacySqrtAttrs
// Skip empty attribute LegacySquareAttrs
MX_V3_REGISTER_NODE_TYPE(LegacySquareSumAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySqueezeAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySumAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacySwapAxisAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyTakeAttrs);
// Skip empty attribute LegacyTanAttrs
// Skip empty attribute LegacyTanhAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyTileAttrs);
MX_V3_REGISTER_NODE_TYPE(LegacyTransposeAttrs);
// Skip empty attribute LegacyTruncAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyUnravelIndexAttrs);
// Skip empty attribute LegacyWhereAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyZerosAttrs);
// Skip empty attribute LegacyZerosLikeAttrs
MX_V3_REGISTER_NODE_TYPE(LegacyZerosWithoutDtypeAttrs);

}  // namespace
}  // namespace attrs
}  // namespace op
}  // namespace v3
}  // namespace mxnet
#endif
