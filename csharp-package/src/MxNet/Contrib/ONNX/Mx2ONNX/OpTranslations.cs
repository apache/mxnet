using System;
using System.Collections.Generic;
using System.Text;
using Onnx;
    
namespace MxNet.Contrib.ONNX.Mx2ONNX
{
    public class OpTranslations
    {
        public static T ParseHelper<T>(Dictionary<string, object> attrs, string attrs_name, T alt_value= default(T))
        {
            throw new NotImplementedRelease2Exception();
        }

        public static int TransformPadding(int pad_width)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static string[] ConvertStringToList(string string_val)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static int GetBooleanAttributeValue(Dictionary<string, object> attrs, string attr_name)
        {
            throw new NotImplementedRelease2Exception();
        }

        public  static (string, NodeProto[], Dictionary<string,object>) GetInputs(NodeProto node, FuncArgs kwargs, bool with_shapes= false)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto CreateBasicOpNode(string op_name, NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertWeightsAndInputs(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertConvolution(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertDeconvolution(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertCrop(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertFullyConnected(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertBatchNorm(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertTanh(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertCos(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertSin(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertTan(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertAcos(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertAsin(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertAtan(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertSigmoid(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertRelu(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertActivation(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertPad(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto CreateHelperTensorNode(NDArray input_vals, string output_name, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto CreateHelperReshapeNode(string input_name, string output_name, Shape shape, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto CreateHelperTransNode(string input_name, string output_name, int[] perm = null)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto CreateHelperConcatNode(NDArrayList inputs, string output_name, int axis=0)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto CreateHelperExpandNode(string input_name, string output_name, Shape expand_shape)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto CreateHelperGatherNode(string input_name, string output_name, NDArray indices, FuncArgs kwargs, int? axis = null)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto CreateHelperBuildValuesNode(NDArrayList inputs, string output_name, DType dtype, FuncArgs kwargs, int axis = 0)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto CreateHelperShapeNode(string input_name, string output_name)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertDot(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }


        public static NodeProto ConvertLinalgGemm2(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertPooling(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertExp(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertCopy(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertIdentity(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertInstantNorm(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertLeakyRelu(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertSoftmax(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertBlockgrad(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertMakeloss(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertConcat(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertRNN(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertRNNParamConcat(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertFull(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertTranspose(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertLrn(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertL2Normalization(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertDropout(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertFlatten(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertClip(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto scalar_op_helper(NodeProto node, string op_name, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertMulscalar(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertMinusScalar(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertRminusScalar(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertAddScalar(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertDivScalar(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertRdivScalar(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertPowScalar(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertArgmax(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertArgmin(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertMaximum(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertMinimum(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertMin(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertMax(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertMean(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertProd(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertElementwiseAdd(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertBroadcastAdd(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertElementwiseSub(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertBroadcastSub(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertElementwiseMul(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertBroadcastMul(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertElementwiseDiv(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertBroadcastDiv(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertNegative(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertAbs(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertAddN(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertFloor(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertReshape(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertCast(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertSliceAxis(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertSliceChannel(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertExpandDims(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertSqueeze(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertLog(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertReciprocal(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertPower(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertBroatcastPower(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertDepthToSpace(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertSpaceToDepth(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertSquare(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertSum(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertShape(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertHardSigmoid(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }
        public static NodeProto ConvertBroatcastGreater(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertBroatcastLesser(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertBroatcastEqual(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertBroatcastLogicalAnd(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertBroatcastLogicalOr(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertBroatcastLogicalXor(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertLogicalNot(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertSize(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertLogSoftmax(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertNorm(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertMultinomial(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertRandomNormal(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertROIPooling(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertTile(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertBroatcastTo(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertTopK(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static NodeProto ConvertTake(NodeProto node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }
    }
}
