using System;
using System.Collections.Generic;
using NumpyDotNet;

namespace MxNet.Contrib.ONNX.Mx2ONNX
{
    public class MXNetGraph
    {
        public MXNetGraph()
        {
            throw new NotImplementedRelease2Exception();
        }

        public static void Register(Func<dynamic, FuncArgs, dynamic> func)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static dynamic ConvertLayer(dynamic node, FuncArgs kwargs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static (NDArrayDict, NDArrayDict) SplitParams(Symbol sym, NDArrayDict @params)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static Dictionary<string, ndarray> ConvertWeightsToNumpy(NDArrayDict weights_dict)
        {
            throw new NotImplementedRelease2Exception();
        }

        public dynamic CreateOnnxGraphProto(Symbol sym, NDArrayDict @params, Shape in_shape, DType in_type, bool verbose= false, int? opset_version= null)
        {
            throw new NotImplementedRelease2Exception();
        }
    }
}
