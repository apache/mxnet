using System.Collections.Generic;
using Onnx;

namespace MxNet.Contrib.ONNX.ONNX2Mx
{
    public class ImportModel
    {
        public static (Symbol, NDArrayDict, NDArrayDict) Import(string model_file)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static Dictionary<string, ValueInfoProto[]> GetModelMetadata(string model_file)
        {
            throw new NotImplementedRelease2Exception();
        }
    }
}