

using System.Collections.Generic;
using MxNet.Gluon;
using Onnx;

namespace MxNet.Contrib.ONNX.ONNX2Mx
{
    public  class MxGraphProto
    {
        public MxGraphProto()
        {
            throw new NotImplementedRelease2Exception();
        }

        private Symbol _convert_operator(string node_name, string op_name, Dictionary<string, object> attrs, NDArrayList inputs)
        {
            throw new NotImplementedRelease2Exception();
        }

        public (Symbol, NDArrayDict) FromOnnx(GraphProto graph, int opset_version)
        {
            throw new NotImplementedRelease2Exception();
        }

        public Dictionary<string, List<(string, Shape)>> GetGraphMetadata(GraphProto graph)
        {
            throw new NotImplementedRelease2Exception();
        }

        public SymbolBlock GraphToGluon(GraphProto graph, Context[] ctx, int opset_version)
        {
            throw new NotImplementedRelease2Exception();
        }
        
        public SymbolBlock GraphToGluon(GraphProto graph, Context ctx, int opset_version)
        {
            return GraphToGluon(graph, new Context[] {ctx}, opset_version);
        }

        public NDArrayList ParseArray(TensorProto tensor_proto)
        {
            throw new NotImplementedRelease2Exception();
        }
        
        public Dictionary<string, object> ParseAttr(AttributeProto attr_proto)
        {
            throw new NotImplementedRelease2Exception();
        }
    }
}