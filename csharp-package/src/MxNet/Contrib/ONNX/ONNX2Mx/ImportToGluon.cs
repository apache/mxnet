using System;
using MxNet.Gluon;

namespace MxNet.Contrib.ONNX.ONNX2Mx
{
    public class ImportToGluon
    {
        public static SymbolBlock Import(string model_file, Context[] ctx)
        {
            throw new NotImplementedRelease2Exception();
        }
        
        public static SymbolBlock Import(string model_file, Context ctx)
        {
            return Import(model_file, new Context[] {ctx});
        }
    }
}