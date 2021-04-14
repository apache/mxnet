using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Contrib
{
    public class _LayerOutputMinMaxCollector : CalibrationCollector
    {
        public _LayerOutputMinMaxCollector(DType quantized_dtype, string[] include_layers = null)
        {
            throw new NotImplementedRelease2Exception();
        }

        public override void Collect(string name, string op_name, NDArray arr)
        {
            throw new NotImplementedException();
        }
    }
}
