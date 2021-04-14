using System;
namespace MxNet.Contrib.ONNX.Mx2ONNX
{
    public class ExportModel
    {
        public static string Export(Symbol sym, NDArrayDict @params, Shape input_shape, DType input_type= null,
                 string onnx_file_path= "model.onnx", bool verbose= false, int? opset_version= null)
        {
            throw new NotImplementedRelease2Exception();
        }
    }
}
