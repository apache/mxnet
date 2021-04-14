using MxNet.Gluon;
using MxNet.Gluon.Data;
using MxNet.IO;
using System;
using System.Collections.Generic;

namespace MxNet.Contrib
{
    public class Quantization
    {
        public Quantization()
        {
        }

        public static NDArrayDict QuantizeParams(Symbol qsym, NDArrayDict  @params, Dictionary<string, float> min_max_dict)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static (Symbol, string[]) QuantizeSymbol(Symbol sym, Context ctx, string[] excluded_symbols= null, string[] excluded_operators = null,
                     string[] offline_params= null, DType quantized_dtype= null, string quantize_mode= "smart", string quantize_granularity= "tensor-wise")
        {
            throw new NotImplementedRelease2Exception();
        }

        public static Symbol CalibrateQuantizedSym(Symbol qsym, Dictionary<string, float>  min_max_dict)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static int CalibrateQuantizedSym(SymbolBlock sym_block, DataLoader data, CalibrationCollector collector,
                        int num_inputs, int? num_calib_batches= null)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static DataDesc[] GenerateListOfDataDesc(Shape[] data_shapes, DType[] data_types)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static (Symbol, bool, NDArrayDict, NDArrayDict) QuantizeModel(Symbol sym, NDArrayDict arg_params, NDArrayDict aux_params, string[] data_names= null,
                   Context ctx= null, string[] excluded_sym_names= null, string[] excluded_op_names= null, string calib_mode= "entropy",
                   DataLoader calib_data= null, int? num_calib_batches= null, DType quantized_dtype = null, string quantize_mode= "smart",
                   string quantize_granularity= "tensor-wise")
        {
            throw new NotImplementedRelease2Exception();
        }

        public static (Symbol, NDArrayDict, NDArrayDict) QuantizeModelMklDnn(Symbol sym, NDArrayDict arg_params, NDArrayDict aux_params, string[] data_names = null,
                   Context ctx = null, string[] excluded_sym_names = null, string[] excluded_op_names = null, string calib_mode = "entropy",
                   DataLoader calib_data = null, int? num_calib_batches = null, DType quantized_dtype = null, string quantize_mode = "smart",
                   string quantize_granularity = "tensor-wise")
        {
            throw new NotImplementedRelease2Exception();
        }

        public static (Symbol, NDArrayDict, NDArrayDict, CalibrationCollector) QuantizeGraph(Symbol sym, NDArrayDict arg_params, NDArrayDict aux_params,
                 Context ctx = null, string[] excluded_sym_names = null, string[] excluded_op_names = null, string calib_mode = "entropy",
                 DataLoader calib_data = null, int? num_calib_batches = null, DType quantized_dtype = null, string quantize_mode = "full",
                 string quantize_granularity = "tensor-wise", CalibrationCollector LayerOutputCollector = null)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static (Symbol, NDArrayDict, NDArrayDict) CalibGraph(Symbol qsym, NDArrayDict arg_params, NDArrayDict aux_params, CalibrationCollector collector, string calib_mode= "entropy")
        {
            throw new NotImplementedRelease2Exception();
        }

        public static (Symbol, NDArrayDict, NDArrayDict, CalibrationCollector) QuantizeNet(HybridBlock network, 
                    string quantized_dtype= "auto", string quantize_mode= "full", string quantize_granularity= "tensor-wise",
                    string[] exclude_layers= null, string[] exclude_layers_match= null, string[] exclude_operators= null,
                    DataLoader calib_data= null, DataDesc[] data_shapes= null, string calib_mode= "none",
                    int? num_calib_batches= null, Context ctx= null, CalibrationCollector LayerOutputCollector= null)
        {
            throw new NotImplementedRelease2Exception();
        }
    }
}
