using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Contrib
{
    public class _LayerHistogramCollector : CalibrationCollector
    {
        public _LayerHistogramCollector(DType quantized_dtype, int num_bins= 8001, string[] include_layers= null)
        {
            throw new NotImplementedRelease2Exception();
        }

        public override void Collect(string name, string op_name, NDArray arr)
        {
            throw new NotImplementedRelease2Exception();
        }

        public override Dictionary<string, float> PostCollect()
        {
            throw new NotImplementedRelease2Exception();
        }

        public static (float[], NDArray, float, float) CombineHistory(float[] old_hist, NDArray arr, float new_min, float new_max,float  new_th)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static (float, float, float, float) GetOptimalThreshold(float[] hist_data, DType quantized_dtype, int num_quantized_bins= 255)
        {
            throw new NotImplementedRelease2Exception();
        }

        public static (float, float, float, float) GetOptimalThresholds(Dictionary<string, float[]> hist_dict, DType quantized_dtype, int num_quantized_bins = 255)
        {
            throw new NotImplementedRelease2Exception();
        }
    }
}
