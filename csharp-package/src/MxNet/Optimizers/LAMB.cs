using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Optimizers
{
    public class LAMB : Optimizer
    {
        public LAMB(float learning_rate = 0.001f,
                float beta1 = 0.9f,
                float beta2 = 0.999f,
                float epsilon = 1E-06f,
                float? lower_bound = null,
                float? upper_bound = null,
                bool bias_correction = true,
                int aggregate_num = 4,
                bool use_fused_step = true)
        {
            throw new NotImplementedException();
        }

        public override NDArrayDict CreateState(int index, ndarray weight)
        {
            throw new NotImplementedException();
        }

        public override void FusedStep(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            throw new NotImplementedException();
        }

        public override void Step(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            throw new NotImplementedException();
        }
    }
}
