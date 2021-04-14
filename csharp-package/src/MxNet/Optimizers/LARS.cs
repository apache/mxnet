using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Optimizers
{
    public class LARS : Optimizer
    {
        public LARS(float learning_rate = 0.1f,
                float momentum = 0,
                float eta = 0.001f,
                float epsilon = 1E-08f,
                bool lazy_update = false,
                bool use_fused_step = true,
                int aggregate_num = 4)
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
